import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from ..h5_utils import list_embedding_keys, make_temp_subset
from ..runner import run_job
from ..state import upsert_job
from ..types import Job
from ..ui import ICON_BIN, ICON_MC, ICON_RUN, download_buttons


def _load_default_binary_cfg(project_root: str) -> Dict[str, Any]:
    cfg_path = os.path.join(project_root, "best_check", "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _make_pipeline_table(binary_df: pd.DataFrame, multiclass_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    out = binary_df.copy()
    out = out.rename(columns={"pred": "binary_pred"})
    if multiclass_df is None or multiclass_df.empty:
        out["multiclass_pred"] = None
        return out

    m = multiclass_df[[c for c in multiclass_df.columns if c in {"seq_id", "pred", "confidence", "max_prob", "sequence"} or c.startswith("prob_c")]].copy()
    m = m.rename(columns={"pred": "multiclass_pred", "confidence": "multiclass_confidence"})

    # 优先保留 binary 的 sequence；若为空再用 multiclass 的
    if "sequence" in out.columns and "sequence" in m.columns:
        m = m.rename(columns={"sequence": "sequence_mc"})

    out = out.merge(m, on="seq_id", how="left")
    if "sequence" in out.columns and "sequence_mc" in out.columns:
        out["sequence"] = out["sequence"].where(out["sequence"].astype(str).str.len() > 0, out["sequence_mc"])
        out = out.drop(columns=["sequence_mc"])

    return out


def render(project_root: str) -> None:
    st.header("🧩 Pipeline：二分类 →（阳性）→ 多分类")
    st.caption("一键完成：二分类筛阳性样本，再对阳性做多分类功能注释（输入为已提取 embedding 的 h5）")

    with st.container():
        st.markdown("<div class='vf-card'>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            esm_h5 = st.file_uploader("上传 ESM embedding h5", type=["h5", "hdf5"], key="pipe_esm")
        with col2:
            prot_h5 = st.file_uploader("上传 Prot/ProtT5 embedding h5", type=["h5", "hdf5"], key="pipe_prot")

        st.markdown("**多分类模型选择**")
        method = st.selectbox(
            "模型",
            options=[
                "majority_vote_ensemble12（esm/prot/both 三模型投票）",
                "both_model12（单 both 模型）",
            ],
            index=0,
            key="pipe_method",
            label_visibility="collapsed",
        )

        st.markdown("**运行参数**")
        c3, c4, c5 = st.columns(3)
        with c3:
            device = st.selectbox("device", options=[ "cpu", "cuda"], index=0, key="pipe_device")
        with c4:
            bs_bin = int(st.number_input("二分类 batch_size", min_value=1, max_value=4096, value=64, step=1))
        with c5:
            bs_mc = int(st.number_input("多分类 batch_size", min_value=1, max_value=4096, value=128, step=1))

        thr = float(
            st.slider(
                "二分类阈值（prob1 >= threshold 视为阳性）",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01,
            )
        )

        st.markdown("</div>", unsafe_allow_html=True)

    run_now = st.button(f"{ICON_RUN} 一键运行 Pipeline", type="primary")
    if not run_now:
        return

    if esm_h5 is None or prot_h5 is None:
        st.error("请先上传 ESM 与 Prot embedding h5")
        return

    # 保存上传文件到临时路径（Streamlit UploadedFile 不是文件路径）
    esm_path = _save_upload(esm_h5, suffix="_esm.h5")
    prot_path = _save_upload(prot_h5, suffix="_prot.h5")

    # 1) 二分类
    cfg = _load_default_binary_cfg(project_root)
    cfg["test_esm_path"] = esm_path
    cfg["test_prot5_path"] = prot_path

    job_bin = Job(
        job_id=f"job_pipe_bin_{int(time.time() * 1000)}",
        job_type="binary",
        created_at=time.time(),
        status="queued",
        params={
            "cfg": cfg,
            "batch_size": int(bs_bin),
            "device": device,
            "save_per_model_prob": False,
        },
    )
    upsert_job(job_bin)

    st.markdown(f"### {ICON_BIN} Step 1/2：二分类")
    job_bin = run_job(job_bin)
    if job_bin.status != "done" or job_bin.result_df is None:
        st.error(f"二分类失败：{job_bin.error}")
        return

    bin_df = job_bin.result_df.copy()
    if "prob1" not in bin_df.columns:
        st.error("二分类结果缺少 prob1 列，无法按阈值筛选")
        return

    bin_df["pred"] = (bin_df["prob1"].astype(float) >= thr).astype(int)
    p = bin_df["prob1"].astype(float)
    bin_df["confidence"] = (p.where(bin_df["pred"] == 1, 1.0 - p)).astype(float)

    pos_ids = bin_df.loc[bin_df["pred"] == 1, "seq_id"].astype(str).tolist()
    st.success(f"二分类完成：总数 {len(bin_df)} | 阳性 {len(pos_ids)}（阈值={thr:.2f}）")
    st.dataframe(bin_df, use_container_width=True, height=260)

    if not pos_ids:
        st.warning("阳性样本为 0：Pipeline 到此结束（无需多分类）")
        payload = {
            "pipeline": {"threshold": thr, "positive": 0, "total": int(len(bin_df))},
            "binary": job_bin.result_json,
        }
        download_buttons(bin_df, payload, prefix="pipeline")
        return

    # 2) 多分类：只对阳性子集跑（同时保证 esm/prot 都存在的 id）
    st.markdown(f"### {ICON_MC} Step 2/2：多分类（仅阳性子集）")

    esm_keys = list_embedding_keys(esm_path)
    prot_keys = list_embedding_keys(prot_path)
    common_pos = [sid for sid in pos_ids if sid in esm_keys and sid in prot_keys]

    if not common_pos:
        st.error("阳性样本在 ESM/Prot 的 embedding 中无法对齐（交集为空）")
        return

    sub_esm = make_temp_subset(esm_path, common_pos, suffix="_pos_esm.h5")
    sub_prot = make_temp_subset(prot_path, common_pos, suffix="_pos_prot.h5")

    job_type = "multiclass_ensemble12" if method.startswith("majority_vote_ensemble12") else "multiclass_both12"
    job_mc = Job(
        job_id=f"job_pipe_mc_{int(time.time() * 1000)}",
        job_type=job_type,
        created_at=time.time(),
        status="queued",
        params={
            "esm_path": sub_esm,
            "prot_path": sub_prot,
            "batch_size": int(bs_mc),
            "device": device,
        },
    )
    upsert_job(job_mc)

    job_mc = run_job(job_mc)
    if job_mc.status != "done" or job_mc.result_df is None:
        st.error(f"多分类失败：{job_mc.error}")
        return

    mc_df = job_mc.result_df.copy()
    st.success(f"多分类完成：N={len(mc_df)}")
    st.dataframe(mc_df, use_container_width=True, height=260)

    # 汇总表 + 下载
    st.markdown("### 汇总")
    combined = _make_pipeline_table(bin_df, mc_df)
    st.dataframe(combined, use_container_width=True, height=420)

    payload = {
        "pipeline": {
            "threshold": thr,
            "total": int(len(bin_df)),
            "positive": int(len(pos_ids)),
            "positive_aligned": int(len(common_pos)),
            "multiclass_method": job_type,
        },
        "binary": job_bin.result_json,
        "multiclass": job_mc.result_json,
    }
    download_buttons(combined, payload, prefix="pipeline")


def _save_upload(uploaded_file, suffix: str) -> str:
    # local helper to avoid circular import
    import tempfile

    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path
