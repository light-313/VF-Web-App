import json
import os
import time
from typing import Any, Dict

import streamlit as st

from ..files import save_upload_to_temp
from ..runner import run_job
from ..state import upsert_job
from ..types import Job
from ..ui import ICON_ADD, ICON_BIN, ICON_RUN, download_buttons
from ..analysis_utils import make_share_payload
from ..fasta_utils import length_stats, parse_fasta_text, validate_records


def render(project_root: str) -> None:
    st.header(f"{ICON_BIN} 二分类预测")
    st.caption("4 个模型加权投票：输出 pred 与 P(class=1)")

    with st.container():
        st.markdown("<div class='vf-card'>", unsafe_allow_html=True)
        use_default = st.checkbox("使用默认 config（best_check/config.json）", value=True)
        config_upload = None if use_default else st.file_uploader("上传 config.json", type=["json"])

        col_a, col_b = st.columns(2)
        with col_a:
            esm_h5 = st.file_uploader("上传 ESM test h5（可选）", type=["h5", "hdf5"], key="bin_esm")
        with col_b:
            prot_h5 = st.file_uploader("上传 ProtT5 test h5（可选）", type=["h5", "hdf5"], key="bin_prot")

        col_c, col_d = st.columns(2)
        with col_c:
            batch_size = int(st.number_input("batch_size", min_value=1, max_value=4096, value=64, step=1))
        with col_d:
            device = st.selectbox("device", options=["cpu", "cuda"], index=0)

        save_per_model_prob = st.checkbox("JSON 中包含每个模型的 prob（体积较大）", value=False)
        st.markdown("</div>", unsafe_allow_html=True)

    col_run, col_queue = st.columns(2)
    run_now = col_run.button(f"{ICON_RUN} 立即运行", type="primary")
    addq = col_queue.button(f"{ICON_ADD} 加入任务队列")
    if not (run_now or addq):
        return

    try:
        if use_default:
            cfg_path = os.path.join(project_root, "best_check", "config.json")
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg: Dict[str, Any] = json.load(f)
        else:
            if config_upload is None:
                st.error("请上传 config.json 或勾选使用默认 config")
                return
            cfg = json.loads(config_upload.getvalue().decode("utf-8"))

        if esm_h5 is not None:
            cfg["test_esm_path"] = save_upload_to_temp(esm_h5, suffix="_esm.h5")
        if prot_h5 is not None:
            cfg["test_prot5_path"] = save_upload_to_temp(prot_h5, suffix="_prot.h5")

        job = Job(
            job_id=f"job_bin_{int(time.time() * 1000)}",
            job_type="binary",
            created_at=time.time(),
            status="queued",
            params={
                "cfg": cfg,
                "batch_size": int(batch_size),
                "device": device,
                "save_per_model_prob": bool(save_per_model_prob),
            },
        )
        upsert_job(job)
        st.success(f"已创建任务：{job.job_id}")

        if run_now:
            job = run_job(job)
            if job.status == "done" and job.result_df is not None:
                df = job.result_df.copy()
                st.success(f"完成：N={len(df)}")

                if "prob1" in df.columns:
                    thr = float(
                        st.slider(
                            "阈值（prob1 >= threshold 视为阳性）",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.5,
                            step=0.01,
                        )
                    )
                    df["pred"] = (df["prob1"].astype(float) >= thr).astype(int)
                    p = df["prob1"].astype(float)
                    df["confidence"] = (p.where(df["pred"] == 1, 1.0 - p)).astype(float)

                st.dataframe(df, use_container_width=True, height=520)
                payload = make_share_payload(job)
                download_buttons(df, payload, prefix="binary")
            else:
                st.error(f"失败：{job.error}")
        else:
            st.info("已加入队列：请到『任务队列』页面运行")

    except Exception as e:
        st.exception(e)
