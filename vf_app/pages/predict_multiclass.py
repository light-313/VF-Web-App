import time
from typing import Any, Dict

import numpy as np
import streamlit as st

from ..files import save_upload_to_temp
from ..runner import run_job
from ..state import upsert_job
from ..types import Job
from ..ui import ICON_ADD, ICON_MC, ICON_RUN, download_buttons
from ..analysis_utils import make_share_payload


def render() -> None:
    st.header(f"{ICON_MC} 多分类预测")
    st.caption("输出 pred 与每一类概率 probs[N,C]")

    with st.container():
        st.markdown("<div class='vf-card'>", unsafe_allow_html=True)
        method = st.selectbox(
            "选择模型",
            options=[
                "majority_vote_ensemble12（esm/prot/both 三模型投票）",
                "both_model12（单 both 模型）",
            ],
            index=0,
        )

        st.caption("可选：上传后将覆盖默认 test_esm_path / test_prot_path")
        col_a, col_b = st.columns(2)
        with col_a:
            esm_h5 = st.file_uploader("上传 ESM test h5（可选）", type=["h5", "hdf5"], key="mc_esm")
        with col_b:
            prot_h5 = st.file_uploader("上传 ProtT5 test h5（可选）", type=["h5", "hdf5"], key="mc_prot")

        col_c, col_d = st.columns(2)
        with col_c:
            batch_size = int(st.number_input("batch_size", min_value=1, max_value=4096, value=128, step=1))
        with col_d:
            device = st.selectbox("device", options=["auto", "cpu", "cuda"], index=0, key="mc_device")
        st.markdown("</div>", unsafe_allow_html=True)

    col_run, col_queue = st.columns(2)
    run_now = col_run.button(f"{ICON_RUN} 立即运行", type="primary", key="mc_run")
    addq = col_queue.button(f"{ICON_ADD} 加入任务队列", key="mc_add")
    if not (run_now or addq):
        return

    try:
        esm_path = save_upload_to_temp(esm_h5, suffix="_esm.h5") if esm_h5 is not None else None
        prot_path = save_upload_to_temp(prot_h5, suffix="_prot.h5") if prot_h5 is not None else None

        job_type = "multiclass_ensemble12" if method.startswith("majority_vote_ensemble12") else "multiclass_both12"

        job = Job(
            job_id=f"job_mc_{int(time.time() * 1000)}",
            job_type=job_type,
            created_at=time.time(),
            status="queued",
            params={
                "esm_path": esm_path,
                "prot_path": prot_path,
                "batch_size": int(batch_size),
                "device": device,
            },
        )
        upsert_job(job)
        st.success(f"已创建任务：{job.job_id}")

        if run_now:
            job = run_job(job)
            if job.status == "done" and job.result_df is not None:
                probs_arr = np.asarray(job.result_json.get("probs")) if job.result_json else None
                classes = int(probs_arr.shape[1]) if isinstance(probs_arr, np.ndarray) and probs_arr.ndim == 2 else None
                st.success(f"完成：N={len(job.result_df)}" + (f"，classes={classes}" if classes else ""))
                st.dataframe(job.result_df, use_container_width=True, height=520)
                payload = make_share_payload(job)
                download_buttons(job.result_df, payload, prefix="multiclass")
            else:
                st.error(f"失败：{job.error}")
        else:
            st.info("已加入队列：请到『任务队列』页面运行")

    except Exception as e:
        st.exception(e)
