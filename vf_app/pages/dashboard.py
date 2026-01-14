import streamlit as st

from ..state import job_list, summarize_jobs
from ..ui import ICON_BIN, ICON_DASH, ICON_MC


def render() -> None:
    st.header(f"{ICON_DASH} 整体展示（仪表盘）")
    st.caption("对本会话内已完成任务做汇总")

    done_jobs = [j for j in job_list() if j.status == "done" and j.result_df is not None]
    if not done_jobs:
        st.info("暂无已完成结果。请先在队列中运行任务。")
        return

    n_jobs = len(done_jobs)
    n_rows = int(sum(len(j.result_df) for j in done_jobs if j.result_df is not None))
    type_counts = {}
    for j in done_jobs:
        type_counts[j.job_type] = type_counts.get(j.job_type, 0) + 1

    st.markdown(
        """
        <div class="vf-kpi">
          <div><div class="vf-kpi-lab">✅ 已完成任务</div><div class="vf-kpi-num">{n_jobs}</div></div>
          <div><div class="vf-kpi-lab">📄 结果总行数</div><div class="vf-kpi-num">{n_rows}</div></div>
          <div><div class="vf-kpi-lab">{bin_icon} 二分类任务</div><div class="vf-kpi-num">{n_bin}</div></div>
          <div><div class="vf-kpi-lab">{mc_icon} 多分类任务</div><div class="vf-kpi-num">{n_mc}</div></div>
        </div>
        """.format(
            n_jobs=n_jobs,
            n_rows=n_rows,
            n_bin=type_counts.get("binary", 0),
            n_mc=type_counts.get("multiclass_ensemble12", 0) + type_counts.get("multiclass_both12", 0),
            bin_icon=ICON_BIN,
            mc_icon=ICON_MC,
        ),
        unsafe_allow_html=True,
    )

    st.markdown("### 任务列表")
    st.dataframe(summarize_jobs(done_jobs), use_container_width=True, height=260)
