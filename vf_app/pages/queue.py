import streamlit as st

from ..runner import run_job
from ..state import job_list, select_job, set_job_list, summarize_jobs
from ..ui import ICON_QUEUE, ICON_RUN


def render() -> None:
    st.header(f"{ICON_QUEUE} 任务队列")
    st.caption("支持把预测任务加入队列，按顺序执行并保留历史结果")

    jobs = job_list()
    if not jobs:
        st.info("当前没有任务。请在『二分类/多分类』页面添加任务到队列。")
        return

    st.dataframe(summarize_jobs(jobs), use_container_width=True, height=260)

    col1, col2, col3 = st.columns(3)
    with col1:
        run_next = st.button(f"{ICON_RUN} 运行下一个", type="primary")
    with col2:
        run_all = st.button(f"{ICON_RUN} 运行全部")
    with col3:
        clear = st.button("🧹 清空队列（仅本会话）", type="secondary")

    if clear:
        set_job_list([])
        select_job(None)
        st.rerun()

    def next_job():
        for j in job_list():
            if j.status == "queued":
                return j
        return None

    if run_next:
        j = next_job()
        if j is None:
            st.info("没有待运行任务")
        else:
            run_job(j)
            st.rerun()

    if run_all:
        any_run = False
        while True:
            j = next_job()
            if j is None:
                break
            any_run = True
            run_job(j)
        if not any_run:
            st.info("没有待运行任务")
        st.rerun()

    st.markdown("---")
    done_jobs = [j for j in job_list() if j.status == "done"]
    if done_jobs:
        st.markdown("### 快速查看")
        selected = st.selectbox("选择已完成任务", options=[j.job_id for j in done_jobs], index=0)
        select_job(selected)
        st.caption("在『结果分析』页面查看图表与高置信样本")
