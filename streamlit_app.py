import os
import sys

import streamlit as st


_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from vf_app.files import ensure_project_cwd
from vf_app.state import init_state, job_list
from vf_app.ui import (
    APP_TITLE,
    ICON_ANALYSIS,
    ICON_BIN,
    ICON_DASH,
    ICON_HOME,
    ICON_MC,
    ICON_QUEUE,
    apply_base_style,
)
from vf_app.pages import analysis, dashboard, home, pipeline, predict_binary, predict_multiclass, queue


def _sidebar_resource_panel() -> None:
    st.markdown("---")
    st.markdown("**性能监控**")
    last_t = st.session_state.get("last_timing_sec")
    if isinstance(last_t, (int, float)):
        st.caption(f"最近一次推理耗时：{float(last_t):.3f}s")

    try:
        import psutil  # type: ignore

        cpu = psutil.cpu_percent(interval=0.0)
        mem = psutil.virtual_memory()
        st.caption(f"CPU：{cpu:.1f}% | 内存：{mem.percent:.1f}%")
    except Exception:
        st.caption("CPU/内存：需要安装 psutil")


def main() -> None:
    ensure_project_cwd(_ROOT)

    st.set_page_config(page_title=APP_TITLE, layout="wide")
    apply_base_style()
    init_state()

    st.title("🦠 VF-Web: 细菌毒力因子深度学习在线预测平台")
    

    with st.sidebar:
        st.markdown("### 🧭 导航")
        page = st.radio(
            "page",
            options=[
                f"{ICON_HOME} 介绍",
                "🧩 Pipeline",
                f"{ICON_QUEUE} 任务队列",
                f"{ICON_BIN} 二分类",
                f"{ICON_MC} 多分类",
                f"{ICON_DASH} 整体展示",
                f"{ICON_ANALYSIS} 结果分析",
            ],
            index=0,
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown("**队列状态**")
        jobs = job_list()
        st.caption(
            f"总任务：{len(jobs)} | 待运行：{sum(1 for j in jobs if j.status=='queued')} | 已完成：{sum(1 for j in jobs if j.status=='done')} | 失败：{sum(1 for j in jobs if j.status=='error')}"
        )

        _sidebar_resource_panel()

    if page.endswith("介绍"):
        home.render()
    elif page.endswith("Pipeline"):
        pipeline.render(project_root=_ROOT)

    elif page.endswith("二分类"):
        predict_binary.render(project_root=_ROOT)
    elif page.endswith("多分类"):
        predict_multiclass.render()
    elif page.endswith("整体展示"):
        dashboard.render()
    elif page.endswith("任务队列"):
        queue.render()
    else:
        analysis.render()


if __name__ == "__main__":
    main()
