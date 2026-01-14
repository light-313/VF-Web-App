import streamlit as st

from ..analysis_utils import analyze_common, binary_confusion, binary_confusion_fig, make_share_payload, top_confidence
from ..state import find_job, job_list, selected_job_id
from ..ui import ICON_ANALYSIS, ICON_DOWNLOAD, download_buttons


def _get_plotly():
    try:
        import plotly.express as px  # type: ignore

        return px
    except Exception:
        return None


def _plotly_chart(fig) -> None:
    try:
        st.plotly_chart(fig, use_container_width=True)
    except TypeError:
        st.plotly_chart(fig)


def render() -> None:
    st.header(f"{ICON_ANALYSIS} 结果分析")

    done_jobs = [j for j in job_list() if j.status == "done" and j.result_df is not None]
    if not done_jobs:
        st.info("暂无可分析结果。")
        return

    default_id = selected_job_id()
    options = [j.job_id for j in done_jobs]
    idx = options.index(default_id) if default_id in options else 0

    job_id = st.selectbox("选择任务", options=options, index=idx)
    job = find_job(job_id)
    if job is None or job.result_df is None:
        st.warning("任务不存在或无结果")
        return

    df = job.result_df.copy()

    # 二分类阈值动态调节：不重新推理，只更新 pred/统计
    if job.job_type == "binary" and "prob1" in df.columns:
        st.markdown("### 阈值调节")
        thr = float(
            st.slider(
                "判定阈值（prob1 >= threshold 视为阳性）",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01,
            )
        )
        df["pred"] = (df["prob1"].astype(float) >= thr).astype(int)
        p = df["prob1"].astype(float)
        df["confidence"] = (p.where(df["pred"] == 1, 1.0 - p)).astype(float)
    st.markdown(
        """
        <div class="vf-card">
          <div style="font-weight:600; margin-bottom:0.25rem;">任务信息</div>
          <div class="vf-muted vf-small">job_id: {job_id} | type: {t} | rows: {n}</div>
        </div>
        """.format(job_id=job.job_id, t=job.job_type, n=len(df)),
        unsafe_allow_html=True,
    )

    if job.job_type == "binary" and job.result_json and isinstance(job.result_json.get("metrics"), dict):
        m = job.result_json["metrics"]
        if m:
            st.markdown(
                """
                <div class="vf-card">
                  <div style="font-weight:600; margin-bottom:0.25rem;">二分类指标</div>
                  <div class="vf-muted vf-small">SN: {sn:.2f}% | SP: {sp:.2f}% | ACC: {acc:.2f}% | F1: {f1:.2f}% | MCC: {mcc:.2f}% | AUC: {auc:.2f}% | AUPR: {aupr:.2f}%</div>
                </div>
                """.format(
                    sn=float(m.get("sn", 0.0)),
                    sp=float(m.get("sp", 0.0)),
                    acc=float(m.get("acc", 0.0)),
                    f1=float(m.get("f1", 0.0)),
                    mcc=float(m.get("mcc", 0.0)),
                    auc=float(m.get("auc", 0.0)),
                    aupr=float(m.get("aupr", 0.0)),
                ),
                unsafe_allow_html=True,
            )

    analyze_common(df)

    st.markdown("### 高置信度样本")
    top_k = int(st.slider("Top-K", min_value=10, max_value=200, value=30, step=10))
    top_df = top_confidence(df, top_k=top_k)

    px = _get_plotly()
    if px is not None and not top_df.empty:
        plot_df = top_df.copy()
        if "seq_id" in plot_df.columns:
            plot_df["seq_id"] = plot_df["seq_id"].astype(str)
        if "pred" in plot_df.columns:
            plot_df["pred"] = plot_df["pred"].astype(str)

        y_col = "confidence" if "confidence" in plot_df.columns else None
        if y_col is not None:
            fig = px.scatter(
                plot_df.reset_index(drop=True),
                x=plot_df.reset_index(drop=True).index,
                y=y_col,
                color="pred" if "pred" in plot_df.columns else None,
                hover_data=[c for c in ["seq_id", "prob1", "max_prob", "label"] if c in plot_df.columns],
                labels={"x": "rank (Top-K)", y_col: y_col},
            )
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
            _plotly_chart(fig)
    elif px is None:
        st.info("未安装 plotly，Top-K 置信度交互散点图已跳过。安装 `plotly>=5.22.0` 后可用。")

    st.dataframe(top_df, use_container_width=True, height=420)

    if job.job_type == "binary":
        st.markdown("### 混淆矩阵（若存在 label）")
        cm_fig = binary_confusion_fig(df)
        if cm_fig is not None:
            _plotly_chart(cm_fig)
        else:
            cm_df = binary_confusion(df)
            if cm_df is not None:
                st.dataframe(cm_df, use_container_width=True)

    st.markdown("---")
    st.markdown(f"### {ICON_DOWNLOAD} 下载 ")
    payload = make_share_payload(job)
    download_buttons(df, payload, prefix=job.job_id)
