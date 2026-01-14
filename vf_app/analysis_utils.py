from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix

from .types import Job


def _get_plotly():
    """Return (px, go) if plotly is installed, else (None, None)."""
    try:
        import plotly.express as px  # type: ignore
        import plotly.graph_objects as go  # type: ignore

        return px, go
    except Exception:
        return None, None


def _warn_plotly_missing() -> None:
    key = "_vf_plotly_missing_warned"
    if st.session_state.get(key):
        return
    st.session_state[key] = True
    st.info("未安装 plotly：图表将回退为基础版本。建议安装 `plotly>=5.22.0` 以获得可交互图表。")


def _plotly_chart(fig) -> None:
    """Render plotly figure with broad Streamlit version compatibility."""
    try:
        st.plotly_chart(fig, use_container_width=True)
    except TypeError:
        st.plotly_chart(fig)


def analyze_common(df: pd.DataFrame, pred_col: str = "pred", conf_col: str = "confidence") -> None:
    if df is None or df.empty:
        st.warning("没有可分析的数据")
        return

    preds = df[pred_col].astype(int)
    counts = preds.value_counts().sort_index()
    total = int(len(df))

    st.markdown("### 类别占比")
    prop = pd.DataFrame({"class": counts.index.astype(int), "count": counts.values.astype(int)})
    prop["ratio"] = (prop["count"] / max(total, 1)).astype(float)
    st.dataframe(prop, use_container_width=True)

    px, _ = _get_plotly()
    if px is not None:
        fig = px.bar(
            prop,
            x="class",
            y="ratio",
            hover_data={"count": True, "ratio": ":.4f", "class": True},
            labels={"class": "class", "ratio": "ratio"},
            title=None,
        )
        fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
        _plotly_chart(fig)
    else:
        _warn_plotly_missing()
        st.bar_chart(prop.set_index("class")["ratio"], height=200)

    st.markdown("### 置信度分布")
    if conf_col in df.columns:
        conf = df[conf_col].astype(float)
        px, _ = _get_plotly()
        if px is not None:
            conf_df = pd.DataFrame({"confidence": conf})
            fig = px.histogram(conf_df, x="confidence", nbins=50, labels={"confidence": "confidence"})
            fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
            _plotly_chart(fig)
        else:
            _warn_plotly_missing()
            st.line_chart(conf.sort_values(ignore_index=True), height=200)
    else:
        st.info("当前结果没有 confidence 列，跳过置信度分布")


def top_confidence(df: pd.DataFrame, top_k: int = 30) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    k = int(min(max(top_k, 1), len(df)))
    if "confidence" not in df.columns:
        return df.head(k).reset_index(drop=True)
    return df.sort_values("confidence", ascending=False).head(k).reset_index(drop=True)


def binary_confusion(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if "label" not in df.columns:
        return None
    y_true = df["label"].astype(int).values
    y_pred = df["pred"].astype(int).values
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"])


def binary_confusion_fig(df: pd.DataFrame):
    """Interactive confusion matrix (plotly). Returns None if not available or missing label."""
    if "label" not in df.columns:
        return None
    px, go = _get_plotly()
    if go is None:
        _warn_plotly_missing()
        return None

    y_true = df["label"].astype(int).values
    y_pred = df["pred"].astype(int).values
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    z = cm.astype(int)
    x = ["pred_0", "pred_1"]
    y = ["true_0", "true_1"]

    fig = go.Figure(
        data=
            go.Heatmap(
                z=z,
                x=x,
                y=y,
                colorscale="Blues",
                showscale=True,
                hovertemplate="%{y} → %{x}<br>count=%{z}<extra></extra>",
            )
    )

    # annotate
    annotations = []
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            annotations.append(
                dict(
                    x=x[j],
                    y=y[i],
                    text=str(int(z[i, j])),
                    showarrow=False,
                    font=dict(color="black"),
                )
            )
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title=None,
        yaxis_title=None,
        annotations=annotations,
    )
    return fig


def make_share_payload(job: Job) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "job_id": job.job_id,
        "job_type": job.job_type,
        "created_at": job.created_at,
        "status": job.status,
    }
    if job.result_json is not None:
        payload["result"] = job.result_json
    if job.result_df is not None:
        core_cols = [c for c in ["seq_id", "pred", "confidence", "prob1", "max_prob"] if c in job.result_df.columns]
        payload["top_confidence"] = top_confidence(job.result_df[core_cols].copy(), top_k=30).to_dict(orient="records")
    if job.error:
        payload["error"] = job.error
    return payload


def make_binary_df(seq_ids, pred, probs) -> pd.DataFrame:
    df = pd.DataFrame({"seq_id": list(seq_ids), "pred": list(pred), "prob1": list(probs)})
    p = df["prob1"].astype(float)
    pred_s = df["pred"].astype(int)
    conf = np.where(pred_s.values == 1, p.values, 1.0 - p.values)
    df["confidence"] = conf.astype(float)
    return df


def make_multiclass_df(seq_ids, pred, probs) -> pd.DataFrame:
    df = pd.DataFrame({"seq_id": list(seq_ids), "pred": np.asarray(pred).astype(int)})
    probs = np.asarray(probs)
    if probs.ndim == 2:
        for c in range(int(probs.shape[1])):
            df[f"prob_c{c}"] = probs[:, c].astype(float)
        df["max_prob"] = probs.max(axis=1).astype(float)
        df["confidence"] = df["max_prob"].astype(float)
    else:
        df["prob"] = probs.astype(float)
        df["confidence"] = probs.astype(float)
    return df
