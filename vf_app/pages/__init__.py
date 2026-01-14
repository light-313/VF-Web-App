"""Streamlit pages (one file per page)."""

from . import analysis, dashboard, home, pipeline, predict_binary, predict_multiclass, queue

__all__ = [
    "analysis",
    "dashboard",
    "home",
    "pipeline",
    "predict_binary",
    "predict_multiclass",
    "queue",
]
