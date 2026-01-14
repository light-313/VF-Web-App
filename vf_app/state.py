from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from .types import Job


def init_state() -> None:
    if "jobs" not in st.session_state:
        st.session_state.jobs = []
    if "selected_job_id" not in st.session_state:
        st.session_state.selected_job_id = None


def _coerce_job(obj: Any) -> Optional[Job]:
    # 兼容：旧代码中可能把 Job 放在 streamlit_app.Job；或者是 dict。
    if obj is None:
        return None

    if isinstance(obj, Job):
        return obj

    if isinstance(obj, dict) and "job_id" in obj and "job_type" in obj:
        return Job(
            job_id=str(obj.get("job_id")),
            job_type=obj.get("job_type"),
            created_at=float(obj.get("created_at", 0.0)),
            status=obj.get("status"),
            params=dict(obj.get("params", {})),
            error=obj.get("error"),
            result_df=obj.get("result_df"),
            result_json=obj.get("result_json"),
        )

    # duck-typing (legacy dataclass)
    if hasattr(obj, "job_id") and hasattr(obj, "job_type"):
        try:
            return Job(
                job_id=str(getattr(obj, "job_id")),
                job_type=getattr(obj, "job_type"),
                created_at=float(getattr(obj, "created_at")),
                status=getattr(obj, "status"),
                params=dict(getattr(obj, "params")),
                error=getattr(obj, "error", None),
                result_df=getattr(obj, "result_df", None),
                result_json=getattr(obj, "result_json", None),
            )
        except Exception:
            return None

    return None


def job_list() -> List[Job]:
    raw = st.session_state.jobs
    jobs: List[Job] = []
    changed = False
    for item in raw:
        j = _coerce_job(item)
        if j is not None:
            jobs.append(j)
            if not isinstance(item, Job):
                changed = True
    if changed:
        st.session_state.jobs = jobs
    return jobs


def set_job_list(jobs: List[Job]) -> None:
    st.session_state.jobs = jobs


def find_job(job_id: str) -> Optional[Job]:
    for j in job_list():
        if j.job_id == job_id:
            return j
    return None


def upsert_job(job: Job) -> None:
    jobs = job_list()
    for i, j in enumerate(jobs):
        if j.job_id == job.job_id:
            jobs[i] = job
            set_job_list(jobs)
            return
    jobs.append(job)
    set_job_list(jobs)


def select_job(job_id: Optional[str]) -> None:
    st.session_state.selected_job_id = job_id


def selected_job_id() -> Optional[str]:
    return st.session_state.selected_job_id


def summarize_jobs(jobs: List[Job]) -> pd.DataFrame:
    import time

    def row(job: Job) -> Dict[str, Any]:
        return {
            "job_id": job.job_id,
            "type": job.job_type,
            "status": job.status,
            "created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(job.created_at)),
            "n": int(len(job.result_df)) if job.result_df is not None else None,
            "error": job.error,
        }

    return pd.DataFrame([row(j) for j in jobs])
