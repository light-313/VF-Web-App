from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import pandas as pd


JobType = Literal["binary", "multiclass_ensemble12", "multiclass_both12"]
JobStatus = Literal["queued", "running", "done", "error"]


@dataclass
class Job:
    job_id: str
    job_type: JobType
    created_at: float
    status: JobStatus
    params: Dict[str, Any]
    error: Optional[str] = None
    result_df: Optional[pd.DataFrame] = None
    result_json: Optional[Dict[str, Any]] = None
