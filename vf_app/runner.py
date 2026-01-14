from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import streamlit as st
import h5py
import time

from m_maxvote import predict_both_model12, predict_majority_vote_ensemble12
from s_maxvote import run_majority_vote_4models

from .analysis_utils import make_binary_df, make_multiclass_df
from .state import upsert_job
from .types import Job


def _try_load_sequences(h5_path: Optional[str]) -> Dict[str, str]:
    if not h5_path:
        return {}
    try:
        with h5py.File(h5_path, "r") as f:
            if "sequences" not in f:
                return {}
            grp = f["sequences"]
            out: Dict[str, str] = {}
            for k in grp.keys():
                raw = grp[k][()]
                out[str(k)] = raw.decode("ascii") if isinstance(raw, (bytes, bytearray)) else str(raw)
            return out
    except Exception:
        return {}


def run_job(job: Job) -> Job:
    job.status = "running"
    job.error = None
    upsert_job(job)

    try:
        t0 = time.perf_counter()
        if job.job_type == "binary":
            cfg = job.params["cfg"]
            batch_size = int(job.params.get("batch_size", 64))
            device = str(job.params.get("device", "auto"))
            save_per_model_prob = bool(job.params.get("save_per_model_prob", False))

            with st.spinner("正在推理（二分类）..."):
                res = run_majority_vote_4models(
                    cfg,
                    batch_size=batch_size,
                    device=device,
                    out_path=None,
                    save_per_model_prob=save_per_model_prob,
                )

            df = make_binary_df(res["seq_ids"], res["pred"], res["probs"])

            # 尝试补齐 sequence 列（方便 FASTA 导出）
            seq_map = _try_load_sequences(cfg.get("test_esm_path"))
            if not seq_map:
                seq_map = _try_load_sequences(cfg.get("test_prot5_path"))
            if seq_map:
                df["sequence"] = [seq_map.get(str(sid), "") for sid in df["seq_id"].astype(str).tolist()]
            if isinstance(res.get("results"), list) and res["results"] and "label" in res["results"][0]:
                labels = [int(r.get("label", -1)) for r in res["results"]]
                if len(labels) == len(df):
                    df.insert(1, "label", labels)

            job.result_df = df
            job.result_json = {
                "seq_ids": res.get("seq_ids"),
                "pred": res.get("pred"),
                "probs": res.get("probs"),
                "metrics": res.get("metrics", {}),
            }

        elif job.job_type in {"multiclass_ensemble12", "multiclass_both12"}:
            batch_size = int(job.params.get("batch_size", 128))
            device = str(job.params.get("device", "auto"))
            kwargs: Dict[str, Any] = {"batch_size": batch_size, "device": device}
            if job.params.get("esm_path"):
                kwargs["esm_path"] = job.params["esm_path"]
            if job.params.get("prot_path"):
                kwargs["prot_path"] = job.params["prot_path"]

            with st.spinner("正在推理（多分类）..."):
                if job.job_type == "multiclass_ensemble12":
                    seq_ids, pred, probs = predict_majority_vote_ensemble12(**kwargs)
                else:
                    seq_ids, pred, probs = predict_both_model12(**kwargs)

            df = make_multiclass_df(seq_ids, pred, probs)

            seq_map = _try_load_sequences(job.params.get("esm_path"))
            if not seq_map:
                seq_map = _try_load_sequences(job.params.get("prot_path"))
            if seq_map:
                df["sequence"] = [seq_map.get(str(sid), "") for sid in df["seq_id"].astype(str).tolist()]
            job.result_df = df
            job.result_json = {
                "seq_ids": list(seq_ids),
                "pred": np.asarray(pred).astype(int).tolist(),
                "probs": np.asarray(probs).astype(float).tolist(),
            }

        else:
            raise ValueError(f"Unknown job_type: {job.job_type}")

        elapsed = float(time.perf_counter() - t0)

        if job.result_json is None:
            job.result_json = {}
        job.result_json["timing_sec"] = elapsed
        st.session_state["last_timing_sec"] = elapsed

        job.status = "done"
        upsert_job(job)
        return job

    except Exception as e:
        job.status = "error"
        job.error = str(e)
        upsert_job(job)
        return job
