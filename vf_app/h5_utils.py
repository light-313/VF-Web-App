from __future__ import annotations

import os
import tempfile
from typing import Iterable, List, Optional, Sequence, Set

import h5py


def list_embedding_keys(h5_path: str) -> Set[str]:
    with h5py.File(h5_path, "r") as f:
        if "embeddings" not in f:
            return set()
        return {str(k) for k in f["embeddings"].keys()}


def write_subset_h5(src_path: str, dst_path: str, keep_ids: Sequence[str], default_label: int = 0) -> int:
    """Write a subset of an embedding H5 (embeddings/labels[/sequences]) to dst_path.

    Returns number of sequences written.
    """
    written = 0
    keep_ids = [str(x) for x in keep_ids]

    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        emb_src = src.get("embeddings")
        if emb_src is None:
            raise ValueError(f"H5 缺少 embeddings 组: {src_path}")

        emb_dst = dst.create_group("embeddings")
        seq_dst = dst.create_group("sequences") if "sequences" in src else None
        lbl_dst = dst.create_group("labels")

        has_seq = "sequences" in src
        has_lbl = "labels" in src

        for sid in keep_ids:
            if sid not in emb_src:
                continue

            src.copy(emb_src[sid], emb_dst, name=sid)

            if has_seq and seq_dst is not None:
                raw = src["sequences"][sid][()]
                seq_dst.create_dataset(sid, data=raw)

            if has_lbl:
                raw_lbl = src["labels"][sid][()]
                lbl_dst.create_dataset(sid, data=raw_lbl)
            else:
                lbl_dst.create_dataset(sid, data=int(default_label))

            written += 1

        dst.attrs["total_sequences"] = int(written)

    return written


def make_temp_subset(src_path: str, keep_ids: Sequence[str], suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    write_subset_h5(src_path, path, keep_ids)
    return path
