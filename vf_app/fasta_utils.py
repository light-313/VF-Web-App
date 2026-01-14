from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


_ALLOWED_AA = set("ACDEFGHIKLMNPQRSTVWY") | {"X", "U", "O"}


@dataclass(frozen=True)
class FastaRecord:
    seq_id: str
    sequence: str


def parse_fasta_text(text: str) -> List[FastaRecord]:
    """Parse FASTA from a text blob.

    Accepts multi-line sequences, ignores blank lines.
    If input has no header lines, returns empty (caller can decide how to handle).
    """
    if not text:
        return []

    lines = [ln.strip() for ln in text.splitlines()]
    records: List[FastaRecord] = []
    cur_id: str | None = None
    cur_seq_parts: List[str] = []

    def flush() -> None:
        nonlocal cur_id, cur_seq_parts
        if cur_id is None:
            return
        seq = "".join(cur_seq_parts).replace(" ", "").replace("\t", "").upper()
        records.append(FastaRecord(seq_id=cur_id, sequence=seq))
        cur_id = None
        cur_seq_parts = []

    for ln in lines:
        if not ln:
            continue
        if ln.startswith(">"):
            flush()
            header = ln[1:].strip()
            cur_id = header.split()[0] if header else f"seq_{len(records)+1}"
        else:
            if cur_id is None:
                # not a FASTA (no header yet)
                return []
            cur_seq_parts.append(ln)

    flush()
    return records


def format_fasta(records: Sequence[FastaRecord], line_width: int = 60) -> str:
    chunks: List[str] = []
    for r in records:
        chunks.append(f">{r.seq_id}")
        seq = r.sequence
        for i in range(0, len(seq), line_width):
            chunks.append(seq[i : i + line_width])
    return "\n".join(chunks) + ("\n" if chunks else "")


def validate_records(records: Sequence[FastaRecord]) -> Tuple[int, List[str]]:
    """Return (num_invalid_records, sample_messages)."""
    bad = 0
    msgs: List[str] = []
    for r in records:
        illegal = sorted({ch for ch in r.sequence if ch.isalpha() and ch not in _ALLOWED_AA})
        if illegal:
            bad += 1
            if len(msgs) < 10:
                msgs.append(f"{r.seq_id}: 非法字符 {''.join(illegal)}")
    return bad, msgs


def length_stats(records: Sequence[FastaRecord]) -> dict:
    lens = [len(r.sequence) for r in records]
    if not lens:
        return {"n": 0}
    s = sorted(lens)
    mid = s[len(s) // 2]
    return {
        "n": len(lens),
        "min": int(s[0]),
        "median": int(mid),
        "max": int(s[-1]),
    }
