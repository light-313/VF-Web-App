import json
from typing import Any, Dict

import pandas as pd
import streamlit as st


APP_TITLE = "VF-Web"

# 简单图标（emoji）用于 UI 美观与可读性
ICON_HOME = "🏠"
ICON_QUEUE = "🧾"
ICON_BIN = "🧬"
ICON_MC = "🧫"
ICON_DASH = "📊"
ICON_ANALYSIS = "🔍"
ICON_DOWNLOAD = "⬇️"
ICON_RUN = "▶️"
ICON_ADD = "➕"


def apply_base_style() -> None:
    st.markdown(
        """
        <style>
          .block-container { padding-top: 1.2rem; }
          h1, h2, h3 { letter-spacing: 0.2px; }

          .vf-card {
            padding: 1rem 1.1rem;
            border: 1px solid rgba(49, 51, 63, 0.15);
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.03);
          }
          .vf-muted { opacity: 0.8; }
          .vf-small { font-size: 0.92rem; }

          .vf-kpi {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.6rem;
          }
          .vf-kpi > div {
            padding: 0.8rem 0.85rem;
            border: 1px solid rgba(49, 51, 63, 0.15);
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.02);
          }
          .vf-kpi .vf-kpi-num { font-size: 1.45rem; font-weight: 750; line-height: 1.1; }
          .vf-kpi .vf-kpi-lab { opacity: 0.78; font-size: 0.92rem; }

          section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def download_buttons(df: pd.DataFrame, payload_json: Dict[str, Any], prefix: str) -> None:
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    col1, col2,col3 = st.columns(3)
    with col1:
      st.download_button(
          label=f"{ICON_DOWNLOAD} 下载 CSV",
          data=csv_bytes,
          file_name=f"{prefix}_results.csv",
          mime="text/csv",
      )

      json_bytes = json.dumps(payload_json, ensure_ascii=False, indent=2).encode("utf-8")
    with col2:
      st.download_button(
          label=f"{ICON_DOWNLOAD} 下载 JSON",
          data=json_bytes,
          file_name=f"{prefix}_results.json",
          mime="application/json",
      )
    with col3:
      if "sequence" in df.columns and "seq_id" in df.columns:
        # FASTA（湿实验/后续分析更常用）
        lines = []
        for sid, seq in zip(df["seq_id"].astype(str).tolist(), df["sequence"].astype(str).tolist()):
          lines.append(f">{sid}")
          s = (seq or "").strip()
          for i in range(0, len(s), 60):
            lines.append(s[i : i + 60])
        fasta_bytes = ("\n".join(lines) + "\n").encode("utf-8")
        st.download_button(
          label=f"{ICON_DOWNLOAD} 下载 FASTA",
          data=fasta_bytes,
          file_name=f"{prefix}_results.fasta",
          mime="text/plain",
        )
