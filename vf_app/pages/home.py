import os

import streamlit as st

from ..ui import ICON_ADD, ICON_BIN, ICON_DOWNLOAD, ICON_HOME, ICON_MC, ICON_RUN


_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))

# TODO: 把下面链接替换成你自己的 Colab Notebook 分享链接
COLAB_FEATURE_EXTRACTION_URL = "https://colab.research.google.com/drive/1GpPVOhF9ixIMSzzHHeQWOtrSui08whBv"


def render() -> None:
    st.markdown("""
    **VF-Web** 是一个面向生物学家的在线分析平台，旨在解决细菌毒力因子（Virulence Factors, VFs）识别中的核心挑战。
    它利用最先进的深度学习技术，帮助研究人员从海量蛋白质序列中快速筛查潜在的致病因子，并对其进行精细的功能亚型注释。
    我们的目标是弥合复杂算法与实际应用之间的技术鸿沟，让高性能 AI 模型触手可及。
    """)

    st.markdown("### 模型构建思路")

    img1 = os.path.join(_ROOT, "1.png")
    img2 = os.path.join(_ROOT, "2.png")
    c1, c2 = st.columns(2)
    with c1:
          st.markdown(
            """
            <div class="vf-card">
              <div class="vf-muted vf-small">
                <ul style="margin:0;">
                  <li>输入来自 ESM2 / ProtT5 的 embedding（或拼接）。</li>
                  <li>构建 4 个深度模型：融合模型 <b>DPF</b> + 三个单/拼接特征的 <b>VFITER</b>，进行<b>加权投票</b>。</li>
                </ul>
              </div>

            </div>
            """,
            unsafe_allow_html=True,
        )
          st.image(
                img1,
                caption="二分类：ESM2/ProtT5 特征 + DPF/VFITER + 加权投票",
                use_column_width=True,
            )

    with c2:
            st.markdown(
            """
            <div class="vf-card">
              <div class="vf-muted vf-small">
                <ul style="margin:0;">
                  <li>先进行<b>对比学习特征细化</b>（ProjectionHead），再训练分类头（Linear/MLP）。</li>
                  <li>分别在 <b>esm / prot / both</b> 三种输入模式上训练推理，并通过<b>投票/平均概率</b>得到最终类别。</li>
                </ul>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
            st.image(
                img2,
                caption="多分类：对比学习特征细化 + 分类头；esm/prot/both 投票",
                use_column_width=True,
            )

    st.markdown("### 如何使用")
    st.markdown(
        """
        <div class="vf-card">
          <div class="vf-muted vf-small">
            <ol style="margin:0; padding-left:1.1rem;">
              <li><b>Colab 离线特征提取</b>：把 FASTA 序列转换为 ESM2 / ProtT5 的 embedding（输出 h5）</li>
              <li><b>上传 embedding(h5)</b>：Web 端读取 embeddings/labels（可选 sequences），并在不同模态间按 seq_id 对齐</li>
              <li><b>推理预测</b>：二分类走 4 模型加权投票；多分类走对比学习模型投票或单 both</li>
              <li><b>结果下载</b>：表格预览 + CSV/JSON 导出，便于复现/分享</li>
            </ol>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"特征提取 Colab：[{COLAB_FEATURE_EXTRACTION_URL}]({COLAB_FEATURE_EXTRACTION_URL})")


    st.markdown("#### Web 端快速上手")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"**1) {ICON_BIN} / {ICON_MC} 选择任务**")
        st.caption("左侧选择二分类或多分类")
    with c2:
        st.markdown(f"**2) {ICON_ADD} 上传输入（可选）**")
        st.caption("不上传则使用仓库默认路径/权重")
    with c3:
        st.markdown(f"**3) {ICON_RUN} 运行并 {ICON_DOWNLOAD} 下载结果**")
        st.caption("结果会以表格展示，可下载 CSV/JSON")

    with st.expander("输入文件要求（简要）", expanded=False):
        st.write("上传的 h5 需要包含 embeddings/labels（以及可选 sequences）等字段，且不同模态之间能按 seq_id 对齐。")
        st.write("如果二分类使用自定义 config.json，请确保其中的模型 path 指向可读取的 .pth 权重文件。")
