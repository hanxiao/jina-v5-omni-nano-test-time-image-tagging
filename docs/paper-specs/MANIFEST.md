# Data Room: Image Tagging / Multi-Label via Embeddings & Vocabulary

主题：用模型(CLIP/embedding)+预定义词表(terms.txt)给图片做标注/多标签打标签。
聚焦近两年 (2023-2026) training-free / open-vocabulary 方向。

## 论文清单 (arxiv id -> papers/)

### A. 词表检索式打标签的鼻祖线
- CLIP Interrogator (无 paper, 工具) -> repo only: pharmapsychotic/clip-interrogator
  - BLIP caption + CLIP 在 artists/mediums/movements/flavors 词表上贪心检索最大化相似度

### B. Image Tagging 基础模型 (专门做 tag + 词表, 最相关)
- 2303.05657  Tag2Text: Guiding Vision-Language Model via Image Tagging
- 2306.03514  RAM: Recognize Anything - A Strong Image Tagging Model (CVPRW 2024)
- 2310.15200  RAM++: Open-Set Image Tagging with Multi-Grained Text Supervision
  - repo (三者同仓): xinyu1205/recognize-anything

### C. Training-free 开放词表多标签 (CLIP 双塔, 词表向量做分类器)
- 2312.12828  TagCLIP: Local-to-Global Framework (AAAI 2024) -> linyq2117/TagCLIP
- 2605.25821  PIAA: [CLS] is Not Enough - Patch-Level Inference + Adaptive Aggregation (ICML 2026) -> akang-wang/PIAA
- 2510.23894  Improving Visual Discriminability of CLIP for Training-Free Open-vocab
- 2412.06190  Category-Adaptive Cross-Modal Semantic Refinement and Transfer
- 2407.09073  Open Vocabulary Multi-Label Video Classification (ECCV 2024)

### D. 相关 (label embedding / 无监督多标签)
- CDUL: Abdelfattah et al. 2023 (unsupervised multi-label via CLIP pseudo-label)
- ADDS repo: Thomas2419/ADDS (open-vocab multi-label impl)

## Repo 清单 (-> repos/)
- pharmapsychotic/clip-interrogator
- xinyu1205/recognize-anything  (Tag2Text + RAM + RAM++)
- linyq2117/TagCLIP
- akang-wang/PIAA
- Thomas2419/ADDS

## 核心发现 (先写, 细节待精读)
- CLIP [CLS] 全局向量对比训练 -> 被最显著类主导, 多标签召回差 (所有论文共识)
- TagCLIP: 砍最后一层 attention 用倒二层 patch token; softmax over class + per-image min-max 归一后卡 0.5
- PIAA: modality gap 是瓶颈; 用无标签图 patch 在视觉空间学 GDA 闭式分类器替代文本词表向量
- RAM/RAM++: 走"训练一个 tagging 基础模型"路线, 不是纯 training-free; RAM++ 支持开放词表注入语义概念
- 我们优势: jina-embeddings-v5-omni text/image 同空间, modality gap 本就比 CLIP 小
