# 图片打标签论文核心算法示意图规格 (Diagram Specs)

说明：8 篇论文的核心算法流程图规格，用于喂给 Gemini 画每篇的 method pipeline。术语尽量保留论文原名 (英文)。范式区分：
- **RAM 系 (Tag2Text / RAM / RAM++)**：训练一个 tagging 基础模型，核心是 image-tag recognition/alignment decoder + 词表 label queries。
- **CLIP training-free 系 (TagCLIP / PIAA / VisualDiscriminability)**：冻结 CLIP，用 patch token 做 dense 分类/分割，无梯度训练。
- **训练式 open-vocab (CategoryAdaptive C2SRT / OpenVocabVideo)**：微调/prompt CLIP，引入 GAT 或 LLM 语义增强。

---

## 1. Tag2Text (2303.05657)

**一句话核心思想**：用文本语义解析 (text semantic parser) 从 image-text pair 里免标注地抽出 image tags，把 tagging 作为一个受监督子任务，和 caption 生成、image-text 对齐一起多任务预训练；tag 同时充当"可见的对齐/生成桥梁"。

**完整数据流 (tagging 推理路径)**：
1. Input image → **Image Encoder** (Swin-Base 或 ViT-Base) → image spatial features (视觉空间特征，非全局)。
2. image features + 预定义 **3,429 类 tag 的 learnable label queries** → **Image-Tag Recognition Decoder** (2 层 transformer，无 self-attention，cross-attention 让 query 去 attend image features)。
3. Recognition decoder 对每个 tag query 输出一个 logit → 阈值判定 → 输出 tags。
4. (预训练时额外分支) 识别出的 tags → 重排去位置偏置 → 与 image features 在 **Image-Tag Interaction Encoder** 交互 → **Image-Tag-Text Generation Decoder** 生成 caption；以及 **Image-Text Alignment Encoder** 做 ITC/ITM 对齐。

**关键模块清单**：
- **Text Semantic Parser** (offline)：把 caption 解析成 head→object/scene、modifier→attribute、relation→action，产生免标注 tag 真值。仅训练时用。
- **Image Encoder**：Swin/ViT，出 spatial features。
- **Image-Tag Recognition Decoder**：ML-Decoder 式，label query 驱动的多标签识别头，核心 tagging 模块。
- **Image-Tag Interaction Encoder + Image-Tag-Text Generation Decoder**：tag 引导的 caption 生成。
- **Image-Text Alignment Encoder**：ITC + ITM。

**词表在哪一步/怎么用**：词表 = 3,429 个高频 tag 类 (从 4M image-text 解析出 top-5000 再人工过滤合并同义词)。每个 tag 对应 recognition decoder 里一个 **learnable label query embedding**（注意：Tag2Text 的 query 是随机可学习向量，**没有**语义文本编码，因此只能识别训练见过的固定类别，不能开集）。

**怎么定"命中"**：recognition decoder 对每个类输出独立概率，用 **阈值 thresholding** (多标签，各类独立，非 softmax 竞争)；训练用 robust alignment loss (ASL)。

**相对前作创新点**：
1. 首次用 text semantic parsing 免标注拿到大规模 (3,429 类) tag 监督，tagging 性能超过全监督 ML-Decoder。
2. image-tag-text 三元生成范式：tag 作为可控 caption 生成的桥梁 + 检索时的可见对齐指示器。

**画图建议**：从左到右。最左 input image → Image Encoder (大框) 出 spatial features。中间画三条并行分支从 image features 拉出：上=Recognition Decoder (输入 label queries 小方块列 "cat/dog/..."，输出 tags，**高亮这条**)，中=Tag→Interaction Encoder→Generation Decoder 出 caption，下=Alignment Encoder 出 ITC/ITM。左下角单独画一个 offline 虚线框 "Text Semantic Parser: caption→tags" 指向 recognition 的监督信号。突出"tag 作为桥梁"的箭头贯穿三分支。

---

## 2. RAM — Recognize Anything Model (2306.03514)

**一句话核心思想**：在 Tag2Text 架构上，把 recognition decoder 的随机 learnable label query 换成 **CLIP text encoder 编码的 textual label queries (带语义)**，从而具备开集 (open-vocabulary) 识别能力——可识别训练没见过的任意类别；再配一个自动 data engine 把词表扩到 6,449 类。

**完整数据流 (推理)**：
1. Input image → **Image Encoder** (Swin-B/L) → image features；(训练时) 同时用 **CLIP image encoder 做特征蒸馏** 对齐图文空间。
2. 待识别的 tag list (可自定义任意类别) → **off-the-shelf CLIP Text Encoder** + prompt ensembling (offline) → **Textual Label Queries** (语义丰富的类向量)。
3. image features + textual label queries → **Image-Tag Recognition Decoder** (2 层, 去掉 self-attention) → 每类 logit。
4. 阈值判定 → 输出 tags。（可选接 Grounding-DINO + SAM 做定位分割 pipeline。）

**关键模块清单**：
- **Image Encoder** (Swin)；**CLIP Image Encoder** (仅训练时蒸馏，提升未见类识别)。
- **CLIP Text Encoder** (off-the-shelf, 冻结)：把 tag 文本编码为 label queries——RAM 的关键新增。
- **Image-Tag Recognition Decoder**：与 Tag2Text 同，但 query 来自文本而非随机。
- **Text Generation Encoder-Decoder**：captioning，与 tagging 联合训练。
- **Data Engine** (offline)：Generation (baseline 模型补全 tag/caption) + Cleaning (Grounding-DINO 裁区域 + K-Means++ 去 outlier)，把 4M 图的 tag 从 12M 扩到 39.8M。

**词表在哪一步/怎么用**：词表 = 6,449 fixed tags (合并同义词后 4,585 tag ID)，从 top-10k 高频解析 tag 挑选，覆盖分类/检测/分割主流数据集类别。词表在 **推理前经 CLIP text encoder 编码成 textual label queries**，喂进 recognition decoder；开集时可临时替换成任意类别文本。

**怎么定"命中"**：每类独立 logit + **阈值** (多标签)；训练用 ASL alignment loss。

**相对前作创新点 (vs Tag2Text)**：
1. **Open-vocabulary**：textual label queries (CLIP text 编码) 取代随机 learnable query，可识别未见类别。
2. **自动 data engine** (generation + cleaning) 大幅扩充并清洗 tag 标注，词表 3,429→6,449。

**画图建议**：从左到右。上支路：input image → Image Encoder → image features。下支路 (offline，虚线框)：tag list → CLIP Text Encoder + prompt ensemble → Textual Label Queries。两支路汇入中央 **Image-Tag Recognition Decoder** (大框，高亮)，右出 tags (阈值)。相对 Tag2Text 的图，**重点高亮 CLIP Text Encoder→label queries 这条新增语义支路**，并在图上标注 "open-set: 词表可换任意类别"。右下角可加小虚线块 "Data Engine (train only)"。

---

## 3. RAM++ (2310.15200)

**一句话核心思想**：把 image tagging 从"图-tag 对齐"升级为**多粒度文本对齐 (multi-grained text alignment)**：用一个 shared alignment decoder 同时对齐 (a) batch 内的整句 caption (global text supervision) 和 (b) 每个 tag 的 **LLM 生成的多条视觉描述 (tag descriptions)**，并用 automatic re-weighting 融合每类的多条描述。

**完整数据流 (推理)**：
1. Input image → **Image Encoder** (Swin-Base) → image spatial features + global feature。
2. 每个 tag 类 → 5 个 LLM prompt → **GPT-3.5-turbo** 生成 50 条 visual descriptions (offline) → **CLIP Text Encoder** 编码 (offline 预存) → 每类 50 个 description embeddings。
3. **Automatic Re-weighting 模块**：用 image global feature 和各 description 算相似度 softmax 权重，把 50 条描述 **加权合成一个 tag embedding**。
4. image spatial features (Key/Value) + 加权 tag embedding (Query) → **Alignment Decoder** (2 层, cross-attention + FFN, 无 self-attention) → 每类 alignment probability。
5. 阈值判定 → tags。

**关键模块清单**：
- **Image Encoder** (Swin)：出 spatial + global feature。
- **CLIP Text Encoder** (冻结, off-the-shelf)：编码 caption 和 tag descriptions。
- **Alignment Decoder (shared)**：ITTA 范式核心——同一 decoder 既做 image-text 对齐 (训练) 又做 image-tag 对齐 (tagging)。text/tag 作 Query，image spatial feature 作 Key/Value。
- **LLM (GPT-3.5)**：offline 生成 tag descriptions，把语义受限的 tag 监督扩成开放语义描述。
- **Automatic Re-weighting 模块**：公式 `Softmax(τ·g_v(V_global)·g_w(d_ij))` 对每类多描述加权。

**词表在哪一步/怎么用**：词表 = 4,585 类。每类不再是单个词，而是 **LLM 扩写的 50 条视觉描述**；描述经 CLIP text encoder 编码、offline 预存，推理时按图像相关性 re-weight 合成一个 tag embedding，喂进 alignment decoder。开集时词表/描述可任意扩展。

**怎么定"命中"**：alignment decoder 每类输出 probability，**阈值** 多标签判定；训练用 ASL。三种范式对比：CLIP=ITC (dot product), BLIP=ITM (deep fusion 单对), RAM++=**ITTA** (Image-Tag-Text Alignment，spatial feature + 轻量 decoder，兼顾效率与多类)。

**相对前作创新点 (vs RAM)**：
1. **Unified ITTA**：shared alignment decoder 同时统一 image-text 和 image-tag 对齐 (用 spatial feature 而非 global)，开集能力大增。
2. **LLM tag descriptions + automatic re-weighting**：把训练阶段就注入 LLM 知识 (前作只在推理注入)，每类多描述自适应加权。

**画图建议**：从左到右，双输入汇聚式。左上：image → Image Encoder → spatial feature (Key/Value) + global feature。左下 (offline 虚线区)：每个 tag → LLM (5 prompts) → 50 descriptions → CLIP Text Encoder → embeddings → **Automatic Re-weighting** (用 global feature 打权重，高亮这个模块) → 每类一个 tag embedding (Query)。两路进入中央 **Alignment Decoder** (大框, 高亮, 标 ITTA)，右出每类概率 + 阈值 → tags。可在角落放 ITC/ITM/ITTA 三小图对比。

---

## 4. TagCLIP (2312.12828)

**一句话核心思想**：**training-free**，冻结 CLIP-ViT。关键发现：CLIP 最后一层 self-attention 破坏空间信息，所以**去掉最后一层 self-attention**，用倒数第二层的 patch (dense) token 做 patch 级多标签分类，再经 DMAR (注意力精炼) + CWR (类级重识别) 两步纠错得到可靠 image tags。

**完整数据流**：
1. Input image → **CLIP Image Encoder (ViT-B/16, "ViT-modified")**：正常前传但**跳过最后一层 self-attention** (公式 6-8)，得 penultimate 层的 dense token `x_dense ∈ R^{N×D}`。
2. prompt + {categories} → **CLIP Text Encoder** → 文本 classifier `T ∈ R^{D×C}` (offline)。
3. 每个 patch token 与 T 算相似度 `s_i = Linear(x_dense,i)·T` → **softmax over classes** → **Patch-Level (Coarse) Classification Scores** `P_coarse ∈ R^{N×C}` (公式 9-10)。
4. **DMAR (Dual-Masking Attention Refinement)**：用 ViT 各层 attention 权重 A_l 做 patch 间 affinity 精炼；dual-mask = 注意力 mask M_attn (投票选置信元素) + 类级 mask M_cls，滤噪 → **Refined Scores** (公式 11-14)。
5. **CWR (Class-Wise Reidentification)**：对每类取高响应 patch 组成 class-wise region → 按 bbox 裁剪 resize 到 224 → 用 class-wise mask 作 attention mask 输入 **原始 CLIP** 用 [CLS] token 重新分类 → **Global Scores**；与 local 融合 `P_final = λP_local + (1-λ)P_global` (λ=0.5)。
6. 对 P_final 阈值 (0.5) → predicted tags (可作 WSSS 伪标签)。

**关键模块清单**：
- **CLIP Image Encoder (ViT-modified)**：去最后 self-attention 保空间信息。
- **CLIP Text Encoder**：编码 80 prompt ensemble 类名作 classifier。
- **DMAR**：无训练，用 ViT self-attention 做 affinity refine + dual masking 去噪。
- **CWR**：裁剪-重识别，用原始 CLIP [CLS] 从全局视角 double-check，纠 patch 级误判。

**词表在哪一步/怎么用**：词表 = 数据集类名 (VOC 20 / COCO 80)，经 CLIP text encoder + prompt ensemble 编成文本 classifier T，在 step 3 与每个 patch token 点积算分。开集能力继承 CLIP。

**怎么定"命中"**：patch 打分先 **softmax over classes** (作者强调 softmax 对 CLIP 很关键)；最终对 logits 做 **min-max 归一化到 [0,1] 后阈值 0.5** 判定正类 (多标签)。

**相对前作创新点**：
1. 发现并去除 CLIP 最后一层 self-attention 以保留 patch 空间信息，从 global [CLS] 转向 **local-to-global** 的 patch 级分类。
2. 无训练的 DMAR + CWR 两级精炼 (attention affinity + 裁剪重识别) 大幅压制假阳。

**画图建议**：从左到右三阶段流水线。input image → **CLIP ViT (modified, 去最后 self-attn)** 出 patch tokens；上方 prompt+categories → Text Encoder 出 classifiers。patch×classifier → **Patch-Level Coarse Scores** (画一张带噪类图，如 diningtable/car/horse/bird) → **DMAR** 框 (标 "attention affinity + dual mask") → Refined Scores (更干净) → **CWR** 框 (画裁剪 crop&resize→CLIP [CLS] 重识别) → Final Scores → 阈值 0.5 → tags。突出"去掉 last self-attention"这个关键点 (可在 ViT 框上标红注释)。自上而下融合 local+global 用一个 Fuse 小节点。

---

## 5. PIAA — Patch-level Inference and Adaptive Aggregation (2605.25821)

**一句话核心思想**：**training-free / 无反传**，冻结 CLIP。口号 "[CLS] is not enough"：把多标签识别重构为 **patch 级推理 + 自适应聚合**；核心是用 **GDA (Gaussian Discriminant Analysis) 从 patch 特征闭式解出一个无监督视觉 classifier** 来弥合 vision-language modality gap，再自适应融合 patch 分数与 [CLS] 分数。

**完整数据流**：
1. Input image → **CLIP-based segmentation-style Image Encoder** (ViT-B/16，可选叠加 SC-CLIP/ITACLIP 等 disentanglement 前端) → patch embeddings `{x_i}` + global `[CLS]` embedding。
2. category names → **CLIP Text Encoder** → textual prototypes `{w_c}` (用于初始 zero-shot patch 概率 `p_i,c` 引导)。
3. **PVCL (Patch-based Visual Classifier Learning)**：三阶段从无标注 patch 闭式估计视觉 classifier `{w_c,b_c}`：
   - Stage I 熵引导 bootstrapping：用文本对齐概率 p_i,c 取每类最低熵 top-K=512 patch 建 memory bank；估初始 μ_c、Σ 得临时 GDA。
   - Stage II 视觉驱动净化：用临时 classifier 打 vision-driven 分 q_i,c，按 `q ≥ μ+σ` 统计阈值净化 bank。
   - Stage III 稳健收缩：置信加权类原型 μ_c + trace-regularized shrinkage 协方差 Σ → 闭式 `w_c=Σ⁻¹μ_c, b_c=-½μ_cᵀΣ⁻¹μ_c`。
4. 用 GDA classifier 给每个 patch 出 discriminant logit `ỹ_i,c` → softmax → **Spatial Evidence Distillation**：类内 max-pooling 取最判别 patch → 二次 softmax → `S_patch,c`。
5. **PAA (Prediction Adaptive Aggregation)**：`S_f,c = α·S_patch,c + (1-α)·S_cls,c` (α=0.9，重 patch 轻 [CLS])。
6. 阈值 → 多标签输出。

**关键模块清单**：
- **CLIP seg-style Image Encoder (+可选 disentanglement 前端)**：出判别性 patch embedding，抑制背景。
- **CLIP Text Encoder**：只用于 Stage I 熵 bootstrapping 的初始概率 (最终判别靠视觉 classifier)。
- **PVCL**：GDA 闭式无监督视觉 classifier，三阶段净化，弥合 modality gap，**无梯度**。
- **PAA**：max-pool patch 证据 + 与 [CLS] 全局锚自适应凸组合。

**词表在哪一步/怎么用**：词表 = 数据集类名 (VOC/COCO/NUS-WIDE)，经 CLIP text encoder 编成 textual prototypes；**仅在 PVCL Stage I** 用来给 patch 打初始 zero-shot 概率、筛选低熵 anchor patch，之后判别完全交给视觉侧 GDA classifier (刻意绕开跨模态 misalignment)。

**怎么定"命中"**：patch 级 GDA logit → softmax；类内 **max-pooling** 取峰值 → 二次 softmax；与 [CLS] softmax 凸组合 → 最终每类分数阈值判定 (多标签, mAP 评测)。

**相对前作创新点 (vs TagCLIP/SPARC 等 training-free)**：
1. **PVCL**：首次用 GDA 闭式解从无标注 patch 学纯视觉 classifier，直接消除 vision-language modality gap，无需反传/伪标签迭代。
2. **PAA**：原理化的 patch→image 自适应聚合 (小物体靠 patch max-pool，大物体靠 [CLS] 全局锚)。

**画图建议**：从左到右，双分支后融合。input image → CLIP seg-style Encoder (可标 optional disentanglement) → 分出 patch embeddings (主) 和 [CLS] embedding (辅)。patch 走进 **PVCL 大框 (高亮)**，框内画三阶段小步 (Entropy Bootstrapping → Vision Purification → Shrinkage GDA)，左上角 text prototypes 用虚线只指向 Stage I。PVCL 出 patch scores → max-pool (Spatial Evidence Distillation) → S_patch。[CLS] 出 S_cls。两者进 **PAA 融合节点** (标 α=0.9) → 阈值 → tags。强调 "无梯度/闭式解" 与 "patch 主导、[CLS] 正则" 的对比。

---

## 6. Visual Discriminability (2510.23894)

**一句话核心思想**：**training-free**，冻结 CLIP。诊断出 CLIP 深层的两大病灶——(a) 出现 high-norm/稀疏激活的 **abnormal tokens** 拉低 patch 判别性，(b) 末层 self-attention/FFN 提升语义对齐但严重损失视觉判别性——然后用三个免训练组件 (ATR + SSR + SHE) 在保持语义对齐的同时恢复 patch 级视觉判别性，用于 open-vocab 语义分割 (也即 dense patch 打标签)。

**完整数据流 (推理)**：
1. Input image → patch 化 → **CLIP ViT Encoder** 前传各层。
2. **ATR (Abnormal Token Replacement)**：在**倒数第二层 (L-1)** 用 **hoyer sparsity score** `H(x_i)>τ` 识别 abnormal tokens，用其 8 邻域正常 token 加权平均替换 (公式 6)。
3. **SSR (Spatial-Semantic Reweighting)**：在最后几层 (ViT-B 第 10-11 层) 重加权前传，**上调 residual 通路、下调 MSA/FFN**：`X̂=(1+α)X^{l-1}+(1-α)MSA(...)` (公式 7-8)，保住早层的判别性特征。
4. **SHE (Selective Head Enhancement)**：离线按 visual discriminability 分数选 top-k 判别性 attention heads，聚合其特征 `X_k` 构相似度图 S，阈值 β 过滤成 soft pseudo-mask，列归一化后精炼末层特征 `X^{L-1}←Norm(S_β)X^{L-1}`。
5. 精炼后的 patch 特征与 **CLIP Text Encoder** 的类名 embedding 算相似度 → **argmax over classes** (公式 3) → 每 patch 类别 → 分割/dense tags。

**关键模块清单**：
- **CLIP ViT Encoder** (冻结)。
- **ATR**：hoyer score 检测 + 邻域替换，去 abnormal token (仅 L-1 层)。
- **SSR**：residual/attention 重加权，跨末几层，救回视觉判别性。
- **SHE**：head 级选择 + pseudo-mask 精炼，用高判别 head 引导末层特征。
- **CLIP Text Encoder**：类名→文本 embedding 做逐 patch argmax 分类。

**词表在哪一步/怎么用**：词表 = 分割数据集类名 (VOC/Context/ADE/COCO-Stuff 等)，经 CLIP text encoder 编码，在最后一步与精炼后的每个 patch/token 特征算余弦相似度做 **逐 patch argmax** 归类。这是**单标签 per-patch** (分割) 而非 per-image 多标签。

**怎么定"命中"**：逐 patch **argmax over C classes** (分割语义)；评测 mIoU。SHE 的 pseudo-mask 用阈值 β。ATR 用 hoyer 阈值 τ。

**相对前作创新点**：
1. 系统诊断 CLIP 深层 abnormal tokens (high-norm、稀疏、bias-like、编码全局信息) 是视觉判别性崩塌主因，并指出末层 attention/FFN 只给边际语义增益却大损判别性。
2. 三个免训练组件 ATR+SSR+SHE 协同，在保语义对齐的前提下恢复 patch 判别性 (前作多只改末层)。

**画图建议**：自上而下沿 ViT 层堆叠展开更直观。左边一条 CLIP ViT 层栈 (layer 1…L)。中间标出三处干预：中层→末层区间挂 **SSR** (标 "(1+α)residual, (1-α)MSA/FFN")；倒数第二层挂 **ATR** (小图：检测 high-norm 稀疏 abnormal token 用邻域替换)；旁路挂 **SHE** (选 top-k 判别 head→相似度图→pseudo-mask→精炼 L-1 特征)。末端 patch 特征 → 与右侧 Text Encoder 类名 embedding 算相似度 → **argmax** → 分割图。高亮 "abnormal token" 病灶和三组件如何各治一处。

---

## 7. C2SRT — Category-Adaptive Cross-modal Semantic Refinement and Transfer (2412.06190)

**一句话核心思想**：**训练式** open-vocabulary 多标签识别 (OV-MLR)。两大自适应模块：**ISR** 按每类语义自适应地选可变数量的判别 patch (取代固定 patch 数)，**IST** 用 LLM 挖掘类间关系构 GAT 图做 inter-category 语义传递，从而把知识迁移到未见类。

**完整数据流**：
1. Input image → 切 P=196 patch + [CLS] → **learnable Vision Encoder (ViT-B/16, 从 CLIP 初始化)** → patch features `F_p ∈ R^{P×D}` + global `f_G`。训练时用 `L_dist=‖f_G - f_G^CLIP‖` 向**冻结 CLIP vision encoder 蒸馏**保泛化。
2. 每个类 c 的 prompt (ensemble 模板) → **冻结 CLIP Text Encoder** → 文本特征 `f_txt^(c)`。
3. **ISR (Intra-category Semantic Refinement)**：算每个 patch 与 `f_txt^(c)` 相似度 `s_i^(c)` → softmax → 降序排序 → 按信息阈值 α=0.5 累加选 top-n patch (n 自适应，上限 32) → pooling 得类特定局部特征 `f_L^(c)` (公式 4-5)。
4. 融合 `f_img^(c)=(f_L^(c)+f_G)/2`，与 `f_txt^(c)` 拼接 → FFN → 类节点特征 `h_0^(c)` (公式 6)。
5. **IST (Inter-category Semantic Transfer)**：用 **LLM (GPT-4o)** 挖掘类间关系 (synonym/is-a/functional/co-occurrence/part-whole + 强度) 建 sparse 有向图 → **GAT (GATv2, 2 层, multi-head)** 传递邻类语义 → 输出节点特征 `h_l^(i)` (公式 7-9)。
6. 预测 `ŷ_i = cos(h_l^(i), f_txt^(i))` (公式 10)，ranking loss 训练；阈值/排序判定多标签。

**关键模块清单**：
- **Learnable Vision Encoder (ViT-B/16)** + **知识蒸馏** (向冻结 CLIP vision encoder 对齐 global feature 防过拟合)。
- **冻结 CLIP Text Encoder**：prompt ensemble 出类文本特征。
- **ISR**：文本引导的自适应 patch 选择 (可变数量)，出类特定局部特征。
- **IST**：LLM 关系挖掘 + GAT 图，跨类语义传递到未见类。

**词表在哪一步/怎么用**：词表 = seen + unseen 类 (NUS-WIDE / Open Images)。类名经 prompt ensemble + CLIP text encoder 编码，(a) 在 ISR 里作 query 选每类判别 patch，(b) 作 GAT 图节点/边关系来源，(c) 最后与图输出特征算余弦相似度分类。open-vocab：unseen 类靠 text 特征 + IST 图从相关 seen 类迁移知识。

**怎么定"命中"**：`ŷ_i = cos(h_l^(i), f_txt^(i))` 每类打分；训练 ranking loss (正类分 > 负类分 + margin 1)；评测 ZSL/GZSL 用 mAP、F1，按分数排序/阈值判多标签。

**相对前作创新点 (vs MKT 等)**：
1. **ISR 自适应局部特征**：按每类语义自适应选可变数量判别 patch，解决固定 patch 数带来的噪声/次优。
2. **IST + LLM 关系图**：用 LLM 挖类间关系建 GAT 稀疏图做 inter-category 语义传递，显式建模标签相关性并迁移到未见类。

**画图建议**：从左到右三段式。input image → 切 patch+[CLS] → **Learnable Vision Encoder** (下方虚线连 "冻结 CLIP，Knowledge Distillation")，出 F_p + f_G。中段 **ISR 框 (高亮)**：patch features × 类文本特征 → 相似度排序 → 阈值 α 自适应选 top-n → pooling 出 f_L (画"不同类选不同数量 patch")。融合 f_img 与 f_txt 拼接→FFN 成节点。右段 **IST 框 (高亮)**：一张类关系 GAT 图 (节点=类, 边来自 LLM 关系挖掘, 画 GPT-4o 小标)，message passing → 输出节点特征 → 与 text 特征算 cos → 多标签输出。上方一条支路画 CLIP Text Encoder + prompt ensemble 供给 ISR 和 IST。

---

## 8. Open Vocabulary Multi-Label Video Classification (2407.09073)

**一句话核心思想**：**训练式** open-vocab 多标签**视频**分类。适配 CLIP：(a) 端到端可训练的 **label encoder**——冻结 LLM + learnable prefixes + prompt transformer，把 LLM 生成的类属性以可微方式喂进冻结 CLIP text encoder，统一不同概念 (物体/动作) 的打分尺度；(b) CLIP vision encoder 加 **temporal modeling branch** 建模时序，配正则化微调保开集泛化。

**完整数据流 (推理)**：
1. Input video (多帧) → **冻结 CLIP Image Encoder** + **learnable Temporal Branch** → 时空视频 embedding。
2. label encoder 侧 (可 offline 预算入 label embedding database)：class labels + **learnable prefixes** → **冻结 LLM** → LLM 输出 token 序列 → **learnable Prompting Transformer** (把 LLM token 变成 CLIP text encoder 的 soft prompts，绕开 detokenize/tokenize 的不可微问题) → **冻结 CLIP Text Encoder** → 每类 label embedding。
3. **Matching**：video embedding 与 label embedding database 逐类算相似度 → per-class scores。
4. 多标签判定 (每类独立分数)。三阶段：(a) training 联合训 label+video encoder；(b) classifier vocabulary expansion 把新类 label embedding 存库；(c) inference 匹配。

**关键模块清单**：
- **冻结 CLIP Image Encoder + Temporal Branch (learnable)**：把图像 CLIP 升级到视频时空建模；配 regularized finetuning 防过拟合、保零样本。
- **冻结 LLM**：为类标签生成语义属性/世界知识 (理解类层级)。
- **Learnable Prefixes**：prompt LLM 的可学习前缀。
- **Prompting Transformer (learnable)**：把 LLM token 序列转成 CLIP text encoder 的 soft prompts，实现端到端可微 (核心工程点)。
- **冻结 CLIP Text Encoder**：当 "label encoder"，出每类 embedding。
- **Label Embedding Database**：预存类 embedding，支持随时扩词表 (open-vocab)。

**词表在哪一步/怎么用**：词表 = 任意推理时给定的类 (entities 物体/场景 + actions 动作)。类标签经 learnable prefixes→LLM→prompting transformer→CLIP text encoder 编成 label embedding，存入 **label embedding database**；vocabulary expansion 阶段可随时加新类。推理时 video embedding 与库内所有类 embedding 匹配。open-vocab 靠这个可扩展的 label embedding database。

**怎么定"命中"**：video-label **相似度匹配**，每类独立 class-wise score (图 2c 例：0.9/0.2/0.8)；因不同概念 (动作 vs 物体) CLIP 分数尺度不一致，端到端微调 + LLM 引导来对齐尺度，使多标签阈值判定可行 (非单纯 top-1)。

**相对前作创新点**：
1. **端到端可微的 LLM-guided label encoder**：learnable prefixes + prompting transformer 让 "LLM 生成属性→CLIP text encoder" 全程可反传 (前作 Menon&Vondrick 式 LLM prompting 因 tokenize 不可微、无法训练)，并校正不同概念的打分尺度。
2. **Temporal modeling branch + 正则化微调**：给 CLIP vision encoder 加时序建模同时保留开集零样本能力；首次定义 open-vocab 多标签视频分类任务与基准。

**画图建议**：从左到右，双编码器汇入 matching。下支 (video)：input video 多帧 → 冻结 CLIP Image Encoder + **Temporal Branch (高亮, 标 🔥learnable)** → video embedding。上支 (label encoder，高亮整条)：class labels + **🔥Learnable Prefixes** → ❄LLM → **🔥Prompting Transformer** → ❄CLIP Text Encoder → Label Embedding → 存入 **Label Embedding Database** (画成可扩展的库, 标 "add new classes anytime")。两支在右侧 **Matching** 节点汇合 → per-class scores (0.9/0.2/0.8) → 多标签。用雪花❄标冻结 (CLIP/LLM)、火焰🔥标可训练 (prefixes/prompting transformer/temporal branch)，突出"冻结大模型 + 少量可学习桥接"的范式。
