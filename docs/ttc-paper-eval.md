# Test-Time Compute / Training-Free 图像分类论文精读评估

评估对象：9 篇论文能否嫁接到我们现有的 **jina-embeddings-v5-omni-nano 单模型自洽 pipeline**。

## 我们 pipeline 的关键约束（评估基准线）
- 单模型：只用 v5-omni-nano，图文同 768 维空间，last-token pooling。
- 标签空间 = tokenizer 词表筛出的 ~2.5 万词，encode_text 成标签矩阵。
- 每图打分：`base = 0.7·s_patch + 0.3·s_global`（patch 类内 max-pool + global）。
- 背景中心化：减 50 张背景图的 per-label 先验 μ。
- CWR multi-crop：14 crop 跨 crop max，`S = base + 1.3·s_crop`。
- embedding-NMS 去重。
- COCO-150 / 80 类：mAP 0.710，P@1 0.813。
- **硬性排除**：① 第三方模型（DINO/BLIP/CLIP/LLM）；② heavy training / 反传。
- **允许**：test-time 轻量计算（贝叶斯/kernel/概率/图传播/最优传输）；一次一 batch 图（transductive 可行）。
- **已知无效**：换层 / 白化 / softmax-over-classes 这类特征再处理在 v5-omni 上无效或有害。

> 注意：本任务是**多标签检测**（COCO 一图多物，mAP + P@1），而绝大多数论文是**单标签 top-1 分类**（每图恰好一类）。这是一个贯穿所有评估的核心 gap：凡是假设"每图属于唯一类""类先验和为 1""batch 内类分布可估"的方法，直接搬过来都会和多标签设定冲突，需要改造。下面每篇都会点明。

---

## 1. OTTER — Zero-Shot Optimal Transport (2404.08461)

**一句话核心方法**：把 zero-shot 预测看成 batch 图像↔类之间的最优传输问题，用类的目标边际分布（label distribution）约束传输，从而校正 CLIP 的 label-distribution mismatch（base-rate 偏差）。

**具体数学机制**：
- 输入 batch n 张图、K 个类。定义图像边际 `μ = (1/n)·1_n`（每图等权），类边际 `ν = (p_1,...,p_K)`（目标类分布，需给定）。
- 代价矩阵 `C_ij = -log P_t(Y=j|X=i)`，即用分类器分数（cos 相似度经 softmax）的负 log 作代价。
- 解熵正则最优传输：`π = argmin_{γ∈Π(μ,ν)} <γ, C>`，Sinkhorn 迭代求传输计划 π。
- 每图预测 = π 中该行 argmax。
- R-OTTER 变体：从 OTTER 的伪标签学一个 reweight 向量 `r ≈ P_t(Y)/P_s(Y)`，等价于 logit adjustment，可用于在线（单图）预测。

**需要什么输入**：**一 batch 图（transductive）**；无需标注；**但需要一个类目标分布 ν 的估计**（论文核心假设，实验里用真值或估计值）。

**第三方模型**：**否**。纯在 CLIP 分数上做 OT。✅

**训练/反传**：**否**。Sinkhorn 是迭代但无梯度。✅

**计算量**：test-time 轻量，Sinkhorn 几十次迭代，batch × K 规模。

**能否嫁接（可移植性 4/5）**：
- 怎么接：这是我们**背景中心化（减 μ）的直接竞品/升级**。背景中心化本质是手工去 base-rate；OTTER 用 OT 从 batch 统计里自适应校正 base-rate，理论上更优雅。可在 `S`（融合分）之后、NMS 之前，用 `-log softmax(S)` 当代价矩阵跑一次 batch 级 Sinkhorn。
- **多标签冲突（关键风险）**：OTTER 假设每图分到一个类（行边际 = 1/n，即每图总"质量"固定分给各类）。COCO 一图多物，不满足。需改造：放松图像边际约束，或用 unbalanced OT（KL 松弛边际），或对 multi-label 用逐类阈值而非 argmax。
- 需要 ν：我们没有类分布先验。可从 batch 自身 s_global 估一个软 ν（自洽），但这会引入循环依赖，需实验。
- 预期收益：中高。若 batch 内类分布不均（COCO 常见 person 泛滥），OT 校正比固定 μ 更能压高频类、提低频类召回 → mAP 有望涨。
- 风险：多标签边际假设需改；ν 估计不准会伤害；和已有背景中心化可能功能重叠（要做消融，二选一或叠加）。

---

## 2. Transductive Zero/Few-Shot CLIP — EM-Dirichlet (2405.18437)

**一句话核心方法**：把一 batch query 图的图文概率向量放到 K 维单纯形上，用 Dirichlet 分布对每类建模，转化为单纯形上的正则化聚类，用 block Majorization-Minimization（EM 式）联合估计 Dirichlet 参数和类分配。

**具体数学机制**：
- 每图先算 `z_n = softmax(T·cos(f_im, f_text_k))`，得到单纯形 Δ_K 上的概率向量。
- 假设每类 z 服从 Dirichlet(α_k)。
- 目标：`min_{u,α} -L(u,α) + Φ(u) + λΨ(u)`，u_n∈Δ_K 是软分配，L 是 Dirichlet 对数似然，Ψ(u) = -Σπ_k ln π_k 是分区复杂度惩罚（防止类塌缩），π_k = batch 内类 k 平均分配。
- 交替优化：MM 更新 Dirichlet 参数 α_k（用 Gamma 函数的切线 majorant，避免 Newton）；软分配更新 `u_n = softmax(ln p(z_n|α_k) + (λ/|Q|)·ln π_k)`。
- 也给了 EM-GMM 变体（协方差固定）。

**需要什么输入**：**一 batch query 图（transductive，论文用 batch=75）**；zero-shot 无标注即可（few-shot 时 support 作硬约束）。

**第三方模型**：**否**。只用 CLIP 图文特征。✅

**训练/反传**：**否**，是 test-time EM 迭代（MM）。✅

**计算量**：中等。EM 外层几十次迭代，每次 MM 内层更新，涉及 Gamma/digamma；batch × K。比 OT 略重但可控。

**能否嫁接（可移植性 3/5）**：
- 怎么接：作为 **batch 内后处理再分配层**，替代或补充 softmax-over-classes（我们已知 softmax-over-classes 有害，但这里是 Dirichlet 建模 + batch 统计，不是简单 softmax）。用 batch 图的分数向量做 Dirichlet 聚类，得到更 calibrated 的类后验。
- **多标签冲突（严重）**：整个框架建立在"每图一个软分配 u_n∈Δ_K，Σ_k u_{n,k}=1"上，是单标签范式。COCO 多标签下"每图分配总和为 1"直接矛盾。改造成本高（要放弃单纯形约束，改成逐类独立 Beta/logistic），基本等于重做。
- **已知风险叠加**：我们实测 softmax-over-classes 有害，说明 v5-omni 空间不喜欢"跨类归一化"。Dirichlet on simplex 本质也是跨类归一化建模，很可能同样有害。
- 预期收益：低-中，且和已知教训相悖。
- 结论：理论漂亮但和我们的多标签 + "跨类归一化有害"两条都撞，不优先。

---

## 3. ZLaP — Label Propagation for Zero-Shot (LabelPropagation_CVPR2024)

**一句话核心方法**：把类文本节点 + 无标注图像节点建成一张 kNN 图，用 label propagation 从文本节点把标签沿图像流形传播到图像节点；并给出对偶解 + 稀疏化实现高效 inductive 推理。

**具体数学机制**：
- 节点 = {C 个类文本向量 w_c, M 个图像向量 u_i}。构邻接矩阵 S：`s_ij = x_i^T x_j` 若 x_j∈kNN(x_i) 否则 0；对称化 `S̄=S+S^T`，归一化 `Ŝ = D^{-1/2} S̄ D^{-1/2}`。
- LP 迭代：`ŷ_c^{(t+1)} = α Ŝ ŷ_c^{(t)} + (1-α) y_c`，收敛等价于解线性系统 `L ŷ_c = y_c`，`L = I - αŜ`（graph Laplacian），闭式 `ŷ_c = L^{-1} y_c`。
- 对偶解：解 `ẑ_j = L^{-1} e_j`，用 L^{-1} 的对称性把 C 个系统换成按需列求解，配合 CG + 稀疏化做 inductive。
- **关键工程点**：CLIP 图文有 modality gap，直接 kNN 会图-图、文-文各自成团。ZLaP **分开做 kNN**：图↔图 和 图↔文 分别建边，文本节点只在作为某图的近邻时才连入。

**需要什么输入**：**一 batch / 全测试集图（transductive）**，inductive 变体支持流式；无标注。

**第三方模型**：**否**。黑盒 VLM 特征即可。✅

**训练/反传**：**否**。闭式 + CG，纯线性代数。✅

**计算量**：transductive 需解线性系统（CG，稀疏后可接受）；inductive 用对偶解 + 稀疏。COCO-150 图规模非常小，L 是 (C+M)×(C+M)，几百维，**直接闭式求逆都毫无压力**。

**能否嫁接（可移植性 5/5）**：
- 怎么接：**最契合的一篇**。在算完标签矩阵和图像 patch/global 向量后，把"标签词向量 + batch 图像向量"建图跑 LP，作为 `s_crop` / 融合之后的**流形平滑再打分层**。类节点标签沿图像相似度流形传播 → 利用 batch 内图像互相印证（图 A、B 都像"斑马"会互相加强）。
- **modality gap 处理直接可用**：ZLaP 的分开 kNN 正好解决图文 gap，v5-omni 虽 gap 小但仍存在，可直接借鉴。
- **多标签兼容性好**：LP 是逐类独立传播（每个类一列 ŷ_c），不强制每图归一化，**天然适配多标签**（只要不 argmax，改成逐类阈值/mAP 排序即可）。这是相对 OT/Dirichlet 的最大优势。
- 预期收益：高。batch 内 transductive 平滑通常给 zero-shot 稳定涨点，且我们 batch 小、算力零压力。
- 风险：kNN 的 k、α 需调；类节点只有 1 个/类（文本），传播源稀疏 → 可给每类多 prompt/多 crop 文本节点缓解。稀疏化/inductive 那套我们规模小用不上，只取 transductive 闭式即可。

---

## 4. ECALP — Efficient & Context-Aware Label Propagation (2412.18303)

**一句话核心方法**：ZLaP 的动态化升级——在文本 prompt + (few-shot) + test 样本上**动态构图**，用**迭代** LP（而非闭式）支持增量传播和 label reset，并加一个 context-aware 特征再加权提升适配。

**具体数学机制**：
- 图节点 = 文本 prompt + few-shot（可选）+ test 样本，kNN 建图。
- 用**迭代解**而非闭式（便于增量：来一个新样本就扩图、局部传播，label reset 避免漂移）。
- **context-aware edge re-weighting**：不用裸 cos 相似度，而是用 prompt/few-shot 提供的上下文对特征做再加权后再算边权（针对 CLIP 特征各向异性/gap 问题）。
- 支持图扩展做流式 inductive。

**需要什么输入**：文本 prompt + test 样本（zero-shot），few-shot 可选；transductive + 流式都行。

**第三方模型**：**否**。training-free，只 VLM 特征。✅

**训练/反传**：**否**。✅

**计算量**：迭代 LP，增量友好，比 ZLaP 闭式更省（针对大规模）；我们规模小，收益不明显但也无害。

**能否嫁接（可移植性 4/5）**：
- 怎么接：本质是 ZLaP 的同族方法，**二选一**。若上了 ZLaP，ECALP 的增量/流式对我们（batch 小、离线）价值不大；但它的 **context-aware 特征再加权** 值得单独借鉴——不过要警惕：我们已知"特征再处理在 v5-omni 上无效或有害"，这条正好是 feature re-weighting，**大概率无效**。
- 多标签兼容性：同 LP，逐类传播，天然好。✅
- 预期收益：中（相对 ZLaP 的增量收益）；feature re-weighting 部分预期在 v5-omni 上负收益。
- 结论：把 ZLaP 当主力，ECALP 作为"若需流式/增量再上"的备选；它的 re-weighting 别抄。

---

## 5. BCA — Bayesian Class Adaptation / Bayesian TTA (BayesianTTA_CVPR2025)

**一句话核心方法**：在线 test-time 用 Bayes 定理，不仅更新类嵌入（likelihood），还用每个到来样本的**后验去更新每类的先验 prior**，双更新应对分布漂移。

**具体数学机制**：
- Bayes：`P(Y|x) ∝ P(x|U)·P(Y|U)`，likelihood P(x|μ) + prior P(Y|μ)。
- Likelihood 更新：来图 x_i 映到 f_i^v，选最可能类 s = argmax P(U|x_i)，若 P(U[s]|x_i)>τ 则用 f_i^v 更新 μ_s（EMA/统计，DOTA 用高斯 `P(x|μ,Σ)=N(x|μ,Σ)`），计数器 C1[s]++。
- **Prior 更新（本文新点）**：prior P(Y|μ_s) 初始化为 one-hot，来样本后用后验 P(Y|x_i) 累计更新 `P(Y|μ_s) ← running mean of posterior`，计数器 C2[s]。
- 两个计数器 C1/C2 节奏不同分别控制 likelihood 和 prior 更新。

**需要什么输入**：**流式/在线单图序列**（TTA 设定，逐图到达持续 adapt）；无标注。

**第三方模型**：**否**。只 CLIP。✅

**训练/反传**：**否**。统计式在线更新，无梯度。✅

**计算量**：极轻，每图 O(K) 更新。

**能否嫁接（可移植性 3/5）**：
- 怎么接：**先验自适应**思想可嫁接到我们**背景中心化**这一步——把固定的 μ（50 张背景图算的先验）换成**在 batch/流上自适应更新的 prior**。这和 OTTER 目标类似（都在校 base-rate），但 BCA 是在线 EMA、更简单。
- 亮点：DOTA 式高斯 likelihood `N(x|μ_k, Σ_k)` 给每类估一个协方差，比纯 cos 更 calibrated——**这个可试**（test-time 轻量，每类协方差从 batch 估）。
- **多标签冲突（中）**：BCA 选 s=argmax 单类更新 + prior 是 K 维和为 1 的分布，是单标签在线范式。多标签下"选一个类更新"和"prior 归一"都要改。
- **和已知教训**：prior 更新不涉及跨类特征归一化，风险比 Dirichlet 小；但"自适应 prior"若估不准会漂移（论文自己也要 label reset 类机制）。
- 预期收益：中。自适应 base-rate 校正 + 高斯 likelihood 值得一试，但和 OTTER/我们已有背景中心化功能重叠。
- 结论：作为"背景中心化的自适应升级"候选之一，和 OTTER 二选一实验；高斯 likelihood 单独值得测。

---

## 6. Concept-Guided Bayesian (ConceptGuidedBayesian_2603.07911)

**一句话核心方法**：把 zero-shot 分类写成对"类的隐概念"边际化的 Bayes 后验；用 **LLM** 生成判别性概念构造 proposal 分布（DPP 选多样子集），再用一个 training-free 的 adaptive soft-trim likelihood 抑制离群概念。

**具体数学机制**：
- `P(Y_i|X) ≈ Σ_{C_{i,j}} P(Y_i|X,C_{i,j}) P(X|C_{i,j})`，对概念边际化。
- 概念 proposal q(C_i)：LLM 生成"A photo of {class} with {concept}"式判别概念（hard-negative 邻域对比 prompt），DPP 选多样子集降冗余。
- Adaptive soft-trim likelihood：对每类的概念-图相似度分布，用中位数 m_i、MAD_i 估污染率 ρ，`w_{i,j} = σ(-log((1-ρ)/ρ)·(S_{i,j}-m_i)/MAD_i)`，离群概念降权；后验 = 加权均值 Σ w_{i,j} P(Y_i|X,C_{i,j})。

**需要什么输入**：单图即可；无标注；**但离线要 LLM 生成概念**。

**第三方模型**：**是——需要 LLM 生成概念**。❌ 硬性排除项命中。

**训练/反传**：否（推理端 training-free）。但 LLM 依赖是致命伤。

**计算量**：推理轻（soft-trim 单次前向），但概念生成靠 LLM。

**能否嫁接（可移植性 2/5）**：
- LLM 依赖直接违反"不能引入第三方模型"。整套 concept synthesis 不可用。
- **唯一可剥离的干货**：**adaptive soft-trim likelihood**（中位数+MAD 的鲁棒均值/离群降权）是纯统计、无模型依赖。我们的 **CWR multi-crop 跨 crop max-pool** 可以借鉴：把 14 个 crop 的 per-label 分数看成一个分布，用 soft-trim 鲁棒聚合替代裸 max（max 对离群 crop 敏感）。这是一个小而美的可试点。
- 预期收益：低-中，仅限 soft-trim 聚合替换 crop max 这一处。
- 结论：主方法排除（LLM）；只偷 soft-trim 统计聚合思想到 CWR。

---

## 7. NoLA — No Labels Attached / CLIP meets DINO (CLIPmeetsDINO_2411.19346)

**一句话核心方法**：三步 label-free：① LLM 生成类描述做 CDE 分类器；② 用 CDE 伪标签把 **DINO** SSL 特征对齐到 VLM 空间做自动标注器；③ 用 DINO 伪标签在冻结 CLIP 上 **prompt-tuning**（FixMatch 式）。

**具体数学机制**：CDE = LLM 描述文本嵌入平均；DINO alignment module h 用 top-k 置信样本训练；再用 DINO labeler 生成伪标签，训练视觉 prompt θ_P（smoothed CE loss，弱/强增广两视图）。

**需要什么输入**：无标注 target 图集 + 类名。

**第三方模型**：**是——DINO + LLM**。❌
**训练/反传**：**是——训练 alignment module + prompt tuning（反传）**。❌

**能否嫁接（可移植性 1/5）**：双硬性排除（DINO/LLM + 训练反传）全中。整套架构与我们"单模型自洽、零训练"哲学完全相反。**无可嫁接部分**。仅作反例参考：它靠第二个视觉模型（DINO）补 CLIP 视觉特征之不足——而我们的立场是 v5-omni 视觉空间已足够好、不引第三方。

---

## 8. L2C — Learning to Complement Frozen CLIP (FrozenCLIPFewShotTTA_2506.17307)

**一句话核心方法**：few-shot test-time domain adaptation：在冻结 CLIP 旁挂一个并行 side branch（CPNet），用 revert attention 只学 CLIP 没有的 dataset-specific 知识来补全视觉特征；再用 greedy text ensemble + refinement 提升文本类间离散度；最后用 batch 生成的 domain prompt 做域感知融合。

**具体数学机制**：
- CPNet 并行：`Ĩ(x) = I(x) + A·CP(x)`，revert attention `A = 1 - softmax(CP(x)·I(x))`（只补差异信息）。
- Greedy text ensemble：按 uniformity `L_uni = Σ_{i≠j} exp(-t‖T_i-T_j‖²)` 贪心选能增大类间离散度的 prompt 模板集合。
- Text refinement：`T̃ = M_c·T·M_d + T`，两个可学矩阵沿类/特征维调整。
- Domain-aware fusion：batch 内实例间 attention 提域信息 + learnable domain cache → domain prompt 引导图文融合。

**需要什么输入**：源域标注训练 + few-shot 无标注 target（domain adaptation 设定）。

**第三方模型**：否（只 CLIP）。✅
**训练/反传**：**是——CPNet、refinement、domain cache 全部要在源域训练（反传）**。❌ 硬性排除命中。

**能否嫁接（可移植性 2/5）**：
- 主体（CPNet/domain prompt/refinement）都要训练，排除。
- **唯一无训练可剥离**：**greedy text ensemble**——用 uniformity（类间离散度）指标贪心筛选 prompt 模板，是纯 test-time 无训练操作。我们标签矩阵是 tokenizer 词表单 token 编码，没有 prompt 模板集，但思想可迁移：**若给标签加多模板/多描述编码，可用 uniformity 准则选最能拉开类间距的编码方式**，而不是全平均。
- 但注意：这又是"文本特征再处理"，和我们已知"特征再处理在 v5-omni 上无效/有害"部分冲突，需实测。
- 预期收益：低。
- 结论：主方法排除（训练）；greedy ensemble 思想可小试但预期收益低。

---

## 9. Collaborative Self-Learning / No Labels Needed (NoLabelsNeeded_2509.18938)

**一句话核心方法**：CLIP 选高置信种子样本 + 一个独立预训练视觉模型（**ViT-G-14**）提特征，在测试集上迭代自训练一个轻量线性分类器（伪标签自学习循环）。

**具体数学机制**：Step A 种子选择用 CLIP cos + 邻域一致性打分 `S(x)=(1/k)Σ cos(I(r_j(x)), T(l_i))`；Step B 用 ViT-G-14 特征训线性 softmax 分类器，自学习循环增量加高置信伪标签，loss 停止准则防过拟合；Step C 用最终分类器预测。

**需要什么输入**：无标注测试集 + 类名（transductive/整集）。

**第三方模型**：**是——独立的 ViT-G-14 (LAION) 特征提取器**（与 CLIP 分离）。❌
**训练/反传**：**是——迭代训练线性分类器**（虽轻量但是反传自学习循环）。❌

**能否嫁接（可移植性 1/5）**：双排除（第二视觉模型 + 训练分类器）。核心卖点就是"解耦 CLIP 与第二个视觉模型降 bias"，与我们单模型哲学相反。
- 可借鉴的碎片：**邻域一致性打分**（S(x) 用 k 近邻是否也指向同类来度量置信）——这是纯 test-time 无模型统计，可用于我们 batch 内给每个 (图,标签) 对做置信重排（近邻图也命中该标签则加强）。但这和 ZLaP 的图传播其实是同一思想的弱化版，有 ZLaP 就不需要它。
- 结论：整体排除，思想被 ZLaP 覆盖。

---

## 总排名表（按"可嫁接性 × 预期收益"排序）

| 排名 | 论文 | 第三方模型 | 需训练 | 多标签兼容 | 可移植性 | 预期收益 | 综合建议 |
|---|---|---|---|---|---|---|---|
| 1 | **ZLaP 标签传播 (CVPR24)** | 否✅ | 否✅ | 好（逐类传播）✅ | 5/5 | 高 | **首推试** |
| 2 | **OTTER 最优传输** | 否✅ | 否✅ | 需改边际约束⚠️ | 4/5 | 中-高 | **次推试**（当背景中心化升级） |
| 3 | ECALP 上下文标签传播 | 否✅ | 否✅ | 好✅ | 4/5 | 中 | ZLaP 备选；re-weighting 别抄 |
| 4 | BCA 贝叶斯 TTA | 否✅ | 否✅ | 需改⚠️ | 3/5 | 中 | 高斯 likelihood 可单独试 |
| 5 | Transductive EM-Dirichlet | 否✅ | 否✅ | 差（单纯形归一）❌ | 3/5 | 低-中 | 和"跨类归一化有害"相悖，缓 |
| 6 | Concept-Guided Bayesian | **LLM**❌ | 否 | — | 2/5 | 低 | 仅偷 soft-trim 聚合到 CWR |
| 7 | L2C Frozen CLIP TTA | 否 | **是**❌ | — | 2/5 | 低 | 仅 greedy text ensemble 思想 |
| 8 | NoLA (CLIP+DINO) | **DINO+LLM**❌ | **是**❌ | — | 1/5 | — | 排除，反例参考 |
| 9 | Collaborative Self-Learning | **ViT-G**❌ | **是**❌ | — | 1/5 | — | 排除，思想被 ZLaP 覆盖 |

---

## 推荐接下来试的 2-3 个方法（含怎么试）

### ★ 首选：ZLaP transductive 标签传播（可移植性 5/5）
**为什么**：唯一同时满足 ①无第三方模型 ②无训练 ③天然多标签（逐类传播不归一）④我们 batch 小算力零压力 的方法。
**怎么试**：
1. 对一 batch COCO 图，节点 = {2.5万标签词向量（或只取候选 top-N 标签，降维）+ batch 图像 global 向量}。
2. 按 ZLaP 分开建 kNN：图↔图、图↔标签分别建边，缓解图文 modality gap（v5-omni gap 小但仍建议分开）。
3. 归一化邻接 `Ŝ = D^{-1/2}(S+S^T)D^{-1/2}`，闭式解 `ŷ_c = (I-αŜ)^{-1} y_c`（我们规模小，直接求逆）。
4. 用传播后分数替换/融合当前 `S`，**保持逐类阈值/排序做 mAP，不 argmax**。
5. 调 k（近邻数）、α（传播系数）。每类可放多个文本节点（多 prompt / patch 类内 top crop 的文本近邻）增强传播源。
**预期**：batch 内图像互印证 → 稀有类召回提升，mAP 涨。**风险最低，先做这个。**

### ★ 次选：OTTER 最优传输做自适应 base-rate 校正（可移植性 4/5）
**为什么**：直接对标我们的背景中心化，用 batch 统计自适应校 base-rate，比固定 50 图 μ 更动态。
**怎么试**：
1. 代价矩阵 `C = -log softmax(S)`（S 为当前融合分）。
2. 图像边际先用等权，**但为多标签放松**：改用 unbalanced OT（KL 松弛行列边际）避免"每图质量固定"假设。
3. 类边际 ν：先用 batch 内 s_global 软估，或先验均匀，做消融。
4. Sinkhorn 几十次迭代，得校正后分数。
5. **和背景中心化对比 A/B**（可能功能重叠，二选一或叠加）。
**预期**：类分布不均时（COCO person 泛滥）压高频、提低频 → mAP 涨。**风险：多标签边际改造 + ν 估计。**

### ☆ 补充小试（低成本、独立）：CWR 聚合的 soft-trim 化（偷自 Concept-Guided Bayesian）
**为什么**：我们 CWR 现在是跨 14 crop 取 max，对离群 crop 敏感。
**怎么试**：把每个 (图,标签) 的 14 个 crop 分数当分布，用中位数 + MAD 估污染率，soft-trim 加权聚合替代裸 max（`w_j = σ(-log((1-ρ)/ρ)·(S_j-m)/MAD)`）。纯统计、无模型、无训练，改动局部。
**预期**：稳健性小涨，低风险，可和 ZLaP 并行做。

**执行顺序建议**：先 ZLaP（收益/风险比最好）→ 再 OTTER（需多标签改造，稍复杂）→ soft-trim CWR 作为随手 A/B。BCA 的高斯 likelihood 若前两者见效可作第四步深挖。EM-Dirichlet 与"跨类归一化有害"教训冲突，暂缓；后 4 篇因第三方模型/训练排除。

---

## 实测结果 (逐个实现, COCO-150)

### ZLaP — 失败 ❌ (mAP 0.635→0.14, prop-only)
实现: 80标签节点+150图节点建图, 图↔图/图↔标签分开 kNN, 闭式 (I-αŜ)⁻¹ 传播, 逐类不归一。
扫 kii∈{10,20,40} kil∈{5,10} α∈{0.8,0.9,0.99} 全崩 (mAP 0.11-0.17)。
**根因 (grounded, 非调参问题)**:
1. ZLaP 图建在图像**global**向量上, 而我们 global-only mAP 只有 0.264 (弱分类器)。LP 不能凭空造信号, garbage in garbage out; 我们的强信号在 patch+CWR (mAP 0.71), 图传播反而稀释。
2. **多标签破坏流形假设**: ZLaP 为单标签设计 (每图一主类, 图↔图边连同类图有效)。COCO 多标签下图A(person+car+dog)和图B(person+bike)只共享person, 传播把A的所有标签抹向B → 污染。这正是评估报告标注的风险坐实。
结论: ZLaP 不转移, 同其他方法的共同根因 (单标签/弱global假设不匹配我们)。

### OTTER — 失败/平 ❌ (mAP 0.693→0.699, 噪声级)
实现: 在 CWR 分数上做 Sinkhorn OT (balanced + unbalanced KL-relaxed), nu=uniform/batch-est。
bal-OT 单独 mAP 0.699 但 P@3 0.302/R@5 0.430 大跌 (OT 质量守恒压多标签召回); 混合 best+2*pi mAP 0.698 (+0.005 噪声)。
根因: 我们分数已被背景中心化+per-label mean-centering 校准好, OT 的 base-rate 校正冗余; 且 OT 质量守恒与多标签召回冲突。

### soft-trim CWR 聚合 — 失败 ❌ (max 已最优)
实现: 14 crop per-label 分数分布上试 median+MAD soft-trim / trimmed-mean / mean / logsumexp 替代 max。
结果: max=0.710 最佳; softtrim 0.671, mean 0.659, lse 0.673, trim0.7 持平 0.710。
根因: 小目标只有**一个** crop 包含它, max 就是正确证据; 任何均值/trim 都稀释这个信号。"max 对离群敏感"的直觉错了——离群就是信号。

## 总结 (3/3 嫏接方法均无提升, 诚实结论)
统一根因: **我们现有 pipeline (patch + CWR + 背景中心化) 已把这些方法想补的信号都捕获了**。这些论文为单标签/弱校准/弱global 场景设计, 而 v5-omni patch+CWR 已是强信号 + 良校准, 所以图传播/OT/鲁棒聚合要么稀释要么冗余。与之前"特征再处理(换层/白化/softmax)都无效"同一教训: v5-omni 本身已很好。真正有效的只有 CWR(喂更好的图)。
脚本: exp_zlap.py/exp_zlap2.py/exp_otter.py/exp_softtrim.py。

### BCA 高斯 likelihood + 自适应 prior — 失败/无效 ❌
- BCA-B 自适应 per-class prior (多标签 per-class logistic, batch 迭代): mAP 0.693 零变化。根因: metric 已做 per-class mean-centering, 再加 per-class log-prior (又一个 per-class 常数) 被中心化吸收, 对排序数学上是 no-op。
- BCA-A 高斯/Mahalanobis likelihood: 需 raw label 向量, 与已失败的白化同操作 (mAP 0.06), 合理跳过。

### EM-Dirichlet — 失败 ❌ (如预测)
- P@1 0.173 崩 (simplex 跨类归一化摧毁多标签)。与"跨类归一化有害"教训直接坐实。
- 脚本: exp_bca_dirichlet.py。

## 最终结论 (5/5 ttc 方法均无提升, 验证完毕)
ZLaP/OTTER/soft-trim/BCA/EM-Dirichlet 全部无提升。三个统一根因:
1. **我们分数已良校准** (背景中心化+metric mean-centering) -> 任何 per-class prior/base-rate 校正都是 no-op 或冗余 (BCA/OTTER)。
2. **多标签 ≠ 单标签**: 跨类归一化/simplex/质量守恒都摧多标签召回 (Dirichlet/OTTER)。
3. **信号已在 patch+CWR**: 图传播建在弱 global 上只会稀释 (ZLaP); max 聚合不能改 (小目标唯一 crop = 信号)。
同"特征再处理无效"一个元结论: **v5-omni 本身已很好, 唯一有效的是 CWR (往模型加新信息, 非重加工已有信息)**。天花板在 1B nano 模型本身。收口: CWR mAP 0.710/P@1 0.813。

