# 正确率提升 birdview 设计备忘 (latency 不限)

## 当前 pipeline (baseline)
labels=tokenizer词表 → encode_text → patch cos, 类内 max-pool + global 融合(α=0.7)
→ 背景中心化(减 per-label 均值 μ) → Ġ gate → embedding-NMS。
benchmark: patch mAP 0.635 / P@1 0.753。唯一失败模式: **小/非显著目标** (zebra里的熊、bus里的stop sign, raw信号弱)。

## 相比 dataroom SOTA, 我们还没用的杠杆 (按 ROI 排序)

### 杠杆1 (最高ROI): CWR 裁剪-重识别 (TagCLIP CWR + 我之前提的 object-aware crop)
- **对症**: 唯一失败模式就是小目标。TagCLIP 的 CWR 正是为此: 对候选类取高响应 patch 组成区域 → 从**原图裁出该区域 resize → 重新过 vision tower 编码 → 重打分**。小目标被放大到占满 crop, 信号从弱变强。
- 之前 TTA 失败是因为**盲目 center-crop**(假设物体居中); CWR 是**响应引导 crop**(先定位再裁), 正是我诊断 TTA 失败时说的正确做法。
- latency 之前是唯一障碍(每类多一次 vision forward), 现在解禁。
- 预期: 直接救 zebra/bus 这类, mAP 天花板上移。**这是 80/20 的那个 20。**

### 杠杆2: patch 特征取哪一层 (TagCLIP 核心发现 + VisualDiscriminability)
- TagCLIP 铁律: CLIP **最后一层 self-attention 破坏空间信息**, 要用**倒数第二层** patch token。
- 我们现在取的是**全 12 层 + final norm 之后**的 hidden。很可能末层双向 attention 也把 patch 特征"全局化"了, 损失 patch 判别性 (VisualDiscriminability 同样诊断: 末层 attn/FFN 提对齐但杀判别性)。
- **动作**: 测 L-1 层(甚至 L-2) 的 patch 特征 vs 现在的末层。可能白捡精度。极便宜。

### 杠杆3: 特征空间白化 / GDA (PIAA PVCL 核心)
- PIAA 整篇论点: patch↔text 有 **modality gap**, 不该拿 patch 直接跟 text label 点积。它用 50张无标注图 patch 闭式解 GDA (w_c=Σ⁻¹μ_c) 学**视觉侧 classifier**。
- 我们现在只做了"减均值"(mean-centering), 是最弱版的 prior 去除。GDA 更进一步: 用背景 patch 估**协方差 Σ**, 做 Mahalanobis/白化, 再算相似度。
- 我们优势: v5-omni 图文**同空间**(比 CLIP 双塔 gap 本就小), 所以可能不需要完整 GDA, 但**白化(减均值+除协方差)几乎肯定比只减均值好**。Σ⁻¹ 预计算, 推理零额外开销。

### 杠杆4: patch 分数 softmax-over-classes (TagCLIP + PIAA 都强调)
- 两篇都明说: CLIP patch 打分**先 softmax over classes 很关键**(制造类间竞争, 压背景)。
- 我们现在是 raw cos 直接 max-pool, **没有类间竞争归一化**。
- 动作: max-pool 前对每个 patch 做 softmax over labels。便宜, 但全词表12万类 softmax 要小心(可先在候选子集上)。

## 次级杠杆 (设计/正确性)
- **多描述 label 扩展 (RAM++)**: 每个词 → 几个描述短语 encode 平均, 图自适应加权。比单 bare token 编码更稳(我们全词表现在用 bare token string, 无模板!)。RAM++ 证明这个涨点明显。
- **prompt ensemble 用到全词表**: 现在只在 80类 benchmark 用了3模板, 全词表 tagger 用的是 bare string。补上模板集成。
- **DMAR 注意力精炼 (TagCLIP)**: 用文本塔 patch-patch attention affinity 平滑 patch 分数图去噪。
- **自适应阈值** (现在固定 top-k): 单图 top-k + margin, 或 per-image 分布拐点。
- **label 共现先验 (C2SRT)**: 用共现结构 boost 一致标签组。

## 推荐落地顺序 (先验证再叠加, 每步在 150图 benchmark 量化)
1. **杠杆2 (换层)** — 最便宜, 先测 L-1/L-2 patch, 定 base。
2. **杠杆4 (softmax over classes)** — 便宜, 叠加。
3. **杠杆3 (白化/GDA)** — 中等, 用现有 50 背景图估 Σ。
4. **杠杆1 (CWR 响应引导 crop 重识别)** — 最重, 救小目标, 期望最大涨点。
5. 次级 (多描述/DMAR/阈值) 按边际收益补。

## 不做 (已验证负或不符约束)
- TTA 盲 center-crop (降点, 已砍) — 但 CWR 是它的正确形态。
- WordNet/dict/POS/regex 查表 (Han 约束)。
- 任何训练/反传。

## 实测结果 (COCO-150 benchmark, latency 不限)
基线 patch fuse a=0.7: mAP 0.635 / P@1 0.753。
- **杠扢2 (换层): 不转移。** L10(L-2) mAP 0.16-0.27 崩; L11(L-1)≈L12(final)。TagCLIP "去末层 self-attn" 对 CLIP-ViT 成立, 但 v5-omni 是 **双向 embedding 模型 + last-token pool**, 末层就是训练好的输出空间, 不该动。baseline 层选已最优。
- **杠扢4 (softmax-over-classes): 平。** best 0.636 vs 0.635, TagCLIP 的 softmax trick 也不转移。
- **杠扢3 (白化/GDA): 灾难。** full whiten mAP 0.06 (崩)。center-only≈0.634≈baseline。v5-omni 同空间已对齐, 白化反而破坏结构; 用 patch 协方差白化 label 也是分布错配。
- **✅ 杠扢1 (CWR 多-crop 重编码): 真胜。** 5-crop(2x2+center) per-label max fuse: base+0.8 -> **mAP 0.693 / P@1 0.813**。细 grid(3x3+2x2+center=14crop) base+1.3 -> **mAP 0.710 / P@3 0.476 / R@5 0.680**。
- **核心结论**: v5-omni 本身特征空间已很好(所以换层/白化/softmax 都白塔/有害), 正确率天花板不在特征再处理, 而在**弱信号/小目标失败模式**。multi-crop 重编码直接治本(物体在含它的 crop 里变大->信号变强), per-label MAX 不稀释(区别于 TTA 均值)。这就是 TTA 做对的形态。
- 脚本: exp_layers.py / exp_whiten.py / exp_cwr.py / exp_cwr2.py。cache: layer_feats.npz / bg_rawpatch.npy / eval_rawpatch.npz / cwr_scores.npz / cwr2_scores.npz。

## 下一步
1. 把 CWR (细 grid multi-crop max fuse) 写进 tag_image.py 作为 --hq 模式 (latency 换精度)。
2. 可选: 响应引导 bbox crop (比固定 grid 更准, 但 grid 已够好)。
3. 剩余次级杠杆 (RAM++ 多描述 label / DMAR) 边际收益递减, 暂缓。
