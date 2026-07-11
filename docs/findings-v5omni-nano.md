# v5-omni-nano-mlx 图片打标签 — 实测发现 (grounded, 2026-07-10)

模型: jinaai/jina-embeddings-v5-omni-nano-mlx (完整带 vision+audio)
本地: /Volumes/vault/ai-models/jinaai/jina-embeddings-v5-omni-nano-mlx

## 架构 (实测确认, 非猜测)
- text backbone: EuroBERT/LLaMA-style 12 层 hidden 768, **bidirectional (非 causal)**
- **last-token pooling + L2 norm** (omni 系列训练用 last-token, 不是 mean)
- vision tower: Qwen3VL 12 层; audio tower: Qwen2.5-Omni
- vocab_size 128260; embed_tokens.weight [128260, 768]
- **tie_word_embeddings = FALSE** (关键!)
- 输出 768 dim, Matryoshka 可截 32/64/128/256/512
- image_token_id = 128259; encode_image(pixel_values, image_grid_thw, input_ids, attention_mask) 完整可用
- 预处理: Qwen2VLImageProcessor (image_mean/std=0.5, patch 16, merge 2), min_pixels 262144 max 1310720

## 关键结论 1: Han 的 "直接走 tokenizer 词表" intuition — 原始形式否决
- 试了: 图片 pooled 向量 直接点积 embed_tokens.weight [128260,768] 出 vocab 分数。
- 结果 GARBAGE: 三图 top 全是 Tutor/avatar/PyTuple/quisites 碎片, 且不同图结果雷同, 无语义。
- 根因: tie_word_embeddings=FALSE -> **输入词嵌入空间 != last-token pooled 输出空间**, 直接比是错的。
- 脚本: probe_vocab_tagging.py

## 关键结论 2: 正确的标签空间 = encode_text pooled
- 图片 pooled 输出跟 **文本整句 pooled 输出** 对齐 (模型就这么训的 image<->text)。
- 所以标签要走完整 encode_text 编码成 pooled 向量, 再跟图片 cos。
- Han intuition 的 solid 落地: 把 tokenizer 词表里有意义的词 (过滤碎片/special) 当候选文本标签, encode_text 成标签矩阵, 图片跟它算 cos 做 multi-label。

## 关键结论 3: 跨图 per-label 标准化 (z-norm) 是决定性的
- raw cosine 有严重 base-rate 病: "bed"/"cat" 几乎每张图都排最前 (某些标签天生离图片模态更近)。
- 跨图 per-label z-score (减均值除标准差) 后立刻对位:
  - cat.jpg -> remote control/cat/couch/television ✅
  - tennis(滑雪) -> snowboard/skis/snow/mountain ✅
  - skate(厨房) -> kitchen/refrigerator/sink/microwave/oven ✅
- 这不是启发式作弊, 是 CLIP/TagCLIP 标准多标签校准 (TagCLIP 用 per-image min-max; 我们批量场景用 per-label centering)。
- prompt ensemble (a photo of a {} 等 5 模板平均) 标准做法, 已用。

## 基线成绩
- 7 张 verified-GT 测试图, 50 类标签: **hit@5 = 5/7** (raw 也 5/7 但排序烂, z-norm 排序对)。
- 1B nano + 纯 embedding + 零训练。扎实基线。

## 失败案例 (诚实)
- zebra.jpg(实为熊) / bus.jpg(实为 stop sign): raw 分数本身低且平, 模型对这两图判别信号弱, z-norm 放大噪声 -> 排序乱。
- 对应 PIAA 论文的 patch 判别性问题。改进方向: patch-level (不只 pooled) + 更强校准。

## 测试图 GT (COCO 文件名不可信, 已 VLM 核实)
- cat.jpg: 猫+沙发+遥控器 | livingroom_tv.jpg: 电视+餐厅椅 | cat_shoe.jpg: 猫+鞋
- zebra.jpg: 熊+草 | bus.jpg: stop sign | tennis.jpg: 滑雪者+雪山 | skate.jpg: 厨房+冰箱

## 关键结论 4: 开放全词表 (15k) 实测 — 比 50 curated 词差, 诊断清楚
- tokenizer vocab 128260 -> 过滤碎片 -> 交 /usr/share/dict/words -> 15006 真词标签 (Han intuition 完整版)。
- 三种计分: raw cos / 跨图 z-norm(7图) / 50图背景先验标准化 (label prior 减除, 标准 open-vocab 校准)。
- 结果: 50 curated 词 hit@5=5/7; 开放 15k hit@5 只 1/7 (exact match)。
- **但语义其实是对的**: cat.jpg->kitty/kat/cat/fluffy/kitten; skate(厨房)->cupboard/kitchen/cabinet; tennis(滑雪)->snowy/snow/olympic/chilly; livingroom->fireplace/apartment/room/furniture。概念簇全对, 只是 exact GT 词被近义词/同义词挤下去 (kitty>cat, snowy>snow, cupboard>cabinet)。
- **两个污染源 (grounded)**: (a) 近义词稀释 exact match; (b) 专有名词污染 (california/kristen/korean/texas 冒高分, 文本 embedding 对地名人名有 spurious 亲和, 无视觉 grounding)。
- diagnose.py 证实: cat 在 15k 里 rawrank=94/15006 (top 0.6%, 信号真实存在), 但 93 个抽象词挤在前面。信号有, 是标签空间太脏。

## 结论: Han intuition 成立, 但标签空间要收敛到"具体物体名词"
- 不是用全 dict 词 (含抽象词/地名/人名, 无视觉可 ground), 而是限定 **具体物体名词** (WordNet physical-object / entity synsets)。
- 这仍保留 Han "用 tokenizer 自带词表, 无外部 tag list" 精神 (从 vocab 里筛), 只是筛选标准换成"可视觉 grounding 的名词"。
- 背景先验校准 (50 图 label prior 减除) 是对的方向, 保留。

## 下一步
1. 用 WordNet 把 15k vocab 词筛到具体物体名词 (physical entity synsets), 重跑评测。
2. 近义词聚合: 同义词归一 (kitty/kat/cat -> cat) 或输出概念而非 exact token。
3. patch-level 提升弱判别图 (zebra->bear / bus->sign 这种 raw 信号弱的; 借 PIAA 思路, 图文同空间 gap 更小)。
4. 阈值策略: 单图 top-k + 背景标准化 margin。

## 关键结论 5: WordNet 具体名词过滤 + 背景校准 (v3/v4 实测)
- v2 全 15k dict 词 + 50图背景校准: hit@5=1/7
- v3 WordNet physical_entity 过滤 (8677词): hit@5=3/7
- v4 WordNet lexname 精筛 (animal/artifact/food/plant/body/object/substance, 排 location/专有名词) (4865词): hit@5=3/7
- **6/7 图概念簇完全对**: cat->kitty/kat/cat/cats/kitten; skate(厨房)->cupboard/cabinets/kitchens/kitchen; tennis(滑雪)->snow/glaciers/sled/glacier; livingroom->fireplace/mansion/rooms/apartment/sofas。
- **唯一真失败 zebra(实为熊) / bus(实为 stop sign)**: 图本身 raw 信号弱 (bear rawrank 5305/15k)。pooled [CLS] 向量对小/非显著目标判别不足 — 正是 PIAA 的 patch-判别性问题。要上 patch-level 才能救。
- WordNet 过滤仍漏碎片 (ion/ers/ass 有奇怪名词义) + 少量地名(cocos/costa)。不影响主结论。

## 总结 (给 Han)
1. **Han intuition 成立**: 用 v5-omni-nano 自带 tokenizer 词表做图片多标签, 不需外部 tag list, 可行。但不能直接点积 embed_tokens (tie=false 空间不通), 必须 encode_text 把词编成标签向量。
2. **三件事决定质量** (都不是启发式, 都是 CLIP/论文标准做法): (a) encode_text 对齐空间; (b) 标签限定具体物体名词 (WordNet); (c) 背景 label-prior 减除校准。
3. **弱项**: 小/非显著物体 (需 patch-level) + 近义词归一 (kitty=cat)。
4. 成绩: 1B nano 纯 embedding 零训练, 50 curated 词 hit@5=5/7, 开放~5k 具体名词 hit@5=3/7 且概念簇 6/7 对。

## 下一步
1. patch-level tagging (借 PIAA/TagCLIP): 不只 last-token pooled, 用 vision patch token 跟标签算分 max-pool -> 救 zebra/bus 弱信号。
2. 近义词归一 (WordNet synset 聚合 kitty/kat/cat)。
3. 阈值策略: 单图 top-k + 背景标准化 margin。

## 关键结论 6: 无 WordNet/regex/dict 的纯数据驱动版 (Han 2026-07-10 要求)
Han 要求: 不要 heavy 训练, 不要 WordNet, 不要 regex 查表。全词表上。
- 全 tokenizer 词表 128260 全部 encode_text (bare token string, 40s, cache full_vocab_emb.npz)。
- **背景中心化 (centered = cos - bg_mean)** 是正确的无资源 base-rate 去除: 非视觉词对所有图相似度均匀 -> centered≈0 自然被压; 只有尖峰词存活。centered 明显优于 z-score (z 把代码token方差放大成噪)。
- **问题**: tokenizer 词表 ≠ 概念表。实测只 33.6% 是纯小写 a-z 词, 66% 是 BPE碎片/代码符号/多语言子词。centered top-k 被同概念多语言变体(猫/Cat/кот/cat)+代码token(GetComponent/_cpp)污染。
- **embedding-NMS 去重** (贪婪, kept之间 cos<tau): 成功消同概念变体, 纯靠 embedding 几何, 零查表。
- **自一致性门** (bare vs 'a photo of a {}' 方向一致): 失败, 碎片在模板下也自洽 (124947/128260 过阈), 没用。
- **✅ 最佳解: tokenizer 自带 word-start 信号 'G̈' (空格前缀)**。byte-BPE 里 'G̈word' = 模型自己认为的整词(非词中碎片)。这是 tokenizer 内生信号, 不是外部资源。gate=word-start+alpha+ascii+lower -> 25465 词。+centered+NMS: **hit@5=4/7**。

## 最终结论 (无外部词典版)
- Han intuition (全 tokenizer 词表 multi-label) 成立。流水线: encode_text 全词表 -> centered(背景减除) -> word-start gate(tokenizer 内生) -> embedding-NMS。全程零 WordNet/dict/语义regex。
- 剩下的失败 (zebra=熊, bus=stop sign) 与最初一样, 是小/非显著目标 pooled 向量判别性不足 (PIAA 问题), 不是过滤问题。patch-level 才能救。
- 成绩总览: 50 curated 词 5/7 | WordNet筛~5k 词 3/7 | 无词典 word-start 全词表 4/7 (且概念簇 6/7 对)。

## 关键结论 7: patch-level (PIAA 思路) — hit@5 4/7 -> 5/7
- 取法 (无模型改造): encode_image 内部把 merged vision 特征注入 image-token 位置跑双向文本塔。TextModel 返回全序列 hidden [1,L,768]。序列全是 image token, 每行 = 一个 patch 在对齐 768 空间的上下文化特征。
- 打分: 每个 patch vs 每个 label cos, 类内 max-pool over patches (PIAA: 小/非显著目标只在少数 patch 着火, global pooled 会 miss)。
- PAA 融合 score = a*(patch_max - bg) + (1-a)*(global - bg)。a=1.0 单 patch=3/7; a=0.7 (patch主+global锤) = 5/7 最优 (呼应 PIAA 用 0.9)。
- 提升点正是预测的弱信号图: skate refrigerator 现 exact 命中; zebra(熊) patch 出 wolf/roar/muzzle/claw/hunting (动物捕食者簇); bus(stop sign) 出 trees/sidewalk/parking 街景。
- 仍未 exact 命中 bear/sign: 目标小且非典型, 1B nano 极限。
- 脚本 tagger_patch.py — 当前最佳, 全部符合 Han 约束 (无训练/无WordNet/无regex查表)。

## 关键结论 8: Han 两个升级方向实测 (TTA + adj-noun)
**方向A TTA (多 augmented 输入提取/压制信号)**: 4 views (orig/hflip/crop0.7/crop0.5), 三种聚合 union-patch-max / mean / mean-std。结果 union=4/7 mean=3/7 都 **不如 baseline patch 5/7**。诊断: center-crop 假设物体居中, 全景图(tennis/bus)被 crop 丢上下文反而退步 (tennis 从 HIT 变 miss); 多 view 混合稀释了单 view 强信号。结论: 朴素 TTA(hflip+center-crop) 不适合, 需物体感知的 crop (得先定位) 才有用。tagger_tta.py。
**方向B 形容词+名词 (多主体修饰关系)**: stage1 patch 出名词, stage2 对每名词拼 "{adj} {noun}" 短语打分取最佳 adj。**失败**: 选出的 adj 全是垃圾(stunned/overdue/figsize/death)。根因(grounded): image-text sim 是 bag-of-words, 拼任何高-image-sim 词都抬高 phrase@image, margin 区分不了"真形容词"vs"任意共现高分词"。没有 POS 信号(Han 不要查表)时, sim 单独无法强制 adj-noun 语法结构。tagger_adjnoun.py。
- **这两个都是真实负结果, 不粉饰**。adj-noun 需重新 formulate (见下)。

## 关键结论 9: 真 benchmark (150 COCO 图, 80 类) + 延迟
之前 7 图太小无法区分 5/7 vs 6/7。建了真评测: COCO val2017 抽 150 图(真 multi-label GT, avg 2.93 label/img), 80 类闭集。
metrics (per-label mean centering):
- global(pooled): P@1=0.433 P@3=0.289 R@5=0.449 mAP=0.264
- **patch(max): P@1=0.753 P@3=0.418 R@5=0.623 mAP=0.626**
- **fuse a=0.7: P@1=0.753 P@3=0.427 R@5=0.631 mAP=0.635 (最优)**
- 结论: patch-level 在真数据上 mAP 0.264->0.635, 巨大提升。P@1=0.753=四分之三图首标签正确。PIAA 论点(global不足)在 v5-omni-nano 上实证。a=0.7 确认最优。
- **延迟**: 单图 ~55-73ms (vision 25ms + text 7ms + score matmul over 128k 词 25ms + prep 3ms) = ~16图/s。score matmul 占一半, 可优化(降维/子集)。TTA×4 view 会×4 延迟, 性价比差。
- bench_latency.py, eval_coco.py, build_eval_set.py, eval set eval_imgs/+eval_gt.json, cache eval_sims.npz。

## 关键结论 10: special-token/attention hack (Han 要求) — 7图上未胜基线, 需在 benchmark 重测
H1 prompt-anchor(image token 前插 'a photo of'): 4/7 降。H2 softpool(patch softmax池化代 max): 5/7 平。H3 anchor+soft: 4/7。但 7图无区分度, 需在 150图 benchmark 重跑。tagger_hack.py。

## 关键结论 11: hacks 在 150图 benchmark 重测
- anchor-prefix: mAP 0.628 vs 0.635 基线, 没用, 砍。
- **softpool (patch softmax 池化代 max, T=0.05): P@1=0.773 P@3=0.449 R@5=0.649 均最高**, 但 mAP 0.608 略降。即 top-k 精度更好, 全排序略差。
- 实用: 打标签要前几个对 -> softpool T=0.05 (P@1 0.773); 要全排序 mAP -> patch max a=0.7。

## 关键结论 12: patch-LOCAL 形容词 (解决了 adj-noun!)
- 思路: 名词找到它 fire 最强的 patch (support region) -> 池化成局部向量 -> 形容词候选只在**这个局部区域**打分。真属性在物体 patch 上高分, 无关共现词不在 -> 物理绑定修饰关系, 替代 POS。
- 结果 (vs global bag-of-words 垃圾 stunned/figsize/death 完全消失): cat->grey/fleece/blanket(猫真是灰的); zebra(熊)->hairy/fur/claws; tennis->snowy/steep/uphill/rocky; cat_shoe->denim/nike/buckle; bus->pine/parking/pole。全是真视觉 grounding 的属性/相关词。
- 无 POS 时无法强制纯形容词(混相关名词如couch/blanket), 但都 grounded 到该物体区域。tagger_adjnoun_local.py。

## 总结 (Han 两方向都解决)
- TTA: 确认不值(降性能+×4延迟), 弃。
- adj-noun: patch-local 打分解决, 无 POS/无查表。
- special-token hack: anchor 无用, softpool T=0.05 提 top-k。
- benchmark + 延迟(60ms) 建好。

## 下一步
固化成可复用 CLI: patch a=0.7 (或 softpool T=0.05) + 可选 patch-local 形容词。
2. adj-noun 重新 formulate 思路: (a) adj 候选限定为自身 patch-视觉-grounded 且属属性词区; (b) 用 patch-level 而非 global 算 phrase (拒拒 bag-of-words); (c) 或反向: 先定名词 patch 位置, 只在该位置局部区域重新打分 adj。
3. TTA 如要用: 得物体定位(patch 高响应区)后 crop, 不是盲 center-crop。

## 脚本
- probe_vocab_tagging.py (否决 embed_tokens 直接点积)
- probe_text_label_space.py (验证 encode_text 空间通)
- eval_tagging.py (50 curated 词, raw vs z-norm)
- encode_full_vocab.py (全 128260 词表 encode, cache)
- tagger_datadriven.py (全词表 centered vs zscore)
- tagger_nms.py (+embedding NMS)
- tagger_selfconsistent.py (自一致性门, 无效)
- **tagger_wordstart.py (word-start gate + centered + NMS, 无外部词典, hit@5=4/7) <- 当前最佳/最符合 Han 约束**
- (历史含外部资源版 build_vocab_labels/vocab_tagger_v2/v3/v4, 已被 Han 约束否决, 保留作对照)

## 脚本
- probe_vocab_tagging.py (否决 embed_tokens 直接点积)
- probe_text_label_space.py (验证 encode_text 空间通)
- eval_tagging.py (7图50类 完整评测 raw vs z-norm)
