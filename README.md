# Topo-DiT v2

面向潜空间视频流匹配的平衡路由研究原型。当前已有 CPU 正确性验证，**没有视频训练结果或生成质量结论**。

## 从这里开始：不训练、不下载视频模型

使用已有的 PyTorch 环境，在项目根目录执行：

~~~powershell
python -m unittest discover -s tests -v
~~~

测试仅运行 CPU 小张量、数值梯度、Euler 积分及临时检查点读写，不做优化器更新，不加载 Cosmos。核心依赖见 requirements.txt；使用 PyTorch 官方适配本机的安装方式。不要为运行检查而安装大型视频模型或创建完整配置中的大模型。

- 中文论文：topo_dit_chinese.md
- 英文论文：topo_dit.md；排版稿：output/pdf/topo_dit_revised.pdf（根目录 topo_dit.pdf 同步）
- 微型配置：configs/topodit_cpu.yaml
- 较大架构示例：configs/topodit_v1_k256.yaml；文件名保留历史标识，内容已升级为 v2，不是本机训练建议
- 改动前论文、代码和配置：archive/v0_2026-09-25/
- REVIEW_2026-09-25.md 是修改前评估，不能当作当前缺陷清单

## 数据契约

网络只接收连续 latent：z_t 的形状为 (B,C,T_l,H_l,W_l)，t 为 (B,)，condition 为 (B,D_c)。输出速度和 z_t 完全同形。codec 与网络解耦，模型构造时不加载权重，不隐式选择 CUDA。

正式训练读取预先缓存的张量。每个 latent 文件由 torch.save 写入一个浮点 Tensor，形状为 (C,T_l,H_l,W_l)，已应用 latent_scale。**数据加载时不会再次缩放。** JSONL 路径相对于 manifest 所在目录；一个 batch 内 latent 形状必须一致，暂不提供变长批处理。

无条件记录示例：

~~~json
{"latent_path":"clip_000.pt","codec_id":"exact-checkpoint-id","latent_scale":1.0}
~~~

有条件记录示例：

~~~json
{"latent_path":"clip_000.pt","codec_id":"exact-checkpoint-id","latent_scale":1.0,"condition_path":"text_000.pt","condition_id":"clip-vit-large-patch14-revision-X"}
~~~

配置中的 codec_id、latent_scale 和 condition_id 必须一致。条件张量为 (D_c,)，无条件模式固定为零向量，绝不随机伪造文本。默认 CLIP ViT-L 特征是 768 维，使用时将 cond_dim 配为 768，并设置 conditioning: cached 与明确的 condition_id。encoder 的身份由缓存创建者提供，不自动认证远程权重。

## 可选：在有合适资源的机器上准备缓存

本次未运行以下命令。prepare_latents.py 使用 imageio 读取视频前缀、等步长抽帧、保持宽高比缩放后中心裁剪，再用冻结 Cosmos continuous JIT 编码。短视频直接报错，不循环制造虚假运动；损坏文件直接报告，不无限递归。所需可选包见 requirements-video.txt。

输入 manifest 的每行含 video_path，可选已生成的 condition_path 和 condition_id。需要用户已有的可信 encoder.jit / decoder.jit；不会自动下载。请根据真实连续 tokenizer 检查点设置压缩率，CV8×8×8 与 CV8×16×16 不能混用。

~~~powershell
python scripts/prepare_latents.py --input-manifest data/videos.jsonl --output-dir data/cache_run1 --codec-dir Cosmos-Tokenizer --codec-id YOUR_EXACT_CODEC_ID --temporal-compression 8 --spatial-compression 8 --scale 1.0 --device cuda
~~~

编码器校验输出连续浮点 latent 的通道与时空形状。scale=1 是显式中性默认值，不声称 latent 已标准化；如果估计其他 scaling，应记录并在所有数据与解码中一致使用。VAE 的真实权重路径尚未在本机验证。

## 训练入口（仅供后续具备资源时使用）

本次没有运行训练命令。先准备有效缓存并修改配置 data.meta_path 与 codec_id。可以跳过预热直接进行 Phase 2。

~~~powershell
python train_phase1_autoenc.py --config YOUR_CONFIG.yaml --device cuda --save_dir checkpoints/warmup
python train_phase2_flow.py --config YOUR_CONFIG.yaml --device cuda --save_dir checkpoints/flow
~~~

Phase 1 是遮挡重建，只在隐藏位置监督，没有原网格复制旁路。Phase 2 是潜空间网格流匹配，路由与主干共同适应噪声输入。可选 --ckpt_phase1 checkpoints/warmup/epoch_0010.pt 迁移预热权重，或 --resume checkpoints/flow/epoch_0005.pt 恢复同阶段优化器与 CPU RNG。两参数互斥。恢复时 max_epochs 表示总目标 epoch；CPU 恢复逻辑不承诺跨设备或多 GPU 逐位复现。

每个 epoch 保存版本化检查点、配置和 JSONL 指标，包括边际残差。旧版 v0 权重明确拒绝加载，避免 silent strict=False。历史 scripts/ 同名入口调用同一实现。

## 推理与编辑

采样从 checkpoint 恢复模型配置和 attention 头数，不另猜架构。无条件 latent 采样示例（必须使用后续训练得到的 Phase 2/3 权重）：

~~~powershell
python inference_edit.py --ckpt checkpoints/flow/epoch_0010.pt --latent-shape 3 16 16 --steps 50 --output output/sample.pt --device cuda
~~~

有条件模型需增加 --condition text.pt --condition-id MATCHING_ID，不接收未编码的 prompt 冒充文本支持。可选 --edit-index 0 --edit-scale 0 是槽位特征干预，不能解释为冻结背景或物理速度。

需实际 MP4 时加 --codec-dir Cosmos-Tokenizer --codec-id MATCHING_ID --video-output output/sample.mp4，并提供正确的压缩率。解码以 tokenizer 的原生尺寸输出，帧数/尺寸由 latent 形状与 codec 决定。程序先保存 latent，只有写出 MP4 才报告视频已保存。质量取决于尚未开展的训练。

## 方法与消融

model.routing 支持 sinkhorn、softmax、uniform。默认 Sinkhorn temperature=0.2、30 次迭代；列边际只近似满足，需监测 column_relative_error。低温或更难的代价可能需要更多迭代。

三种模式保持相同的网络模块，但实际路由耗时不同，需测量。均匀路由是退化控制，不能替代独立 read/write attention 基线。lambda_ent、lambda_ortho 默认 0；这些正则不能保证对象语义。输入位置是加法正弦编码，非 RoPE。最终广播没有直接 latent 残差，但 A 矩阵仍保存布局，整个模型不具备 K-only 信息或显存瓶颈。

## PDF 重建

scripts/render_paper.py 使用 reportlab 和 Windows Arial 字体 将英文 Markdown 与公式排成 PDF；它不运行模型。现有 Codex 文档运行时包含所需包；公式以与源码等价的排版表达式输出。示例：

~~~powershell
python scripts/render_paper.py
~~~

## 验证边界

单元测试通过只说明已检查的公式、形状、梯度和 I/O 契约成立，不证明视频生成、对象发现、物理推理或效率优势。真实 Cosmos 权重、实际 MP4 解码和 GPU 训练未在本机执行。实验计划与参考文献见论文第 6 节。
