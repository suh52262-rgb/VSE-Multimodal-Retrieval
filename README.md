```markdown
# VSE-Multimodal: Dual-Stream Image-Text Retrieval 🚀

基于 PyTorch 实现的多模态双流图文检索模型（Vision-Semantic Embedding）。本项目在 Flickr8k 数据集上实现了高效的跨模态对齐（Image-to-Text & Text-to-Image）。

## 🌟 核心亮点 (Key Features)
* **双流架构 (Two-Stream Architecture)**：视觉端采用 `ResNet50`（砍掉分类头提取 2048 维特征），语言端采用 Hugging Face `DistilBERT`（提取 `[CLS]` 词向量），统一映射至 512 维联合语义空间。
* **InfoNCE Loss (对比学习)**：引入 OpenAI CLIP 同款的对比损失函数，并加入**可学习的温度系数 (Learnable Temperature)**，通过极大化对角线正样本概率，强力推开 Batch 内负样本。
* **工业级性能优化 (Performance Tuning)**：
  * 使用 **AMP (自动混合精度 `autocast`)** 结合 `GradScaler`，显存占用减半，激活 Tensor Core 加速训练。
  * 采用 **分层学习率 (Layer-wise LR)**：预训练骨干网络保留极小学习率 (`1e-5`) 防灾难性遗忘，投影层使用较大学习率 (`5e-4`) 加速收敛。
  * 结合 **余弦退火 (CosineAnnealingLR)** 与动态数据抽样，完美解决过拟合与 Batch 假负例冲突。

## 📊 评测结果 (Results on Flickr8k Val Set)
在仅使用单张 RTX 3090/4090 训练的情况下，模型在 50 Epochs 后取得了如下跨模态检索成绩：

| Task | Recall@1 | Recall@5 | Recall@10 |
| :--- | :---: | :---: | :---: |
| **Image-to-Text** | 28.80% | 58.20% | 71.70% |
| **Text-to-Image** | 25.60% | 56.50% | 69.80% |

## 🛠️ 快速开始 (Quick Start)
1. **环境配置**：
   ```bash
   conda create -n vse_env python=3.11 -y
   conda activate vse_env
   pip install -r requirements.txt
```
2. **数据准备**：下载 Flickr8k 数据集，放置于项目中并修改 `train.py` 中的路径配置。
3. **开始训练**：
   
   ```bash
   python train.py
   ```
4. **模型评估**：
   ```bash
   python test.py
   ```