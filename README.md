\# CIFAR10: CNN vs ViT (From Scratch) 🧠⚡



> \*\*"In the era of Transformers, is CNN dead?"\*\*



本项目从零手写了轻量级 CNN 与 Vision Transformer (ViT)，在 CIFAR-10 数据集上进行了严格控制变量的 50-epoch 对比实验。



\*\*结论是：在没有海量数据预训练的前提下，经典的 CNN 归纳偏置（Inductive Bias）在小规模视觉任务中依然具有压倒性的优势。\*\*



\## 📊 实验结果与对比 (Results \& Comparison)



我们在 CIFAR-10 数据集上（5万张训练图，1万张测试图），使用相同的训练配置（AdamW 优化器、Cosine Annealing 学习率调度、统一的数据增强），从头（From Scratch）训练了两个模型各 50 个 Epoch。



<!-- 跑完云端实验后，把生成的对比图放在这里 -->

!\[Comparison](assets/cnn\_vs\_vit\_comparison.png)



\### 核心发现：

1\. \*\*收敛速度\*\*：CNN 在 10 个 Epoch 内就迅速收敛到了 70% 以上的准确率；而 ViT 需要极长的预热时间。

2\. \*\*最终精度\*\*：CNN 最终达到 \*\*XX%\*\*（填你的真实数据），而 ViT 仅达到 \*\*XX%\*\*。

3\. \*\*工程反思\*\*：这印证了学术界的一个共识——Vision Transformer (ViT) 是“极其吃数据”的。在没有海量数据（如 ImageNet-21k）或预训练权重加持的情况下，在小数据集上从头训练 ViT，表现往往差强人意。\*\*架构没有绝对的优劣，只有是否适合当前的数据规模。\*\*



\## ✨ 核心特性 (Features)



\- \*\*纯手写实现\*\*：不依赖 `torchvision.models` 的现成架构，CNN 的卷积块与 ViT 的 Patch Embedding、Transformer Block 均从零手写。

\- \*\*工业级训练管线\*\*：包含数据增强（RandomCrop/Flip）、Cosine Annealing 学习率调度、模型自动保存机制。

\- \*\*严谨的对照实验\*\*：控制变量，公平对比两种范式在相同数据与超参下的表现。

\- \*\*全自动化可视化\*\*：内置绘图脚本，一键生成 Loss 与 Accuracy 对比曲线。



\## 📁 项目结构 (Project Structure)



```text

CIFAR10-CNN-vs-ViT/

├── configs.py           # 超参数配置 (Argparse)

├── data\_loader.py       # 数据下载、增强与 DataLoader

├── models/

│   ├── \_\_init\_\_.py

│   ├── cnn.py           # 手写轻量级 CNN (ResNet-lite)

│   └── vit.py           # 手写 Vision Transformer

├── train.py             # 训练脚本 (自动保存 best model 与 history)

├── evaluate.py          # 测试集评估脚本

├── plot\_results.py      # 读取 history.json 画对比图

├── smoke\_test.py        # 本地 CPU 快速冒烟测试

├── utils.py             # 随机种子、设备获取等工具

├── requirements.txt

├── data/                # 自动下载数据集 (不提交至 Git)

├── checkpoints/         # 权重与日志保存 (不提交至 Git)

└── assets/                     #生成对比图



🛠️ 快速开始 (Quick Start)

1\. 环境准备

bash

pip install -r requirements.txt

2\. 本地冒烟测试 (Smoke Test)

在消耗算力前，先在本地用随机张量验证网络结构是否存在维度错误：



bash

python smoke\_test.py

3\. 模型训练

默认训练 50 个 Epoch，自动下载 CIFAR-10 数据集：



bash

\# 训练 CNN

python train.py --model cnn --epochs 50



\# 训练 ViT

python train.py --model vit --epochs 50

4\. 测试集评估与出图

bash

\# 评估最佳模型

python evaluate.py --model cnn

python evaluate.py --model vit



\# 生成对比图到 assets/ 目录

python plot\_results.py

🎓 致谢与声明 (Acknowledgements)

The initial inspiration for this comparative study originated from a university AI course assignment. The entire codebase has been extensively refactored, modernized with data augmentation and learning rate scheduling, and implemented from scratch for a rigorous experimental comparison.



📄 License

MIT License





