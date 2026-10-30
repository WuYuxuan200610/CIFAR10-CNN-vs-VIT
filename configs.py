import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser(description="CNN vs ViT on CIFAR-10")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "vit"], help="选择模型")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--data_dir", type=str, default="./data", help="数据存放目录")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="模型保存目录")
    parser.add_argument("--num_workers", type=int, default=4 if torch.cuda.is_available() else 0, help="DataLoader 线程数")
    
    args = parser.parse_args()
    return args