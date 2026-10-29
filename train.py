import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json
import os

from configs import get_args
from utils import set_seed, get_device, save_checkpoint
from data_loader import get_cifar10_loaders
from models.cnn import CNN
from models.vit import ViT

def train():
    args = get_args()
    set_seed(args.seed)
    device = get_device()
    print(f"🚀 Training {args.model.upper()} on {device}...")

    # 数据
    train_loader, val_loader, _ = get_cifar10_loaders(
        batch_size=args.batch_size, data_dir=args.data_dir, num_workers=args.num_workers
    )

    # 模型
    if args.model == "cnn":
        model = CNN(num_classes=10).to(device)
    else:
        model = ViT(img_size=32, patch_size=4, embed_dim=192, depth=6, num_heads=6).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(args.epochs):
        # --- 训练 ---
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        scheduler.step()
        train_loss = running_loss / len(train_loader)
        
        # --- 验证 ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        print(f"✅ Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # 记录数据
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model.state_dict(), args.save_dir, f"{args.model}_best.pth")
            print(f"🌟 发现新高！模型已保存 (Acc: {best_acc:.2f}%)")

    # 保存训练历史
    history_path = os.path.join(args.save_dir, f"{args.model}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f)
    print(f"🎉 训练完成！历史记录已保存到 {history_path}")

if __name__ == "__main__":
    train()