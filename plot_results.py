import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import os
import matplotlib.font_manager as fm

from models.cnn import CNN
from models.vit import ViT

# ==========================================
# 基础配置
# ==========================================
font_path = 'C:/Windows/Fonts/msyh.ttc' 
if os.path.exists(font_path):
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
plt.rcParams['axes.unicode_minus'] = False
os.makedirs("assets", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
classes_cn = ('飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车')

# ==========================================
# 1. 画 Loss 与 Accuracy 对比曲线
# ==========================================
def plot_curves():
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    for model_name, color in [("cnn", "#1f77b4"), ("vit", "#d62728")]:
        path = f"./checkpoints/{model_name}_history.json"
        if not os.path.exists(path):
            print(f"⚠️ 找不到 {path}，跳过该曲线")
            continue
            
        with open(path, 'r') as f:
            hist = json.load(f)
        
        epochs = range(1, len(hist['val_acc']) + 1)
        ax1.plot(epochs, hist['val_loss'], label=f'{model_name.upper()} Val Loss', color=color, linewidth=2)
        ax2.plot(epochs, hist['val_acc'], label=f'{model_name.upper()} Val Acc', color=color, linewidth=2)

    ax1.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()

    plt.tight_layout()
    plt.savefig("assets/cnn_vs_vit_comparison.png", dpi=300)
    plt.close()
    print("✅ 对比曲线已生成: assets/cnn_vs_vit_comparison.png")

# ==========================================
# 2. 画混淆矩阵
# ==========================================
def plot_confusion_matrix(model, model_name, loader):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes_cn, yticklabels=classes_cn)
    
    if os.path.exists(font_path):
        for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
            label.set_fontproperties(font_prop)

    plt.title(f'{model_name.upper()} Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'assets/{model_name}_confusion_matrix.png', dpi=300)
    plt.close()
    print(f"✅ {model_name.upper()} 混淆矩阵已生成")

# ==========================================
# 3. 画预测示例图
# ==========================================
def plot_predictions(model, model_name, loader):
    dataiter = iter(loader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)
    
    outputs = model(images)
    probs = F.softmax(outputs, dim=1)
    _, predicted = outputs.max(1)
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for idx, ax in enumerate(axes.flatten()):
        img = images[idx].cpu() / 2 + 0.5 # 反归一化
        npimg = img.numpy()
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
        
        true_label = classes_cn[labels[idx]]
        pred_label = classes_cn[predicted[idx]]
        conf = probs[idx][predicted[idx]].item() * 100
        
        color = 'green' if labels[idx] == predicted[idx] else 'red'
        ax.set_title(f"True: {true_label}\nPred: {pred_label} ({conf:.1f}%)", color=color, fontsize=10, fontproperties=font_prop)
        ax.axis('off')
    
    plt.suptitle(f'{model_name.upper()} Predictions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'assets/{model_name}_predictions.png', dpi=300)
    plt.close()
    print(f"✅ {model_name.upper()} 预测示例已生成")


if __name__ == "__main__":
    # 1. 画曲线（只要有 json 就能画）
    plot_curves()

    # 2. 准备数据加载器（用于画混淆矩阵和预测图）
    print("正在加载测试集...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    # 3. 加载模型并画混淆矩阵与预测图
    if os.path.exists("./checkpoints/cnn_best.pth"):
        cnn = CNN().to(device)
        cnn.load_state_dict(torch.load("./checkpoints/cnn_best.pth", map_location=device))
        cnn.eval()
        plot_confusion_matrix(cnn, "cnn", test_loader)
        plot_predictions(cnn, "cnn", test_loader)

    if os.path.exists("./checkpoints/vit_best.pth"):
        vit = ViT().to(device)
        vit.load_state_dict(torch.load("./checkpoints/vit_best.pth", map_location=device))
        vit.eval()
        plot_confusion_matrix(vit, "vit", test_loader)
        plot_predictions(vit, "vit", test_loader)

    
    print("🎉 所有图表生成完毕， 于assets/ 文件夹查看。")