import json
import matplotlib.pyplot as plt
import os

def plot_comparison():
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    for model_name, color in [("cnn", "#1f77b4"), ("vit", "#d62728")]:
        path = f"./checkpoints/{model_name}_history.json"
        if not os.path.exists(path):
            print(f"⚠️ 找不到 {path}，跳过该模型")
            continue
            
        with open(path, 'r') as f:
            hist = json.load(f)
        
        epochs = range(1, len(hist['val_acc']) + 1)
        
        # 画 Loss
        ax1.plot(epochs, hist['val_loss'], label=f'{model_name.upper()} Val Loss', color=color, linewidth=2)
        # 画 Acc
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
    os.makedirs("assets", exist_ok=True)
    plt.savefig("assets/cnn_vs_vit_comparison.png", dpi=300)
    print("✅ 对比图已生成: assets/cnn_vs_vit_comparison.png")

if __name__ == "__main__":
    plot_comparison()