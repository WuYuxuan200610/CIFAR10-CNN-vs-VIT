import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.vit import ViT
from models.cnn import CNN  # 假设你的 CNN 写在 models/cnn.py 里

def run_smoke_test(model_name, epochs=1):
    print(f"\n{'='*40}")
    print(f"🚀 开始冒烟测试: {model_name}")
    
    # 1. 伪造数据：2 个 batch，每批 4 张图 (32x32，RGB)
    # 模拟真实 CIFAR-10 的形状
    dummy_images = torch.randn(8, 3, 32, 32)
    dummy_labels = torch.randint(0, 10, (8,))
    dataset = TensorDataset(dummy_images, dummy_labels)
    loader = DataLoader(dataset, batch_size=4)

    # 2. 初始化模型
    if model_name == "vit":
        model = ViT(img_size=32, patch_size=4, embed_dim=192, depth=6, num_heads=6)
    else:
        model = CNN(num_classes=10)

    # 3. 准备训练组件
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 4. 跑一步前向和反向传播
    model.train()
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        
        # 前向
        outputs = model(images)
        
        # 检查维度是否正确
        assert outputs.shape == (4, 10), f"❌ 输出维度错误: {outputs.shape}"
        
        # 反向
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"✅ Batch {batch_idx+1} 跑通 | Loss: {loss.item():.4f}")

    print(f"🎉 {model_name} 冒烟测试通过")

if __name__ == "__main__":
    # 先测试 ViT，如果有 bug 立刻暴露
    run_smoke_test("vit")
    # 再测试 CNN
    run_smoke_test("cnn")