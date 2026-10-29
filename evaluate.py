import torch
from configs import get_args
from utils import get_device
from data_loader import get_cifar10_loaders
from models.cnn import CNN
from models.vit import ViT

def evaluate():
    args = get_args()
    device = get_device()
    
    _, _, test_loader = get_cifar10_loaders(batch_size=args.batch_size, data_dir=args.data_dir)
    
    if args.model == "cnn":
        model = CNN().to(device)
    else:
        model = ViT().to(device)
        
    model.load_state_dict(torch.load(f"{args.save_dir}/{args.model}_best.pth"))
    model.eval()
    
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    print(f"🏆 Test Accuracy of {args.model.upper()}: {100. * correct / total:.2f}%")

if __name__ == "__main__":
    evaluate()