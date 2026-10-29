import torch
import numpy as np
import random
import os

def set_seed(seed):
    """固定所有随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    """获取当前可用设备"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(state, save_dir, filename='best_model.pth'):
    """保存模型权重"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(state, os.path.join(save_dir, filename))