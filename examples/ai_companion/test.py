import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

def analyze_layer_importance(model, dataloader, criterion, device='cuda'):
    """分析模型各层参数的重要性
    
    Args:
        model: 预训练模型
        dataloader: 用于评估的数据加载器
        criterion: 损失函数
        device: 计算设备
    """
    # 存储每层的重要性分数
    layer_importance = {}
    
    # 获取所有命名参数
    for name, param in model.named_parameters():
        if param.requires_grad:
            # 保存原始参数
            original_data = param.data.clone()
            
            # 计算参数扰动
            perturbation = torch.randn_like(param.data) * 0.01
            param.data += perturbation
            
            # 评估扰动后的性能
            perturbed_loss = evaluate_model(model, dataloader, criterion, device)
            
            # 恢复原始参数
            param.data = original_data
            
            # 计算重要性分数（损失变化量）
            layer_importance[name] = perturbed_loss
            
    return layer_importance

def evaluate_model(model, dataloader, criterion, device):
    """评估模型性能"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def analyze_task_specific_layers(model, task_data, base_data, criterion, device='cuda'):
    """分析哪些层更适合特定任务的调整
    
    Args:
        model: 预训练模型
        task_data: 特定任务的数据加载器
        base_data: 基础任务的数据加载器
        criterion: 损失函数
        device: 计算设备
    """
    task_importance = analyze_layer_importance(model, task_data, criterion, device)
    base_importance = analyze_layer_importance(model, base_data, criterion, device)
    
    layer_adaptability = {}
    for name in task_importance:
        # 计算任务特异性分数：特定任务重要性与基础任务重要性的比值
        layer_adaptability[name] = task_importance[name] / base_importance[name]
    
    return layer_adaptability

def visualize_results(importance_scores, title):
    """可视化分析结果"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    plt.bar(importance_scores.keys(), importance_scores.values())
    plt.xticks(rotation=45)
    plt.title(title)
    plt.tight_layout()
    plt.show()

# 使用示例
if __name__ == "__main__":
    # 假设已经有了模型和数据加载器
    # model = YourModel()
    # base_dataloader = BaseDataLoader()
    # task_dataloader = TaskDataLoader()
    # criterion = nn.CrossEntropyLoss()
    
    # 分析基础能力重要性
    # base_importance = analyze_layer_importance(model, base_dataloader, criterion)
    # visualize_results(base_importance, "Layer Importance for Base Capabilities")
    
    # 分析任务特异性
    # adaptability = analyze_task_specific_layers(model, task_dataloader, base_dataloader, criterion)
    # visualize_results(adaptability, "Layer Adaptability for Specific Task")
