# tool.py 提供给train和play的工具函数模块

import os
import re
import shutil
from datetime import datetime

def find_path(time="latest", epoch="latest", type = "actor"):
    """查找模型路径"""
    # 获取时间目录
    time_path = _get_time_path("model", time)

    if type == "actor":
        # 获取actor模型路径
        actor_path = _get_model_path(time_path, "actor_pth", "actor", epoch)
        print(f"📁 找到actor模型路径: {actor_path}")
        return actor_path
    elif type == "critic":
        # 获取critic模型路径
        critic_path = _get_model_path(time_path, "critic_pth", "critic", epoch)
        print(f"📁 找到critic模型路径: {critic_path}")
        return critic_path
    else:
        raise ValueError(f"类型错误: {type}")

def _get_time_path(base_path, time):
    """获取时间目录路径"""
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"基础路径不存在: {base_path}")
    
    if time == "latest":
        time_dirs = [d for d in os.listdir(base_path) 
                    if os.path.isdir(os.path.join(base_path, d)) and 
                    re.match(r"Train_data_\d{2}-\d{2}-\d{2}-\d{2}", d)]
        
        if not time_dirs:
            raise FileNotFoundError(f"在 {base_path} 中未找到时间目录")
            
        # 按创建时间排序获取最新的
        time_dirs.sort(key=lambda x: os.path.getctime(os.path.join(base_path, x)), reverse=True)
        return os.path.join(base_path, time_dirs[0])
    
    else:
        time_dir = f"Train_data_{time}"
        path = os.path.join(base_path, time_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(f"时间目录不存在: {path}")
        return path

def _get_model_path(time_path, model_dir, prefix, epoch):
    """获取具体模型路径"""
    model_dir_path = os.path.join(time_path, model_dir)
    
    if not os.path.exists(model_dir_path):
        raise FileNotFoundError(f"模型目录不存在: {model_dir_path}")
    
    if epoch == "latest":
        # 查找所有模型文件
        model_files = [f for f in os.listdir(model_dir_path) 
                      if f.startswith(f"{prefix}_epoch_") and f.endswith(".pth")]
        if not model_files:
            raise FileNotFoundError(f"在 {model_dir_path} 中未找到模型文件")
            
        # 解析epoch号并获取最新的
        epoch_nums = []
        for f in model_files:
            try:
                num = int(f.split("_")[-1].split(".")[0])
                epoch_nums.append(num)
            except ValueError:
                continue
        
        if not epoch_nums:
            raise ValueError(f"无法解析 {model_dir_path} 中的模型文件")
            
        max_epoch = max(epoch_nums)
        return os.path.join(model_dir_path, f"{prefix}_epoch_{max_epoch}.pth")
    
    elif epoch == "best":
        # 查找所有模型文件
        for f in os.listdir(model_dir_path):
            if f.startswith(f"best_{prefix}") and f.endswith(".pth"):
                return os.path.join(model_dir_path, f)
        raise FileNotFoundError(f"在 {model_dir_path} 中未找到模型文件")
    
    else:
        # 使用指定epoch号
        try:
            epoch = int(epoch)
        except ValueError:
            raise TypeError(f"epoch参数应为整数、'best'或'latest', 而不是 {repr(epoch)}")
            
        path = os.path.join(model_dir_path, f"{prefix}_epoch_{epoch}.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {path}")
        return path

def get_savepath():
    """获取数据保存路径, return: (actor_path, critic_path)"""
    current_time = datetime.now().strftime("%m-%d-%H-%M")
    time_path = f"model/Train_data_{current_time}"
    if not os.path.exists(time_path):
        os.makedirs(os.path.join(time_path, "actor_pth"))
        os.makedirs(os.path.join(time_path, "critic_pth"))
        os.makedirs(os.path.join(time_path, "data"))
        os.makedirs(os.path.join(time_path, "picture"))
    return time_path