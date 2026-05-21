import os

# 保持防冲突环境变量（双保险）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json

# ===================== ⚙️ 核心配置与数据管道导入 =====================
from configs.Bilstm_attention import BaseConfig
# 修改点 1：导入全新数据工具函数的同时，引入核心配置字典 MODULE_FEATURES
from data_utils import create_training_dataset_from_shards, MODULE_FEATURES
from SpdToPwr.model_improved import EVPowerImprovedModel
from train_utils import EarlyStopping


def run_training():
    config = BaseConfig()

    # ⚙️ 任务与车辆动态配置（可自由切换）
    target_vin = "6993"  # 指定本次要训练的车辆 VIN 号
    shards_dir = "../processedData"  # 分片小文件存放的目录
    business_module = "power_prediction"  # 预测任务类型："power_prediction" 或 "temp_prediction"

    # 修改点 2：基于 datautils 自动计算特征和标签维度，实现无缝挂钩
    input_dim = len(MODULE_FEATURES[business_module]["features"])
    output_dim = len(MODULE_FEATURES[business_module]["targets"])

    # 修改点 3：动态生成保存路径，防止多任务、多车辆训练时文件被互相覆盖
    model_dir = "../models"
    history_dir = "../history"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(history_dir, exist_ok=True)

    model_save_path = os.path.join(model_dir, f"best_model_{business_module}_{target_vin}.pth")
    history_save_path = os.path.join(history_dir, f"history_{business_module}_{target_vin}.json")

    print(f"\n🔥 岚图三电动力预测引擎启动 | 训练设备: {config.device} | 批次大小: {config.batch_size}")
    print(f"   当前业务模块: {business_module} (输入特征: {input_dim} 维 | 输出目标: {output_dim} 维)")
    print(f"   当前目标车辆 VIN: {target_vin} | 正在从分片目录 [{shards_dir}] 加载时序流...")
    print("-" * 70)

    # 调用分片组装函数，解包 4 个返回值
    train_loader, val_loader, test_loader, scaler_y = create_training_dataset_from_shards(
        data_dir=shards_dir,
        vin_id=target_vin,
        module=business_module,
        seq_len=config.seq_len,
        batch_size=config.batch_size
    )

    # 修改点 4：将动态获取的 input_dim 传入模型
    # ⚠️ 踩坑提示：如果你的 EVPowerImprovedModel 支持自定义输出维度，请务必把 output_dim=output_dim 也传进去
    model = EVPowerImprovedModel(
        input_dim=input_dim,
        hidden_dim=config.hidden_dim,
        seq_len=config.seq_len
        # output_dim=output_dim  # 如果模型构造函数支持输出维度，请取消本行注释
    ).to(config.device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    scheduler = ReduceLROnPlateau(
        optimizer, mode='min',
        factor=config.lr_factor,
        patience=config.lr_patience,
        min_lr=config.min_lr,
        cooldown=config.lr_cooldown
    )

    # 修改点 5：使用动态的 model_save_path 初始化早停机制
    early_stopping = EarlyStopping(
        patience=config.es_patience,
        min_delta=config.es_min_delta,
        verbose=True,
        save_path=model_save_path
    )

    history = {'train_loss': [], 'val_loss': [], 'lr_history': []}

    # ===================== 🔄 训练与验证循环 =====================
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_loss_accum = 0.0

        for x_batch, y_batch in train_loader:
            x_input = x_batch.to(config.device)
            y_target = y_batch.to(config.device)

            optimizer.zero_grad()
            outputs = model(x_input)
            loss = criterion(outputs, y_target)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)

            optimizer.step()
            train_loss_accum += loss.item()

        model.eval()
        val_loss_accum = 0.0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_input = x_batch.to(config.device)
                y_target = y_batch.to(config.device)

                outputs = model(x_input)
                loss = criterion(outputs, y_target)
                val_loss_accum += loss.item()

        epoch_train_loss = train_loss_accum / len(train_loader)
        epoch_val_loss = val_loss_accum / len(val_loader)
        curr_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['lr_history'].append(curr_lr)

        print(
            f"Epoch [{epoch:03d}/{config.max_epochs}] | Train MSE: {epoch_train_loss:.6f} | Val MSE: {epoch_val_loss:.6f} | 当前 LR: {curr_lr:.2e}")

        scheduler.step(epoch_val_loss)

        early_stopping(epoch_val_loss, model)
        if early_stopping.early_stop:
            print("🛑 [早停触发] 验证集 Loss 已连续无显著改善，提前终止训练。")
            break

    # ===================== 💾 历史记录与模型持化 =====================
    with open(history_save_path, 'w') as f:
        json.dump(history, f)

    # 修改点 6：加载最优权重时使用一致的动态路径
    model.load_state_dict(torch.load(model_save_path))
    print(f"\n🎉 训练圆满结束！最优模型权重已固化保存至: {model_save_path}")
    print(f"📊 训练历史日志已保存至: {history_save_path}")


if __name__ == "__main__":
    run_training()