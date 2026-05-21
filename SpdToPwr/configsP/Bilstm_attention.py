# configsP/Bilstm_attention.py
import torch

class BaseConfig:
    # --- 1. 硬件与底层路径配置 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parquet_path = "data/H56D_6993_processed.parquet"  # 单车数据文件
    module = "power_prediction"  # "power_prediction" or "temp_prediction"
    model_save_path = 'models/best_model_improved.pth'
    history_save_path = 'models/training_history_improved.json'

    # --- 2. 数据管道超参数 ---
    seq_len = 20          # 滑动窗口的时间步长
    batch_size = 64       # 批次大小

    # --- 3. 模型网络结构参数 ---
    input_dim = 9         # 输入特征维度（power: 9维, temp: 7维）
    hidden_dim = 128      # BiLSTM 隐藏层大小
    output_dim = 1        # 输出维度（power: 1, temp: 4）

    # --- 4. 优化器与训练主循环参数 (AdamW) ---
    max_epochs = 150      # 最大训练轮数
    learning_rate = 1e-3  # 初始学习率
    weight_decay = 1e-4   # 权重衰减（L2正则化）
    max_norm = 1.0        # 梯度裁剪的最大范数

    # --- 5. 学习率调度器参数 (ReduceLROnPlateau) ---
    lr_factor = 0.5       # 每次触发衰减时：lr = lr * factor
    lr_patience = 8       # 连续多少个 epoch 验证集 loss 不降则触发衰减
    min_lr = 1e-6         # 允许的最小学习率
    lr_cooldown = 3       # 降完学习率后的冷却轮数

    # --- 6. 早停策略参数 (EarlyStopping) ---
    es_patience = 20      # 连续多少个 epoch 无改善则提早结束训练
    es_min_delta = 1e-4   # 小于该降幅视为未改善