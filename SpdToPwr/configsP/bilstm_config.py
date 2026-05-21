# configsP/bilstm_config.py
import torch


class BiLSTMConfig:
    # --- 1. 硬件与底层路径配置 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 2. 数据管道超参数 ---
    seq_len = 20
    batch_size = 64

    # --- 3. 模型网络结构参数 ---
    input_dim = 9          # 输入特征维度（power: 9维, temp: 7维，训练时自动覆盖）
    hidden_dim = 128       # BiLSTM 隐藏层大小（单向）
    output_dim = 1         # 输出维度（power: 1, temp: 4）
    num_layers = 2         # BiLSTM 层数

    # --- 4. 优化器与训练主循环参数 (AdamW) ---
    max_epochs = 150
    learning_rate = 1e-3
    weight_decay = 1e-4
    max_norm = 1.0          # 梯度裁剪最大范数

    # --- 5. 学习率调度器参数 (ReduceLROnPlateau) ---
    lr_factor = 0.5
    lr_patience = 8
    min_lr = 1e-6
    lr_cooldown = 3

    # --- 6. 早停策略参数 (EarlyStopping) ---
    es_patience = 20
    es_min_delta = 1e-4
