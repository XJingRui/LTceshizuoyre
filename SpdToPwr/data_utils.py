import os
import glob
import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

# 核心配置字典，与你的特征工程无缝挂钩
MODULE_FEATURES = {
    "power_prediction": {
        "features": ['加速踏板开度', '制动踏板状态（开度）', '车速', '驾驶模式', '能量回收模式', '空调EDC实际功率',
                     '座舱PTC实际功率', '环境温度', '乘员舱实际温度'],
        "targets": ['总驱动功率']
    },
    "temp_prediction": {
        "features": ['电池包总电流', '前电机电流', '后电机电流', '车速', '环境温度', '电池PTC实际功率',
                     '空调EDC实际功率'],
        "targets": ['最高电芯温度', '最低电芯温度', '前电机温度', '后电机温度']
    }
}


def _segment_trips(df):
    """基于完整序列切分行程（保持原有逻辑）"""
    trip_ids = np.zeros(len(df), dtype=int)
    if 'Ready' not in df.columns: return trip_ids
    ready = df['Ready'].values
    trip_id, in_trip, last_time = 0, False, None
    for i in range(len(df)):
        if ready[i] == 1:
            current_time = df['时间轴'].iloc[i]
            gap = 999 if last_time is None else (current_time - last_time).total_seconds()
            if not in_trip or gap > 300: trip_id += 1
            in_trip, last_time = True, current_time
        else:
            in_trip = False
        trip_ids[i] = trip_id if in_trip else 0
    return trip_ids


def construct_sliding_windows(df, seq_len, feature_cols, target_cols):
    """对单块分片构建滑动窗口"""
    df_feat = df[feature_cols].apply(pd.to_numeric, errors='coerce')
    df_target = df[target_cols].apply(pd.to_numeric, errors='coerce')

    categorical_in_feat = [c for c in ['驾驶模式', '能量回收模式'] if c in df_feat.columns]
    for col in categorical_in_feat:
        df_feat[col] = df_feat[col].astype('category').cat.codes

    df_feat = df_feat.ffill().bfill()
    df_target = df_target.ffill().bfill()

    X_list, y_list = [], []
    values_x = df_feat.values.astype(np.float32)
    values_y = df_target.values.astype(np.float32)

    for i in range(len(values_x) - seq_len):
        X_list.append(values_x[i:i + seq_len])
        y_list.append(values_y[i + seq_len])

    if len(X_list) == 0:
        return np.empty((0, seq_len, len(feature_cols)), dtype=np.float32), np.empty((0, len(target_cols)),
                                                                                     dtype=np.float32)
    return np.stack(X_list, axis=0), np.stack(y_list, axis=0)


def split_by_trip(X, y, trip_ids, seq_len, train_ratio=0.70, val_ratio=0.15):
    """按行程无交叠切分训练、验证、测试集"""
    sample_trip_ids = trip_ids[seq_len:seq_len + len(X)]
    unique_trips = np.unique(sample_trip_ids)
    valid_trips = unique_trips[unique_trips > 0]

    if len(valid_trips) < 3:
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        return X[:train_end], X[train_end:val_end], X[val_end:], y[:train_end], y[train_end:val_end], y[val_end:]

    n_trips = len(valid_trips)
    train_trip_end = int(n_trips * train_ratio)
    val_trip_end = int(n_trips * (train_ratio + val_ratio))

    train_mask = np.isin(sample_trip_ids, valid_trips[:train_trip_end])
    val_mask = np.isin(sample_trip_ids, valid_trips[train_trip_end:val_trip_end])
    test_mask = np.isin(sample_trip_ids, valid_trips[val_trip_end:])

    print(f"✅ 行程切分完成 -> 训练集: {np.sum(train_mask)} | 验证集: {np.sum(val_mask)} | 测试集: {np.sum(test_mask)}")
    return X[train_mask], X[val_mask], X[test_mask], y[train_mask], y[val_mask], y[test_mask]


def normalize_datasets(train_X, val_X, test_X, train_y, val_y, test_y, save_dir="models"):
    _, seq_len, n_feat = train_X.shape
    scaler_x = StandardScaler()
    scaler_x.fit(train_X.reshape(-1, n_feat))

    train_X_norm = np.array([scaler_x.transform(train_X[i]) for i in range(len(train_X))], dtype=np.float32)
    val_X_norm = np.array([scaler_x.transform(val_X[i]) for i in range(len(val_X))], dtype=np.float32)
    test_X_norm = np.array([scaler_x.transform(test_X[i]) for i in range(len(test_X))], dtype=np.float32)

    scaler_y = StandardScaler()
    train_y_norm = scaler_y.fit_transform(train_y).astype(np.float32)
    val_y_norm = scaler_y.transform(val_y).astype(np.float32)
    test_y_norm = scaler_y.transform(test_y).astype(np.float32)

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "scaler_x.pkl"), 'wb') as f: pickle.dump(scaler_x, f)
    with open(os.path.join(save_dir, "scaler_y.pkl"), 'wb') as f: pickle.dump(scaler_y, f)
    return train_X_norm, val_X_norm, test_X_norm, train_y_norm, val_y_norm, test_y_norm, scaler_y


def create_torch_dataloaders(train_X, val_X, test_X, train_y, val_y, test_y, batch_size=64):
    train_ds = TensorDataset(torch.FloatTensor(train_X), torch.FloatTensor(train_y))
    val_ds = TensorDataset(torch.FloatTensor(val_X), torch.FloatTensor(val_y))
    test_ds = TensorDataset(torch.FloatTensor(test_X), torch.FloatTensor(test_y))
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False))


# ========================================================
# 核心业务入口：在训练模型时替代原本的 create_training_dataset
# ========================================================
def create_training_dataset_from_shards(data_dir="processedData", vin_id="6993", module="power_prediction", seq_len=20,
                                        batch_size=64):
    """
    第二部分代码：
    按升序顺序自动加载特定车辆的所有带重叠的小 CSV，重组完整的训练特征矩阵与行程链条。
    """
    print(f"\n🚀 启动分片流组装管道 | 车辆: {vin_id} | 预测业务: {module}")

    # 1. 严格按数字顺序检索切片文件
    search_pattern = os.path.join(data_dir, f"vin_{vin_id}_chunk_*.csv")
    shard_files = sorted(glob.glob(search_pattern), key=lambda x: int(x.split('_chunk_')[-1].split('.')[0]))

    if not shard_files:
        raise FileNotFoundError(f"❌ 未在 {data_dir} 中检索到车辆 {vin_id} 的任何切片文件！请先运行数据切分工具。")

    X_all, y_all = [], []
    df_global_list = []

    # 配置解析
    feat_cfg = MODULE_FEATURES[module]["features"]
    target_cfg = MODULE_FEATURES[module]["targets"]

    # 2. 顺序遍历加载每一个小分片
    for idx, file_path in enumerate(shard_files):
        df_chunk = pd.read_csv(file_path, engine='python')

        # 特殊处理：为能耗预测计算总驱动功率
        if module == "power_prediction":
            for col in ['前电机机械功率', '后电机功率']:
                if col not in df_chunk.columns: df_chunk[col] = 0.0
            df_chunk['前电机机械功率'] = pd.to_numeric(df_chunk['前电机机械功率'], errors='coerce').fillna(0)
            df_chunk['后电机功率'] = pd.to_numeric(df_chunk['后电机功率'], errors='coerce').fillna(0)
            df_chunk['总驱动功率'] = df_chunk['前电机机械功率'] + df_chunk['后电机功率']

        # A. 生成当前分片的滑动窗口 (利用了重叠区，交界处完全连续不丢样本)
        X_chunk, y_chunk = construct_sliding_windows(df_chunk, seq_len, feat_cfg, target_cfg)
        if X_chunk.size > 0:
            X_all.append(X_chunk)
            y_all.append(y_chunk)

        # B. 还原全局行程链条的核心算法：
        # 从第二个分片开始，必须剔除掉前 seq_len 行（因为它是重叠的旧数据），以此重组纯净无重的状态行
        if idx == 0:
            df_global_list.append(df_chunk)
        else:
            df_global_list.append(df_chunk.iloc[seq_len:])

    # 3. 内存聚合（Numpy 和小 DataFrame 合并，极速且 100% 不会触发 0xC0000005）
    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    df_global = pd.concat(df_global_list, ignore_index=True)

    if '时间轴' in df_global.columns:
        df_global['时间轴'] = pd.to_datetime(df_global['时间轴'])

    print(f"📊 碎片完美拼接完成！全局特征矩阵 X 形状: {X.shape} | 标签 y 形状: {y.shape}")

    # 4. 运行全局行程切分逻辑并进行数据清洗与 DataLoader 构建
    trip_ids = _segment_trips(df_global)
    train_X, val_X, test_X, train_y, val_y, test_y = split_by_trip(X, y, trip_ids, seq_len)

    (train_X, val_X, test_X, train_y, val_y, test_y,
     scaler_y) = normalize_datasets(train_X, val_X, test_X, train_y, val_y, test_y)

    train_loader, val_loader, test_loader = create_torch_dataloaders(train_X, val_X, test_X, train_y, val_y, test_y,
                                                                     batch_size)
    print("🎉 PyTorch 训练数据管道全线打通！\n")

    return train_loader, val_loader, test_loader, scaler_y


if __name__ == "__main__":
    # 本地直接运行测试数据组装管道
    tr_load, val_load, ts_load, sc_y = create_training_dataset_from_shards(
        data_dir="processedData",
        vin_id="6993",
        module="power_prediction",
        seq_len=20,
        batch_size=64
    )