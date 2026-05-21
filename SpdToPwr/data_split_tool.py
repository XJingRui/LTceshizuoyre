import os
import csv
import pandas as pd
import numpy as np

# 离散状态量（用于清洗函数中）
CATEGORICAL_COLS = [
    "档位", "Ready", "驾驶模式", "混动模式", "能量回收模式",
    "智能驾驶是否开启", "前空调开关", "后空调开关", "充电枪状态", "充电状态",
    "制动踏板状态（开度）"
]


def clean_and_impute_single_vin(df_single_car):
    """单车独立清洗与时序插值（已修复：强制转数字，不再报错）"""
    df_clean = df_single_car.copy()
    df_clean['时间轴'] = pd.to_datetime(df_clean['时间轴'], errors='coerce')
    df_clean = df_clean.dropna(subset=['时间轴'])
    df_clean = df_clean.sort_values(by='时间轴').reset_index(drop=True)

    # ----------------------
    # 🔴 关键修复：强制把车速、SOC、加速踏板开度 转成数字（文本自动变NaN）
    # ----------------------
    if '车速' in df_clean.columns:
        df_clean['车速'] = pd.to_numeric(df_clean['车速'], errors='coerce')
        df_clean.loc[(df_clean['车速'] < 0) | (df_clean['车速'] > 300), '车速'] = np.nan

    if 'SOC' in df_clean.columns:
        df_clean['SOC'] = pd.to_numeric(df_clean['SOC'], errors='coerce')
        df_clean.loc[(df_clean['SOC'] < 0) | (df_clean['SOC'] > 100), 'SOC'] = np.nan

    if '加速踏板开度' in df_clean.columns:
        df_clean['加速踏板开度'] = pd.to_numeric(df_clean['加速踏板开度'], errors='coerce')
        df_clean.loc[(df_clean['加速踏板开度'] < 0) | (df_clean['加速踏板开度'] > 100), '加速踏板开度'] = np.nan

    # 离散字段前向填充
    cat_cols_in_df = [c for c in CATEGORICAL_COLS if c in df_clean.columns]
    df_clean[cat_cols_in_df] = df_clean[cat_cols_in_df].ffill()

    # 连续数值列插值
    continuous_cols = [c for c in df_clean.columns if c not in cat_cols_in_df and c not in ['vin', '时间轴']]
    df_clean[continuous_cols] = df_clean[continuous_cols].apply(pd.to_numeric, errors='coerce')
    df_clean[continuous_cols] = df_clean[continuous_cols].interpolate(method='linear').bfill()

    return df_clean


def split_and_save_by_overlap(csv_path, vin_id, seq_len=20, chunk_size=10000, save_dir="processedData"):
    print(f"🔄 启动原生安全引擎，正在扫描 CSV 提取车辆 {vin_id}...")
    os.makedirs(save_dir, exist_ok=True)

    # 自动编码检测
    encoding = 'utf-8'
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            f.readline()
    except UnicodeDecodeError:
        encoding = 'gbk'

    matched_rows = []
    header = []
    target_vin = str(vin_id).strip().lower()

    with open(csv_path, mode='r', encoding=encoding, errors='ignore') as f:
        reader = csv.reader(f)
        header = next(reader)

        # 智能匹配 vin 列
        vin_col_name = None
        for col in header:
            c = col.strip().lower()
            if 'vin' in c or '车架号' in c:
                vin_col_name = col
                break

        if vin_col_name is None:
            print("⚠️ 未找到 VIN 列，直接处理全文件")
            vin_idx = -1
        else:
            vin_idx = header.index(vin_col_name)
            print(f"✅ 找到 VIN 列：{vin_col_name}")

        # 逐行读取（不占内存）
        for row in reader:
            if not row:
                continue
            if vin_idx != -1:
                if len(row) <= vin_idx:
                    continue
                curr_vin = str(row[vin_idx]).strip().lower()
                if curr_vin != target_vin:
                    continue
            matched_rows.append(row)

    if not matched_rows:
        raise ValueError(f"❌ 未找到 VIN={vin_id} 的数据")

    # 清洗 + 切片
    df_single = pd.DataFrame(matched_rows, columns=header)
    df_cleaned = clean_and_impute_single_vin(df_single)
    total_rows = len(df_cleaned)
    print(f"✅ 清洗完成，有效行数：{total_rows}")

    # 切片保存
    start_idx = 0
    chunk_idx = 0
    overlap = seq_len

    while start_idx < total_rows:
        end_idx = start_idx + chunk_size
        df_chunk = df_cleaned.iloc[start_idx:end_idx].copy()
        filename = f"vin_{vin_id}_chunk_{chunk_idx}.csv"
        df_chunk.to_csv(os.path.join(save_dir, filename), index=False, encoding='utf-8')
        print(f"📂 生成切片 {chunk_idx:03d}：{start_idx} -> {min(end_idx, total_rows)} 行")

        if end_idx >= total_rows:
            break
        start_idx = end_idx - overlap
        chunk_idx += 1

    print(f"🎉 全部完成！共生成 {chunk_idx + 1} 个文件")


if __name__ == "__main__":
    split_and_save_by_overlap(
        csv_path="作业数据部分 - 副本.csv",
        vin_id="6993",
        seq_len=20,
        chunk_size=10000,
        save_dir="processedData"
    )