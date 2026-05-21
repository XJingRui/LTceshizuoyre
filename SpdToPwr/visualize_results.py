import os
import glob
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# 配置 Matplotlib 中文字体与高清晰度
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']  # Windows用SimHei, Mac用Arial Unicode MS
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150  # 提高图表分辨率


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# ==========================================
# 维度 1：宏观画像层（全局 SOC 与状态瀑布图）
# ==========================================
def plot_global_soc(df, save_dir, vin_id):
    if 'SOC' not in df.columns or '时间轴' not in df.columns:
        print(f"⚠️ [{vin_id}] 缺少 SOC 或时间轴数据，跳过维度 1 绘图。")
        return

    print(f"📈 [{vin_id}] 正在绘制：维度 1 - 全局 SOC 与状态瀑布图...")
    plt.figure(figsize=(18, 5))

    plt.plot(df['时间轴'], df['SOC'], color='#2ca02c', linewidth=1.5, label='SOC (%)')

    if 'Ready' in df.columns:
        drive_mask = df['Ready'] == 1
        plt.fill_between(df['时间轴'], 0, 100, where=drive_mask, color='skyblue', alpha=0.3, label='行驶中 (Ready=1)')

    if '充电状态' in df.columns:
        charge_mask = df['充电状态'] == 5
        plt.fill_between(df['时间轴'], 0, 100, where=charge_mask, color='salmon', alpha=0.3, label='充电中')

    plt.title(f"维度1：全局车辆运行轨迹与 SOC 变化总览 (VIN: {vin_id})", fontsize=16)
    plt.xlabel("时间", fontsize=12)
    plt.ylabel("SOC (%)", fontsize=12)
    plt.ylim(0, 105)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{vin_id}_1_Global_SOC_Overview.png"))
    plt.close()


# ==========================================
# 维度 2：行程与驾驶行为层（单次行程动态图）
# ==========================================
def plot_trip_dynamics(df, save_dir, vin_id):
    if 'Ready' not in df.columns:
        return

    print(f"📈 [{vin_id}] 正在绘制：维度 2 - 单次行程动态切片与三电功率图...")
    df_drive = df[df['Ready'] == 1].reset_index(drop=True)

    if len(df_drive) < 100:
        print(f"⚠️ [{vin_id}] 行驶数据量过少，跳过行程动态图。")
        return

    # 截取中间最活跃的 1500 个数据点进行微观行为分析
    if len(df_drive) < 1500:
        slice_df = df_drive.copy()
    else:
        start_idx = len(df_drive) // 2
        slice_df = df_drive.iloc[start_idx: start_idx + 1500].copy()

    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # 子图1：车速与踏板意图
    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    ax1.plot(slice_df['时间轴'], slice_df['车速'], color='blue', linewidth=2, label='车速 (km/h)')

    if '加速踏板开度' in slice_df.columns:
        ax1_twin.fill_between(slice_df['时间轴'], 0, slice_df['加速踏板开度'], color='orange', alpha=0.3,
                             label='加速踏板 (%)')
    if '制动踏板状态（开度）' in slice_df.columns:
        ax1_twin.plot(slice_df['时间轴'], slice_df['制动踏板状态（开度）'] * 10, color='red', linestyle='--',
                     label='制动踏板 (标量)')

    ax1.set_ylabel("车速 (km/h)", color='blue', fontsize=12)
    ax1_twin.set_ylabel("踏板开度 / 状态", color='gray', fontsize=12)
    ax1.set_title(f"维度2.1：驾驶员意图与车速响应 (VIN: {vin_id})", fontsize=14)
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 子图2：三电功率溯源与计算
    ax2 = axes[1]

    # 双电机功率安全兼容逻辑
    for col in ['前电机机械功率', '后电机功率']:
        if col not in slice_df.columns:
            slice_df[col] = 0.0

    total_drive_power = slice_df['前电机机械功率'] + slice_df['后电机功率']
    ax2.plot(slice_df['时间轴'], total_drive_power, label='双电机总驱动机械功率 (kW)', color='purple', alpha=0.8)

    # 电池包总电功率在线计算
    if '电池包总电压' in slice_df.columns and '电池包总电流' in slice_df.columns:
        total_pack_power = (slice_df['电池包总电压'] * slice_df['电池包总电流']) / 1000
        ax2.plot(slice_df['时间轴'], total_pack_power, label='电池包总输出电功率 (kW)', color='black', linewidth=1.5,
                 linestyle='-.')

    if '空调EDC实际功率' in slice_df.columns:
        ax2.plot(slice_df['时间轴'], slice_df['空调EDC实际功率'], label='空调功率 (kW)', color='cyan', alpha=0.8)

    ax2.set_ylabel("功率 (kW)", fontsize=12)
    ax2.set_title(f"维度2.2：三电系统功率动态消耗 (VIN: {vin_id})", fontsize=14)
    ax2.set_xlabel("时间", fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{vin_id}_2_Trip_Dynamics_and_Power.png"))
    plt.close()


# ==========================================
# 维度 3：充电特征层（CC-CV 恒流恒压充电曲线）
# ==========================================
def plot_charging_curve(df, save_dir, vin_id):
    if '充电状态' not in df.columns or 'SOC' not in df.columns:
        print(f"⚠️ [{vin_id}] 缺少必要充电字段，跳过维度 3 分析。")
        return

    df_charge = df[df['充电状态'] == 5].reset_index(drop=True).copy()
    if df_charge.empty or len(df_charge) < 10:
        print(f"⚠️ [{vin_id}] 未找到足够充电状态数据，跳过充电分析。")
        return

    print(f"📈 [{vin_id}] 正在绘制：维度 3 - 快充特性曲线 (CC-CV)...")
    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    df_charge = df_charge.sort_values(by='SOC').drop_duplicates(subset=['SOC'])

    # 动态补偿充电功率字段
    if '充电功率' not in df_charge.columns and '充电电压' in df_charge.columns and '充电电流' in df_charge.columns:
        df_charge['充电功率'] = (df_charge['充电电压'] * df_charge['充电电流']) / 1000

    if '充电电流' in df_charge.columns:
        ax1.plot(df_charge['SOC'], df_charge['充电电流'], color='blue', linewidth=2, label='充电电流 (A)')
    if '充电功率' in df_charge.columns:
        ax1.plot(df_charge['SOC'], df_charge['充电功率'], color='green', linewidth=2, linestyle='--',
                 label='充电功率 (kW)')
    if '充电电压' in df_charge.columns:
        ax2.plot(df_charge['SOC'], df_charge['充电电压'], color='red', linewidth=2, label='充电电压 (V)')

    ax1.set_xlabel("SOC (%)", fontsize=12)
    ax1.set_ylabel("电流 (A) / 功率 (kW)", color='blue', fontsize=12)
    ax2.set_ylabel("电压 (V)", color='red', fontsize=12)
    plt.title(f"维度3：电池快充特性曲线 (VIN: {vin_id})", fontsize=16)

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{vin_id}_3_Charging_Curve.png"))
    plt.close()


# ==========================================
# 维度 4：热管理迟滞层（温度与功率延迟分析）
# ==========================================
def plot_temp_hysteresis(df, save_dir, vin_id):
    if 'Ready' not in df.columns or '最高电芯温度' not in df.columns or '电池包总电流' not in df.columns:
        return

    print(f"📈 [{vin_id}] 正在绘制：维度 4 - 热惯性与温度迟滞效应...")
    df_drive = df[df['Ready'] == 1].reset_index(drop=True)
    if len(df_drive) > 3000:
        slice_df = df_drive.iloc[1000:4000].copy()
    else:
        slice_df = df_drive.copy()

    if slice_df.empty:
        return

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()

    time_minutes = (slice_df['时间轴'] - slice_df['时间轴'].iloc[0]).dt.total_seconds() / 60
    current_heat = (slice_df['电池包总电流'].abs() / 100) ** 2

    ax1.fill_between(time_minutes, 0, current_heat, color='orange', alpha=0.4, label='电池包电流波动 (模拟发热量)')
    ax2.plot(time_minutes, slice_df['最高电芯温度'], color='red', linewidth=2.5, label='最高电芯温度 (℃)')

    ax1.set_xlabel("持续行驶时间 (分钟)", fontsize=12)
    ax1.set_ylabel("电流发热指标 (I^2)", color='orange', fontsize=12)
    ax2.set_ylabel("温度 (℃)", color='red', fontsize=12)
    plt.title(f"维度4：热系统的迟滞性分析 (VIN: {vin_id})", fontsize=14)

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{vin_id}_4_Temperature_Hysteresis.png"))
    plt.close()


# ==========================================
# 维度 5：电机能耗层（二维散点工况图）
# ==========================================
def plot_motor_efficiency(df, save_dir, vin_id):
    if 'Ready' not in df.columns or '车速' not in df.columns:
        return

    # 安全兼容前电机机械功率字段
    if '前电机机械功率' not in df.columns:
        df['前电机机械功率'] = 0.0

    print(f"📈 [{vin_id}] 正在绘制：维度 5 - 电机能耗分布散点图...")
    mask = (df['Ready'] == 1) & (df['车速'] > 10) & (df['前电机机械功率'] > 2)
    df_motor = df[mask].copy()

    if len(df_motor) < 50:
        print(f"⚠️ [{vin_id}] 能耗有效数据量不足，跳过电机图。")
        return

    # 在线计算百公里瞬时电耗作为色彩映射机制
    if '电池包总电压' in df_motor.columns and '电池包总电流' in df_motor.columns:
        df_motor['百公里瞬时电耗'] = (df_motor['电池包总电压'] * df_motor['电池包总电流'] / 1000) / df_motor[
            '车速'] * 100
        df_motor = df_motor[(df_motor['百公里瞬时电耗'] > 0) & (df_motor['百公里瞬时电耗'] < 50)]
    else:
        df_motor['百公里瞬时电耗'] = 15.0  # 缺省常数

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df_motor['车速'], df_motor['前电机机械功率'],
                         c=df_motor['百公里瞬时电耗'], cmap='viridis',
                         alpha=0.6, s=15)

    cbar = plt.colorbar(scatter)
    cbar.set_label('瞬时百公里电耗 (kWh/100km)', fontsize=12)

    plt.xlabel('车速 (km/h)', fontsize=12)
    plt.ylabel('前电机机械功率 (kW)', fontsize=12)
    plt.title(f'维度5：电机运行工况与能耗分布散点图 (VIN: {vin_id})', fontsize=15)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{vin_id}_5_Motor_Efficiency_Map.png"))
    plt.close()


# ==========================================
# 核心改造：单辆车的全分片无缝合并与清洗机制
# ==========================================
def run_visualizations_for_vin_chunks(data_dir, vin_id, save_dir="visualizations"):
    """顺序加载、组装、清洗特定车辆的所有分片 CSV 文件后执行可视化"""
    print(f"\n📂 开始检索并重组车辆 [{vin_id}] 的历史时序数据分片...")

    # 1. 严格按数字顺序检索切片文件
    search_pattern = os.path.join(data_dir, f"vin_{vin_id}_chunk_*.csv")
    shard_files = sorted(glob.glob(search_pattern), key=lambda x: int(x.split('_chunk_')[-1].split('.')[0]))

    if not shard_files:
        print(f"❌ 未在 {data_dir} 中找到车辆 {vin_id} 的任何切片文件！跳过该车。")
        return

    # 2. 顺序加载合并
    df_list = []
    for file_path in shard_files:
        df_chunk = pd.read_csv(file_path, engine='python')
        df_list.append(df_chunk)

    df_global = pd.concat(df_list, ignore_index=True)

    # 3. 数据类型鲁棒性清洗（关键防御步骤）
    if '时间轴' in df_global.columns:
        df_global['时间轴'] = pd.to_datetime(df_global['时间轴'], errors='coerce')
        # 核心：基于时间戳排序并剔除因滑动窗口机制产生的重叠重复行
        df_global = df_global.dropna(subset=['时间轴']).sort_values(by='时间轴').drop_duplicates(subset=['时间轴'])

    # 将常用列显式转换为数值型，防止由于特定 CSV 单元格污染导致画图异常
    numeric_cols = ['SOC', 'Ready', '车速', '电池包总电压', '电池包总电流', '最高电芯温度', '充电状态']
    for col in numeric_cols:
        if col in df_global.columns:
            df_global[col] = pd.to_numeric(df_global[col], errors='coerce').ffill().bfill()

    print(f"📊 时序流链条组装成功！无重复状态行数: {len(df_global)}")

    # 4. 建立专属图像输出目录并执行绘制
    car_save_dir = os.path.join(save_dir, str(vin_id))
    ensure_dir(car_save_dir)

    plot_global_soc(df_global, car_save_dir, vin_id)
    plot_trip_dynamics(df_global, car_save_dir, vin_id)
    plot_charging_curve(df_global, car_save_dir, vin_id)
    plot_temp_hysteresis(df_global, car_save_dir, vin_id)
    plot_motor_efficiency(df_global, car_save_dir, vin_id)

    print(f"🎉 车辆 [{vin_id}] 的多维度诊断图表已固化至：{car_save_dir}/")


# ==========================================
# 批处理入口：自动识别 processedData 目录中的 VIN
# ==========================================
def run_visualizations_for_all_cars(data_dir="processedData", save_dir="visualizations"):
    """自动扫描分片目录，智能提取所有不同的 VIN 号进行全局分析"""
    search_pattern = os.path.join(data_dir, "vin_*_chunk_*.csv")
    all_shards = glob.glob(search_pattern)

    if not all_shards:
        print(f"❌ 在 [{data_dir}] 目录下没有检索到任何 vin_*_chunk_*.csv 分片数据！")
        print("💡 请确保数据切分工具或特征提取逻辑已提前生成了 CSV 碎片。")
        return

    # 从所有文件名中解析并提取唯一非重复的 vin_id 集合
    unique_vins = set()
    for file_path in all_shards:
        filename = os.path.basename(file_path)
        try:
            # 文件名：vin_6993_chunk_0.csv -> 拆分出 6993
            vin_id = filename.split('_')[1]
            unique_vins.add(vin_id)
        except IndexExpandedError:
            continue

    print(f"🔍 全局扫描完毕。在 [{data_dir}] 目录中共发现 {len(unique_vins)} 辆独立车辆：{list(unique_vins)}")
    print("🚀 开始依次启动全自动绘图引擎...")

    for vin_id in sorted(list(unique_vins)):
        run_visualizations_for_vin_chunks(data_dir, vin_id, save_dir)


# ============================================================================
# 第二部分：深度学习模型训练结果可视化（Loss 曲线 / 预测对比 / 残差 / 特征重要性 / 模型对比）
# ============================================================================

import torch
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def _get_model_class(model_type):
    """根据模型类型字符串返回对应的模型类"""
    if model_type == "lstm":
        from SpdToPwr.model_lstm import LSTMPredictor
        return LSTMPredictor
    elif model_type == "bilstm":
        from SpdToPwr.model_bilstm import BiLSTMPredictor
        return BiLSTMPredictor
    elif model_type == "bilstm_attention":
        from SpdToPwr.model_improved import EVPowerImprovedModel
        return EVPowerImprovedModel
    else:
        raise ValueError(f"未知模型类型: {model_type}，可选: lstm / bilstm / bilstm_attention")


def _get_config_class(model_type):
    """根据模型类型返回对应的配置类"""
    if model_type == "lstm":
        from configsP.lstm_config import LSTMConfig
        return LSTMConfig()
    elif model_type == "bilstm":
        from configsP.bilstm_config import BiLSTMConfig
        return BiLSTMConfig()
    elif model_type == "bilstm_attention":
        from configsP.Bilstm_attention import BaseConfig
        return BaseConfig()
    else:
        raise ValueError(f"未知模型类型: {model_type}")


# ------------------------------------------------
# 图 1：Loss 曲线 + 学习率衰减
# ------------------------------------------------
def plot_loss_curves(history_path, save_dir, model_name):
    if not os.path.exists(history_path):
        print(f"⚠️ 找不到训练历史文件: {history_path}，跳过 Loss 曲线。")
        return

    with open(history_path, 'r') as f:
        history = json.load(f)

    fig, ax1 = plt.subplots(figsize=(12, 5))

    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], color='#1f77b4', linewidth=1.5, label='训练集 Loss')
    ax1.plot(epochs, history['val_loss'], color='#ff7f0e', linewidth=1.5, label='验证集 Loss')

    # 标注最佳 epoch
    best_epoch = np.argmin(history['val_loss']) + 1
    best_val = min(history['val_loss'])
    ax1.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax1.scatter(best_epoch, best_val, color='red', s=80, zorder=5,
                label=f'最佳 epoch={best_epoch} (Val={best_val:.6f})')

    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("MSE Loss", fontsize=12)
    ax1.set_title(f"[{model_name}] 训练与验证 Loss 曲线", fontsize=14)
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # 学习率副轴
    if 'lr_history' in history and len(history['lr_history']) > 0:
        ax2 = ax1.twinx()
        ax2.plot(epochs, history['lr_history'], color='green', linewidth=1, linestyle=':', alpha=0.7, label='学习率')
        ax2.set_ylabel("Learning Rate", color='green', fontsize=11)
        ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"loss_curve_{model_name}.png"))
    plt.close()
    print(f"📈 Loss 曲线已保存: {save_dir}/loss_curve_{model_name}.png")


# ------------------------------------------------
# 图 2：预测 vs 真实散点图（密度着色）
# ------------------------------------------------
def plot_prediction_scatter(trues, preds, target_names, save_dir, model_name):
    n_targets = len(target_names)
    fig, axes = plt.subplots(1, n_targets, figsize=(6 * n_targets, 5.5))
    if n_targets == 1:
        axes = [axes]

    for i, target_name in enumerate(target_names):
        ax = axes[i]
        t = trues[:, i]
        p = preds[:, i]

        # 密度散点图
        ax.hist2d(t, p, bins=80, cmap='Blues', alpha=0.9)
        # 对角线参考线
        lims = [min(t.min(), p.min()), max(t.max(), p.max())]
        ax.plot(lims, lims, 'r--', linewidth=1.5, label='理想预测线 (y=x)')

        r2 = r2_score(t, p)
        rmse = np.sqrt(mean_squared_error(t, p))
        unit = "kW" if "功率" in target_name else "℃"
        ax.set_xlabel(f"真实值 ({unit})", fontsize=11)
        ax.set_ylabel(f"预测值 ({unit})", fontsize=11)
        ax.set_title(f"[{model_name}] {target_name}\nR²={r2:.4f}  RMSE={rmse:.3f} {unit}", fontsize=11)
        ax.legend(loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"scatter_{model_name}.png"))
    plt.close()
    print(f"📈 预测散点图已保存: {save_dir}/scatter_{model_name}.png")


# ------------------------------------------------
# 图 3：时序对比图（全量 + 500 步放大）
# ------------------------------------------------
def plot_timeseries_comparison(trues, preds, target_names, save_dir, model_name, vin_id):
    n_targets = len(target_names)
    total_steps = len(trues)

    for scope, scope_label, steps in [("FULL", "全量时序全景图", total_steps),
                                       ("ZOOM", "局部动态跟随放大图", min(500, total_steps))]:
        fig, axes = plt.subplots(n_targets, 1, figsize=(16, 4 * n_targets), squeeze=False)

        for i, target_name in enumerate(target_names):
            ax = axes[i, 0]
            t = trues[:steps, i]
            p = preds[:steps, i]
            x_axis = np.arange(steps)

            ax.plot(x_axis, t, color='black', linewidth=1.2, alpha=0.85, label='真实值')
            ax.plot(x_axis, p, color='#e74c3c', linewidth=1.2, linestyle='--', alpha=0.9,
                    label=f'{model_name} 预测')

            unit = "kW" if "功率" in target_name else "℃"
            ax.set_ylabel(f"{target_name} ({unit})", fontsize=12)
            ax.set_title(f"[{model_name}] [{scope_label}] {target_name} (VIN: {vin_id})", fontsize=13)
            ax.legend(loc='upper right')
            ax.grid(True, linestyle='--', alpha=0.5)

        ax.set_xlabel("序列时间步", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"timeseries_{scope}_{model_name}.png"))
        plt.close()
        print(f"📈 时序对比图 ({scope_label}) 已保存: {save_dir}/timeseries_{scope}_{model_name}.png")


# ------------------------------------------------
# 图 4：残差分析（直方图 + 残差 vs 预测值）
# ------------------------------------------------
def plot_residual_analysis(trues, preds, target_names, save_dir, model_name):
    n_targets = len(target_names)
    fig, axes = plt.subplots(n_targets, 2, figsize=(14, 5 * n_targets))
    if n_targets == 1:
        axes = axes.reshape(1, -1)

    for i, target_name in enumerate(target_names):
        t = trues[:, i]
        p = preds[:, i]
        residuals = t - p
        unit = "kW" if "功率" in target_name else "℃"

        # 残差分布直方图 + KDE
        ax1 = axes[i, 0]
        ax1.hist(residuals, bins=60, density=True, color='steelblue', alpha=0.7, edgecolor='white')
        from scipy import stats as scipy_stats
        try:
            kde = scipy_stats.gaussian_kde(residuals)
            x_kde = np.linspace(residuals.min(), residuals.max(), 200)
            ax1.plot(x_kde, kde(x_kde), color='red', linewidth=2, label='KDE')
        except Exception:
            pass
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax1.axvline(x=residuals.mean(), color='orange', linestyle='--', linewidth=1.5,
                    label=f'均值={residuals.mean():.4f} {unit}')
        ax1.set_xlabel(f"残差 ({unit})", fontsize=11)
        ax1.set_ylabel("密度", fontsize=11)
        ax1.set_title(f"[{model_name}] {target_name} 残差分布", fontsize=11)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # 残差 vs 预测值散点图
        ax2 = axes[i, 1]
        ax2.scatter(p, residuals, alpha=0.3, s=8, color='steelblue')
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax2.set_xlabel(f"预测值 ({unit})", fontsize=11)
        ax2.set_ylabel(f"残差 ({unit})", fontsize=11)
        ax2.set_title(f"[{model_name}] {target_name} 残差 vs 预测值", fontsize=11)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"residual_{model_name}.png"))
    plt.close()
    print(f"📈 残差分析图已保存: {save_dir}/residual_{model_name}.png")


# ------------------------------------------------
# 图 5：特征重要性（排列重要性 Permutation Importance）
# ------------------------------------------------
def plot_feature_importance(model, test_loader, scaler_y, feature_names, target_names, config,
                            save_dir, model_name):
    model.eval()
    device = config.device

    # 收集基线预测
    all_baseline_preds = []
    all_trues = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            preds = model(x_batch).cpu().numpy()
            all_baseline_preds.append(preds)
            all_trues.append(y_batch.numpy())

    all_baseline_preds = np.vstack(all_baseline_preds)
    all_trues = np.vstack(all_trues)

    # 反归一化
    baseline_preds_real = scaler_y.inverse_transform(all_baseline_preds)
    trues_real = scaler_y.inverse_transform(all_trues)

    baseline_rmse = np.sqrt(mean_squared_error(trues_real[:, 0], baseline_preds_real[:, 0]))

    # 逐个特征 shuffle 并计算 RMSE 上升幅度
    importance_scores = []
    n_features = len(feature_names)

    for feat_idx in range(n_features):
        shuffled_preds_list = []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                # Shuffle 该特征在所有时间步上的值
                x_shuffled = x_batch.clone()
                perm = torch.randperm(x_batch.size(0) * x_batch.size(1), device=device)
                x_shuffled[:, :, feat_idx] = x_batch[:, :, feat_idx].reshape(-1)[perm].reshape(
                    x_batch.size(0), x_batch.size(1))
                preds = model(x_shuffled).cpu().numpy()
                shuffled_preds_list.append(preds)

        shuffled_preds = np.vstack(shuffled_preds_list)
        shuffled_preds_real = scaler_y.inverse_transform(shuffled_preds)
        shuffled_rmse = np.sqrt(mean_squared_error(trues_real[:, 0], shuffled_preds_real[:, 0]))
        importance_scores.append(max(shuffled_rmse - baseline_rmse, 0))

    # 排序并绘图
    sorted_idx = np.argsort(importance_scores)[::-1]
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_scores = [importance_scores[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(sorted_scores)))
    bars = ax.barh(range(len(sorted_scores)), sorted_scores, color=colors[::-1], edgecolor='black', linewidth=0.5)

    ax.set_yticks(range(len(sorted_scores)))
    ax.set_yticklabels(sorted_names[::-1], fontsize=11)
    ax.set_xlabel(f"RMSE 上升 (基准 RMSE={baseline_rmse:.3f})", fontsize=12)
    ax.set_title(f"[{model_name}] 排列重要性 — {target_names[0]}", fontsize=14)
    ax.invert_yaxis()
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"feature_importance_{model_name}.png"))
    plt.close()
    print(f"📈 特征重要性图已保存: {save_dir}/feature_importance_{model_name}.png")


# ------------------------------------------------
# 图 6：多模型对比（Loss 曲线 + 指标柱状图）
# ------------------------------------------------
def plot_model_comparison(model_results, target_names, save_dir, vin_id):
    """
    model_results: dict, key=模型名, value=dict with keys:
        'history_path', 'rmse', 'mae', 'r2', 'nMAE'
    """
    if len(model_results) < 2:
        print("⚠️ 模型不足 2 个，跳过多模型对比图。")
        return

    # 6.1 Loss 曲线对比
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = {'LSTM': '#1f77b4', 'BiLSTM': '#2ca02c', 'BiLSTM+Attention': '#ff7f0e'}

    for model_name, info in model_results.items():
        hist_path = info.get('history_path')
        if hist_path and os.path.exists(hist_path):
            with open(hist_path, 'r') as f:
                history = json.load(f)
            epochs = range(1, len(history['val_loss']) + 1)
            color = colors.get(model_name, None)
            ax.plot(epochs, history['val_loss'], linewidth=1.5, color=color,
                    label=f"{model_name} (最佳={min(history['val_loss']):.4f})")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("验证集 MSE Loss", fontsize=12)
    ax.set_title(f"多模型验证 Loss 曲线对比 (VIN: {vin_id})", fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "comparison_loss_curves.png"))
    plt.close()
    print(f"📈 多模型 Loss 对比图已保存: {save_dir}/comparison_loss_curves.png")

    # 6.2 指标柱状图
    metrics = ['rmse', 'mae', 'r2']
    metric_labels = ['RMSE', 'MAE', 'R²']
    model_names = list(model_results.keys())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for j, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[j]
        values = [model_results[m].get(metric, 0) for m in model_names]
        bar_colors = [colors.get(m, '#999') for m in model_names]
        bars = ax.bar(model_names, values, color=bar_colors, edgecolor='black', linewidth=0.5)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02,
                    f'{val:.3f}', ha='center', fontsize=10)

        ax.set_title(f"{label} 对比", fontsize=13)
        ax.set_ylabel(label, fontsize=12)
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    plt.suptitle(f"多模型性能对比 — {target_names[0]} (VIN: {vin_id})", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "comparison_metrics.png"))
    plt.close()
    print(f"📈 多模型指标对比图已保存: {save_dir}/comparison_metrics.png")


# ------------------------------------------------
# 单模型完整可视化入口
# ------------------------------------------------
def run_single_model_visualization(model_type, vin_id="6993", module="power_prediction",
                                   shards_dir="processedData"):
    """
    对单个已训练模型生成全套 5 张训练结果可视化图。
    model_type: "lstm" / "bilstm" / "bilstm_attention"
    """
    from data_utils import create_training_dataset_from_shards, MODULE_FEATURES

    config = _get_config_class(model_type)
    ModelClass = _get_model_class(model_type)

    feat_cfg = MODULE_FEATURES[module]
    feature_names = feat_cfg["features"]
    target_names = feat_cfg["targets"]
    input_dim = len(feature_names)
    output_dim = len(target_names)

    # 模型名映射
    model_name_map = {
        "lstm": "LSTM",
        "bilstm": "BiLSTM",
        "bilstm_attention": "BiLSTM+Attention"
    }
    model_label = model_name_map.get(model_type, model_type)
    save_dir = f"training_results/{model_type}"
    ensure_dir(save_dir)

    # 路径
    model_path = f"models/best_model_{model_type}_{module}_{vin_id}.pth"
    if model_type == "bilstm_attention":
        model_path = f"models/best_model_{module}_{vin_id}.pth"
    history_path = f"history/history_{model_type}_{module}_{vin_id}.json"
    if model_type == "bilstm_attention":
        history_path = f"history/history_{module}_{vin_id}.json"

    print(f"\n📊 开始生成 [{model_label}] 模型训练结果可视化...")
    print(f"   车辆: {vin_id} | 任务: {module} | 输出目录: {save_dir}/")
    print("-" * 60)

    # 检查模型是否存在
    if not os.path.exists(model_path):
        print(f"❌ 未找到模型文件: {model_path}，请先训练该模型。")
        return

    # 加载数据
    _, _, test_loader, scaler_y = create_training_dataset_from_shards(
        data_dir=shards_dir, vin_id=vin_id, module=module,
        seq_len=config.seq_len, batch_size=config.batch_size
    )

    # 加载模型
    model = ModelClass(input_dim=input_dim, hidden_dim=config.hidden_dim, output_dim=output_dim)
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model.to(config.device)
    model.eval()

    # 全量推理
    all_preds, all_trues = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(config.device)
            preds = model(x_batch).cpu().numpy()
            all_preds.append(preds)
            all_trues.append(y_batch.numpy())

    all_preds = np.vstack(all_preds)
    all_trues = np.vstack(all_trues)
    preds_real = scaler_y.inverse_transform(all_preds)
    trues_real = scaler_y.inverse_transform(all_trues)

    # 计算指标
    metrics = {}
    for i, target_name in enumerate(target_names):
        p, t = preds_real[:, i], trues_real[:, i]
        metrics['rmse'] = np.sqrt(mean_squared_error(t, p))
        metrics['mae'] = mean_absolute_error(t, p)
        metrics['r2'] = r2_score(t, p)
        val_range = np.max(t) - np.min(t) or 1e-5
        metrics['nMAE'] = (metrics['mae'] / val_range) * 100

    # 图 1：Loss 曲线
    plot_loss_curves(history_path, save_dir, model_label)

    # 图 2：预测 vs 真实散点图
    plot_prediction_scatter(trues_real, preds_real, target_names, save_dir, model_label)

    # 图 3：时序对比图
    plot_timeseries_comparison(trues_real, preds_real, target_names, save_dir, model_label, vin_id)

    # 图 4：残差分析
    plot_residual_analysis(trues_real, preds_real, target_names, save_dir, model_label)

    # 图 5：特征重要性
    plot_feature_importance(model, test_loader, scaler_y, feature_names, target_names, config,
                            save_dir, model_label)

    # 打印指标摘要
    print(f"\n{'=' * 50}")
    print(f"  [{model_label}] 测试集指标摘要")
    print(f"  RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f} | R²: {metrics['r2']:.4f} | F.S.: {metrics['nMAE']:.2f}%")
    print(f"{'=' * 50}")
    print(f"🎉 [{model_label}] 全部可视化图表已生成至: {save_dir}/")
    return metrics, history_path


# ------------------------------------------------
# 多模型对比可视化入口
# ------------------------------------------------
def run_model_comparison(vin_id="6993", module="power_prediction", shards_dir="processedData"):
    """
    依次加载 LSTM / BiLSTM / BiLSTM+Attention 的模型和训练历史，生成对比图。
    如果某个模型未训练，则跳过。
    """
    from data_utils import MODULE_FEATURES

    target_names = MODULE_FEATURES[module]["targets"]
    save_dir = "training_results/comparison"
    ensure_dir(save_dir)

    model_types = ["lstm", "bilstm", "bilstm_attention"]
    model_labels_map = {"lstm": "LSTM", "bilstm": "BiLSTM", "bilstm_attention": "BiLSTM+Attention"}

    model_results = {}

    for model_type in model_types:
        model_label = model_labels_map[model_type]

        # 确定路径
        if model_type == "bilstm_attention":
            model_path = f"models/best_model_{module}_{vin_id}.pth"
            history_path = f"history/history_{module}_{vin_id}.json"
        else:
            model_path = f"models/best_model_{model_type}_{module}_{vin_id}.pth"
            history_path = f"history/history_{model_type}_{module}_{vin_id}.json"

        if not os.path.exists(model_path):
            print(f"⚠️ [{model_label}] 模型未找到 ({model_path})，跳过。")
            continue

        print(f"🔍 正在评估 [{model_label}] ...")
        try:
            config = _get_config_class(model_type)
            ModelClass = _get_model_class(model_type)

            feat_cfg = MODULE_FEATURES[module]
            input_dim = len(feat_cfg["features"])
            output_dim = len(feat_cfg["targets"])

            _, _, test_loader, scaler_y = create_training_dataset_from_shards(
                data_dir=shards_dir, vin_id=vin_id, module=module,
                seq_len=config.seq_len, batch_size=config.batch_size
            )

            model = ModelClass(input_dim=input_dim, hidden_dim=config.hidden_dim, output_dim=output_dim)
            model.load_state_dict(torch.load(model_path, map_location=config.device))
            model.to(config.device)
            model.eval()

            all_preds, all_trues = [], []
            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    x_batch = x_batch.to(config.device)
                    preds = model(x_batch).cpu().numpy()
                    all_preds.append(preds)
                    all_trues.append(y_batch.numpy())

            all_preds = np.vstack(all_preds)
            all_trues = np.vstack(all_trues)
            preds_real = scaler_y.inverse_transform(all_preds)
            trues_real = scaler_y.inverse_transform(all_trues)

            t, p = trues_real[:, 0], preds_real[:, 0]
            model_results[model_label] = {
                'history_path': history_path,
                'rmse': np.sqrt(mean_squared_error(t, p)),
                'mae': mean_absolute_error(t, p),
                'r2': r2_score(t, p),
            }
            print(f"   ✅ [{model_label}] RMSE={model_results[model_label]['rmse']:.4f}")
        except Exception as e:
            print(f"   ❌ [{model_label}] 评估失败: {e}")

    if len(model_results) >= 2:
        plot_model_comparison(model_results, target_names, save_dir, vin_id)
    else:
        print("⚠️ 至少需要 2 个已训练模型才能生成对比图。")

    print(f"\n🎉 多模型对比完成！图表保存至: {save_dir}/")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "training":
        # 用法: python visualize_results.py training <model_type> <vin_id> <module>
        model_type = sys.argv[2] if len(sys.argv) > 2 else "lstm"
        vin_id = sys.argv[3] if len(sys.argv) > 3 else "6993"
        module = sys.argv[4] if len(sys.argv) > 4 else "power_prediction"
        if model_type == "compare":
            run_model_comparison(vin_id=vin_id, module=module)
        else:
            run_single_model_visualization(model_type=model_type, vin_id=vin_id, module=module)
    else:
        # 默认：运行数据 EDA 可视化（原逻辑）
        run_visualizations_for_all_cars(data_dir="processedData", save_dir="visualizations")