import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import warnings

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
    print(f"📈 [{vin_id}] 正在绘制：全局 SOC 与状态瀑布图...")
    plt.figure(figsize=(18, 5))

    plt.plot(df['时间轴'], df['SOC'], color='#2ca02c', linewidth=1.5, label='SOC (%)')

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
    print(f"📈 [{vin_id}] 正在绘制：单次行程动态切片与三电功率图...")
    df_drive = df[df['Ready'] == 1].reset_index(drop=True)
    if len(df_drive) < 1500:
        slice_df = df_drive
    else:
        start_idx = len(df_drive) // 2
        slice_df = df_drive.iloc[start_idx: start_idx + 1500]

    if slice_df.empty:
        print(f"⚠️ [{vin_id}] 无有效的行驶数据，跳过行程动态图。")
        return

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

    # 子图2：三电功率溯源
    ax2 = axes[1]
    ax2.plot(slice_df['时间轴'], slice_df['前电机机械功率'], label='前电机功率 (kW)', color='purple', alpha=0.8)

    total_power = (slice_df['电池包总电压'] * slice_df['电池包总电流']) / 1000
    ax2.plot(slice_df['时间轴'], total_power, label='电池包总输出电功率 (kW)', color='black', linewidth=1.5,
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
    print(f"📈 [{vin_id}] 正在绘制：快充特性曲线 (CC-CV)...")
    if '充电状态' not in df.columns:
        print(f"⚠️ [{vin_id}] 缺少 '充电状态' 字段，跳过充电分析。")
        return

    df_charge = df[df['充电状态'] == 5].reset_index(drop=True)
    if df_charge.empty:
        print(f"⚠️ [{vin_id}] 未找到充电状态数据，该车可能没有充电记录。")
        return

    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    df_charge = df_charge.sort_values(by='SOC').drop_duplicates(subset=['SOC'])

    ax1.plot(df_charge['SOC'], df_charge['充电电流'], color='blue', linewidth=2, label='充电电流 (A)')
    ax1.plot(df_charge['SOC'], df_charge['充电功率'], color='green', linewidth=2, linestyle='--', label='充电功率 (kW)')
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
    print(f"📈 [{vin_id}] 正在绘制：热惯性与温度迟滞效应...")
    df_drive = df[df['Ready'] == 1].reset_index(drop=True)
    if len(df_drive) > 3000:
        slice_df = df_drive.iloc[1000:4000]
    else:
        slice_df = df_drive

    if slice_df.empty:
        return

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()

    time_minutes = (slice_df['时间轴'] - slice_df['时间轴'].iloc[0]).dt.total_seconds() / 60
    current_heat = (slice_df['电池包总电流'].abs() / 100) ** 2

    ax1.fill_between(time_minutes, 0, current_heat, color='orange', alpha=0.4, label='电池包电流波动 (模拟发热量)')
    ax2.plot(time_minutes, slice_df['最高电芯温度'], color='red', linewidth=2.5, label='最高电芯温度 (℃)')

    ax1.set_xlabel("持续行驶时间 (分钟)", fontsize=12)
    ax1.set_ylabel("电流发热指标 (归一化)", color='orange', fontsize=12)
    ax2.set_ylabel("温度 (℃)", color='red', fontsize=12)
    plt.title(f"维度4：热系统的迟滞性分析 (VIN: {vin_id})", fontsize=14)

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{vin_id}_4_Temperature_Hysteresis.png"))
    plt.close()


# ==========================================
# 维度 5：电机效率层（二维散点 MAP 图雏形）
# ==========================================
def plot_motor_efficiency(df, save_dir, vin_id):
    print(f"📈 [{vin_id}] 正在绘制：电机效率散点图...")
    mask = (df['Ready'] == 1) & (df['车速'] > 10) & (df['前电机机械功率'] > 5)
    df_motor = df[mask].copy()

    if df_motor.empty:
        print(f"⚠️ [{vin_id}] 数据量不足以绘制电机MAP图。")
        return

    df_motor['百公里瞬时电耗'] = (df_motor['电池包总电压'] * df_motor['电池包总电流'] / 1000) / df_motor['车速'] * 100
    df_motor = df_motor[(df_motor['百公里瞬时电耗'] > 0) & (df_motor['百公里瞬时电耗'] < 50)]

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
# 统一调用入口：处理单辆车
# ==========================================
def run_visualizations_for_vin(parquet_path, vin_id, save_dir="visualizations"):
    print(f"\n📂 开始生成车辆 [{vin_id}] 的可视化报告...")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"找不到数据文件: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    # 为这辆车建一个专属文件夹存放图片
    car_save_dir = os.path.join(save_dir, str(vin_id))
    ensure_dir(car_save_dir)

    plot_global_soc(df, car_save_dir, vin_id)
    plot_trip_dynamics(df, car_save_dir, vin_id)
    plot_charging_curve(df, car_save_dir, vin_id)
    plot_temp_hysteresis(df, car_save_dir, vin_id)
    plot_motor_efficiency(df, car_save_dir, vin_id)

    print(f"🎉 车辆 [{vin_id}] 的图表已保存在：{car_save_dir}/ 中！")


# ==========================================
# 批处理入口：自动扫描并处理所有车辆
# ==========================================
def run_visualizations_for_all_cars(data_dir="data", save_dir="visualizations"):
    """自动扫描 data 目录下所有的 parquet 文件并执行可视化"""
    parquet_files = glob.glob(os.path.join(data_dir, "H56D_*_processed.parquet"))

    if not parquet_files:
        print(f"❌ 在 {data_dir} 目录下没有找到任何处理过的数据文件！请先运行 data_utils.py。")
        return

    print(f"🔍 发现了 {len(parquet_files)} 辆车的数据文件，准备开始批量绘图...")

    for file_path in parquet_files:
        # 从文件名中提取 VIN (例如 H56D_6993_processed.parquet 提取出 6993)
        filename = os.path.basename(file_path)
        vin_id = filename.split('_')[1]

        run_visualizations_for_vin(file_path, vin_id, save_dir)


if __name__ == "__main__":
    # 直接运行此脚本，将自动处理所有拆分好的车辆数据
    run_visualizations_for_all_cars(data_dir="data", save_dir="visualizations")