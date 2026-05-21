import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def identify_charging_events(df):
    """
    角度 6: 充电段识别与分段
    识别连续的充电事件。如果充电状态==5，且时间间隔小于5分钟（300秒），视为同一段充电。
    """
    if '充电状态' not in df.columns:
        return pd.DataFrame()

    df_charge = df[df['充电状态'] == 5].copy()
    if df_charge.empty:
        return df_charge

    df_charge = df_charge.sort_values('时间轴').reset_index(drop=True)
    df_charge['time_diff'] = df_charge['时间轴'].diff().dt.total_seconds().fillna(0)

    # 间隔大于5分钟（300秒）视为新的充电段
    df_charge['new_event'] = (df_charge['time_diff'] > 300).astype(int)
    df_charge['charge_id'] = df_charge['new_event'].cumsum() + 1

    return df_charge


def analyze_charging_for_vin(parquet_path, vin_id, output_dir="EDA_Reports"):
    print(f"\n🔋 开始生成车辆 [{vin_id}] 的 7 维度充电分析报告...")
    df = pd.read_parquet(parquet_path)
    df['时间轴'] = pd.to_datetime(df['时间轴'])

    car_report_dir = os.path.join(output_dir, str(vin_id))
    car_plot_dir = os.path.join(car_report_dir, "charging_plots")
    ensure_dir(car_plot_dir)

    md_file_path = os.path.join(car_report_dir, f"Charging_Report_VIN_{vin_id}.md")
    md = [f"# 岚图 VOYAH 单车充电深度分析报告 (VIN: {vin_id})\n"]
    md.append("> 本报告围绕充电曲线、效率、温升、偏好等 7 大维度自动生成。\n\n")

    # -----------------------------------------
    # 6. 充电段识别与分段
    # -----------------------------------------
    df_charge = identify_charging_events(df)
    if df_charge.empty:
        md.append("⚠️ **未检测到该车辆的有效充电数据 (充电状态 == 5)**\n")
        with open(md_file_path, "w", encoding="utf-8") as f:
            f.writelines(md)
        print(f"⚠️ 车辆 [{vin_id}] 无充电数据，报告已生成。")
        return

    event_counts = df_charge['charge_id'].nunique()
    md.append("## 6. 充电段识别与分段\n")
    md.append(f"- **独立充电总次数**: `{event_counts}` 次\n")

    # 统计每次充电的基础信息
    events_summary = []
    for cid, group in df_charge.groupby('charge_id'):
        duration_mins = (group['时间轴'].iloc[-1] - group['时间轴'].iloc[0]).total_seconds() / 60
        if duration_mins < 5:  # 过滤少于5分钟的极短噪音段
            continue

        start_soc = group['SOC'].iloc[0]
        end_soc = group['SOC'].iloc[-1]
        max_pwr = group['充电功率'].max()
        events_summary.append({
            'Charge_ID': cid,
            '开始时间': group['时间轴'].iloc[0].strftime('%Y-%m-%d %H:%M'),
            '时长(分钟)': round(duration_mins, 1),
            '起始SOC(%)': start_soc,
            '结束SOC(%)': end_soc,
            'SOC增量(%)': end_soc - start_soc,
            '峰值功率(kW)': round(max_pwr, 2)
        })

    df_events = pd.DataFrame(events_summary)
    if df_events.empty:
        md.append("- *所有充电片段均极短（<5分钟），可能为数据噪音。*\n\n")
        return

    md.append("### 有效充电记录清单 (Top 10)\n")
    md.append(df_events.head(10).to_markdown() + "\n\n")

    # -----------------------------------------
    # 4. 充电时长与充电量
    # -----------------------------------------
    md.append("## 4. 充电时长与充电量分析\n")
    md.append(f"- **平均充电时长**: `{df_events['时长(分钟)'].mean():.1f}` 分钟\n")
    md.append(f"- **平均 SOC 增量**: `{df_events['SOC增量(%)'].mean():.1f}` %\n")
    md.append(f"- **历史最高峰值功率**: `{df_events['峰值功率(kW)'].max()}` kW\n\n")

    # -----------------------------------------
    # 5. SOC 区间偏好
    # -----------------------------------------
    md.append("## 5. 用户充电 SOC 区间偏好\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df_events['起始SOC(%)'], bins=10, ax=axes[0], color='salmon', kde=True)
    axes[0].set_title("充电起始 SOC 分布")

    sns.histplot(df_events['结束SOC(%)'], bins=10, ax=axes[1], color='skyblue', kde=True)
    axes[1].set_title("充电结束 SOC 分布")

    plt.tight_layout()
    pref_plot_path = os.path.join(car_plot_dir, "soc_preference.png")
    plt.savefig(pref_plot_path)
    plt.close()
    md.append(f"![SOC偏好](./charging_plots/soc_preference.png)\n\n")

    # -----------------------------------------
    # 1. 充电曲线特征
    # -----------------------------------------
    md.append("## 1. 典型快充曲线特征 (CC-CV)\n")
    longest_event_id = df_events.loc[df_events['SOC增量(%)'].idxmax(), 'Charge_ID']
    df_typ = df_charge[df_charge['charge_id'] == longest_event_id].sort_values('SOC')

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ax1.plot(df_typ['SOC'], df_typ['充电电流'], color='blue', label='充电电流(A)', linewidth=2)
    ax1.plot(df_typ['SOC'], df_typ['充电功率'], color='green', label='充电功率(kW)', linewidth=2, linestyle='--')
    ax2.plot(df_typ['SOC'], df_typ['充电电压'], color='red', label='充电电压(V)', linewidth=2)

    ax1.set_xlabel('SOC (%)')
    ax1.set_ylabel('电流(A) / 功率(kW)', color='blue')
    ax2.set_ylabel('电压(V)', color='red')
    ax1.set_title(f"典型充电曲线 (Charge ID: {longest_event_id})")
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    curve_plot_path = os.path.join(car_plot_dir, "typical_curve.png")
    plt.savefig(curve_plot_path)
    plt.close()
    md.append(f"![充电曲线](./charging_plots/typical_curve.png)\n\n")

    # -----------------------------------------
    # 2. 充电效率分析
    # -----------------------------------------
    md.append("## 2. 充电效率分析\n")
    df_charge['输入电功率(kW)'] = df_charge['充电功率']
    df_charge['实际存入功率(kW)'] = (df_charge['电池包总电压'] * df_charge['电池包总电流'].abs()) / 1000

    mask_valid_pwr = df_charge['输入电功率(kW)'] > 2
    df_eff = df_charge[mask_valid_pwr].copy()
    df_eff['瞬时效率(%)'] = (df_eff['实际存入功率(kW)'] / df_eff['输入电功率(kW)']) * 100
    df_eff = df_eff[(df_eff['瞬时效率(%)'] > 50) & (df_eff['瞬时效率(%)'] <= 100)]  # 过滤噪声

    if not df_eff.empty:
        avg_eff = df_eff['瞬时效率(%)'].mean()
        md.append(f"- **平均充电机到电池的能量转换效率**: `{avg_eff:.2f}` %\n")

        # 分 SOC 区间效率
        df_eff['SOC区间'] = pd.cut(df_eff['SOC'], bins=[0, 30, 60, 80, 100],
                                   labels=['0-30%', '30-60%', '60-80%', '80-100%'])
        eff_by_soc = df_eff.groupby('SOC区间')['瞬时效率(%)'].mean().to_frame(name='平均效率(%)').dropna()
        md.append("### 不同 SOC 区间的充电效率\n")
        md.append(eff_by_soc.to_markdown() + "\n\n")

    # -----------------------------------------
    # 3. 充电温升分析
    # -----------------------------------------
    md.append("## 3. 充电温升与热管理分析\n")
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ax1.plot(df_typ['SOC'], df_typ['最高电芯温度'], color='red', label='最高电芯温度(℃)', linewidth=2)
    ax1.plot(df_typ['SOC'], df_typ['最低电芯温度'], color='orange', label='最低电芯温度(℃)', linewidth=2,
             linestyle='-.')

    if '电池PTC实际功率' in df_typ.columns:
        ax2.fill_between(df_typ['SOC'], 0, df_typ['电池PTC实际功率'], color='purple', alpha=0.3,
                         label='PTC加热功率(kW)')

    ax1.set_xlabel('SOC (%)')
    ax1.set_ylabel('温度 (℃)', color='red')
    ax2.set_ylabel('PTC功率 (kW)', color='purple')
    ax1.set_title("充电过程温升追踪")
    ax1.legend(loc='upper left')
    if '电池PTC实际功率' in df_typ.columns:
        ax2.legend(loc='upper right')

    temp_plot_path = os.path.join(car_plot_dir, "temp_rise.png")
    plt.savefig(temp_plot_path)
    plt.close()
    md.append(f"![温升分析](./charging_plots/temp_rise.png)\n\n")

    # -----------------------------------------
    # 7. 充电安全性指标
    # -----------------------------------------
    md.append("## 7. 充电安全性指标检测\n")
    max_temp_all = df_charge['最高电芯温度'].max()
    max_volt_all = df_charge['电池包总电压'].max()
    max_temp_diff = (df_charge['最高电芯温度'] - df_charge['最低电芯温度']).max()

    md.append(f"- **记录最高电芯温度**: `{max_temp_all}` ℃ (建议安全阈值: <55℃)\n")
    md.append(f"- **记录最高电池包电压**: `{max_volt_all}` V\n")
    md.append(f"- **最大电芯温差**: `{max_temp_diff:.1f}` ℃ (若温差过大可能暗示内阻不一致或热管理冷却不均)\n\n")

    with open(md_file_path, "w", encoding="utf-8") as f:
        f.writelines(md)
    print(f"✅ 车辆 [{vin_id}] 的充电分析报告已生成：{md_file_path}")


def run_all_charging_reports(data_dir="data", output_dir="EDA_Reports"):
    ensure_dir(output_dir)
    parquet_files = glob.glob(os.path.join(data_dir, "H56D_*_processed.parquet"))
    if not parquet_files:
        print(f"❌ 在 {data_dir} 目录下未找到数据文件！")
        return

    for file_path in parquet_files:
        filename = os.path.basename(file_path)
        vin_id = filename.split('_')[1]
        try:
            analyze_charging_for_vin(file_path, vin_id, output_dir)
        except Exception as e:
            print(f"⚠️ 处理 [{vin_id}] 充电数据时出错: {e}")


if __name__ == "__main__":
    run_all_charging_reports()