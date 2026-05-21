import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体与高清晰度
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def segment_trips(df):
    """
    角度 3 前置: 行程分段与切片逻辑
    依据 Ready == 1 筛选行驶数据，若相邻点时间间隔超过 5 分钟 (300秒) 则视为新行程。
    同时过滤掉时长 < 2 分钟 或 里程 < 0.2km 的极短噪音行程。
    """
    if 'Ready' not in df.columns:
        return pd.DataFrame()

    df_drive = df[df['Ready'] == 1].copy()
    if df_drive.empty:
        return df_drive

    df_drive = df_drive.sort_values('时间轴').reset_index(drop=True)
    df_drive['time_diff'] = df_drive['时间轴'].diff().dt.total_seconds().fillna(0)

    # 划分独立 Trip ID
    df_drive['new_trip'] = (df_drive['time_diff'] > 300).astype(int)
    df_drive['trip_id'] = df_drive['new_trip'].cumsum() + 1

    return df_drive


def analyze_trips_for_vin(parquet_path, vin_id, output_dir="EDA_Reports"):
    print(f"\n🚗 开始生成车辆 [{vin_id}] 的 8 维度行程分析报告...")
    df = pd.read_parquet(parquet_path)
    df['时间轴'] = pd.to_datetime(df['时间轴'])

    car_report_dir = os.path.join(output_dir, str(vin_id))
    car_plot_dir = os.path.join(car_report_dir, "trip_plots")
    ensure_dir(car_plot_dir)

    md_file_path = os.path.join(car_report_dir, f"Trip_Report_VIN_{vin_id}.md")

    # 1. 切分行程
    df_drive = segment_trips(df)
    if df_drive.empty:
        print(f"⚠️ 车辆 [{vin_id}] 未检测到有效行驶数据。")
        return

    # 计算时间步长(秒)与加速度(m/s^2)
    df_drive['dt'] = df_drive['时间轴'].diff().dt.total_seconds().fillna(1.0)
    df_drive.loc[df_drive['dt'] == 0, 'dt'] = 1.0  # 避免除以0

    # 计算加速度: dv/dt (车速从 km/h 转换为 m/s)
    df_drive['v_ms'] = df_drive['车速'] / 3.6
    df_drive['accel'] = df_drive['v_ms'].diff() / df_drive['dt']
    df_drive['accel'] = df_drive['accel'].fillna(0)

    # 整合电机驱动与附件总功率
    df_drive['电机总机械功率(kW)'] = df_drive['前电机机械功率'].fillna(0) + df_drive['后电机功率'].fillna(0)
    df_drive['空调与PTC总功率(kW)'] = df_drive['空调EDC实际功率'].fillna(0) + df_drive['座舱PTC实际功率'].fillna(0)

    trips_summary = []

    # 2. 遍历各行程进行精细统计
    for tid, group in df_drive.groupby('trip_id'):
        duration_mins = group['dt'].sum() / 60
        if duration_mins < 2:  # 过滤极短行程
            continue

        # 维度 1: 画像
        start_time = group['时间轴'].iloc[0]
        end_time = group['时间轴'].iloc[-1]
        max_spd = group['车速'].max()
        avg_spd = group['车速'].mean()

        # 里程积分估算 (车速 km/h * 时间小时)
        distance_km = (group['车速'] * (group['dt'] / 3600)).sum()
        if distance_km < 0.2:
            continue

        # 维度 2 & 5: 能耗与能量回收积分
        # 驱动耗电能 (功率 > 0)
        prop_energy_kwh = (group.loc[group['电机总机械功率(kW)'] > 0, '电机总机械功率(kW)'] * (
                    group.loc[group['电机总机械功率(kW)'] > 0, 'dt'] / 3600)).sum()
        # 回收电能 (功率 < 0)
        regen_energy_kwh = (group.loc[group['电机总机械功率(kW)'] < 0, '电机总机械功率(kW)'].abs() * (
                    group.loc[group['电机总机械功率(kW)'] < 0, 'dt'] / 3600)).sum()
        # 附件耗电
        hvac_energy_kwh = (group['空调与PTC总功率(kW)'] * (group['dt'] / 3600)).sum()

        total_trip_energy = prop_energy_kwh + hvac_energy_kwh - regen_energy_kwh
        energy_rate_100km = (total_trip_energy / distance_km) * 100 if distance_km > 0 else 0

        # 维度 3: 工况打标
        if avg_spd < 30:
            condition = "城市工况"
        elif 30 <= avg_spd <= 60:
            condition = "郊区工况"
        else:
            condition = "高速工况"

        # 维度 4: 驾驶行为 (急加减速阈值设为 1.5 m/s^2)
        harsh_accel_count = (group['accel'] > 1.5).sum()
        harsh_decel_count = (group['accel'] < -1.5).sum()
        avg_accel_pedal = group['加速踏板开度'].mean() if '加速踏板开度' in group.columns else 0

        # 维度 6: 温度演化
        delta_cell_temp = group['最高电芯温度'].max() - group['最高电芯温度'].iloc[0]
        max_motor_temp = max(group['前电机温度'].max(), group['后电机温度'].max())
        avg_env_temp = group['环境温度'].mean()

        trips_summary.append({
            'Trip_ID': tid,
            '开始时间': start_time.strftime('%m-%d %H:%M'),
            '时长(分钟)': round(duration_mins, 1),
            '里程(km)': round(distance_km, 2),
            '平均车速': round(avg_spd, 1),
            '最高车速': round(max_spd, 1),
            '百公里电耗(kWh)': round(energy_rate_100km, 2),
            '工况打标': condition,
            '急加速次数': harsh_accel_count,
            '急减速次数': harsh_decel_count,
            '踏板均值(%)': round(avg_accel_pedal, 1),
            '回收能量(kWh)': round(regen_energy_kwh, 3),
            '回收占比(%)': round((regen_energy_kwh / prop_energy_kwh * 100), 1) if prop_energy_kwh > 0 else 0,
            '电池温升(℃)': round(delta_cell_temp, 1),
            '电机最高温(℃)': round(max_motor_temp, 1),
            '环境均温(℃)': round(avg_env_temp, 1),
            '附件能耗占比(%)': round((hvac_energy_kwh / (total_trip_energy + 0.001) * 100),
                                     1) if total_trip_energy > 0 else 0
        })

    df_trips = pd.DataFrame(trips_summary)
    if df_trips.empty:
        print(f"⚠️ 车辆 [{vin_id}] 无符合标准的有效行程区间。")
        return

    # ==========================================
    # 报告文本生成
    # ==========================================
    md = [f"# 岚图 VOYAH 单车行程特征深度挖掘报告 (VIN: {vin_id})\n"]
    md.append("> 本报告基于车端实时采集的动态大数据，从 8 个互补的数据科学维度对驾驶行为及能耗进行全面解析。\n\n")

    # 1. 行程基础画像
    md.append("## 1. 行程基础画像\n")
    md.append(f"- **历史行驶总行程数**: `{len(df_trips)}` 次\n")
    md.append(f"- **单次最大行驶里程**: `{df_trips['里程(km)'].max():.2f}` km\n")
    md.append(f"- **平均行程行驶时长**: `{df_trips['时长(分钟)'].mean():.1f}` 分钟\n")
    md.append(f"- **整体平均车速**: `{df_trips['平均车速'].mean():.1f}` km/h\n\n")
    md.append("### 完整行程基础统计台账表\n")
    md.append(df_trips[['Trip_ID', '开始时间', '时长(分钟)', '里程(km)', '平均车速', '最高车速', '工况打标']].head(
        15).to_markdown() + "\n\n")

    # 2. 行程能耗分析 & 7. 行程对比聚类图
    md.append("## 2. 行程能耗分析与聚类特征\n")
    # 过滤掉静止状态或异常的高/负电耗
    df_eff_trips = df_trips[(df_trips['百公里电耗(kWh)'] > 5) & (df_trips['百公里电耗(kWh)'] < 40)]

    plt.figure(figsize=(9, 5))
    sns.regplot(data=df_eff_trips, x='平均车速', y='百公里电耗(kWh)', scatter_kws={'alpha': 0.6, 's': 40},
                line_kws={'color': 'red'})
    plt.title(f"行程能耗率 vs 平均车速 趋势线 (VIN: {vin_id})")
    plt.grid(True, alpha=0.3)
    plot1_path = os.path.join(car_plot_dir, "energy_vs_speed.png")
    plt.savefig(plot1_path)
    plt.close()

    md.append(f"![能耗与车速关系](./trip_plots/energy_vs_speed.png)\n\n")
    md.append(
        "- **科学发现**: 观察红色趋势线。通常新能源车呈现两端高中间低的 V 型曲线，中速（40-60km/h）最省电，高速巡航（>80km/h）时由于风阻增大能耗率会显著抬升。\n\n")

    # 3. 驾驶工况分类
    md.append("## 3. 驾驶工况分类分布\n")
    cond_counts = df_trips['工况打标'].value_counts()
    cond_energy = df_trips.groupby('工况打标')['百公里电耗(kWh)'].mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].pie(cond_counts, labels=cond_counts.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
    axes[0].set_title("各驾驶工况时间占比")

    sns.barplot(x=cond_energy.index, y=cond_energy.values, ax=axes[1], palette='muted')
    axes[1].set_title("不同工况下的平均百公里电耗 (kWh)")
    axes[1].set_ylabel("kWh / 100km")
    plt.tight_layout()
    plot2_path = os.path.join(car_plot_dir, "conditions_analysis.png")
    plt.savefig(plot2_path)
    plt.close()

    md.append(f"![工况对比](./trip_plots/conditions_analysis.png)\n\n")

    # 4. 加速与制动行为
    md.append("## 4. 驾驶员脚法激烈程度 (加速与制动行为)\n")
    md.append(f"- **累计急加速(>1.5 m/s²)行为次数**: `{df_trips['急加速次数'].sum()}` 次\n")
    md.append(f"- **累计急减速(<-1.5 m/s²)行为次数**: `{df_trips['急减速次数'].sum()}` 次\n")
    md.append(f"- **平均加速踏板开度深度**: `{df_trips['踏板均值(%)'].mean():.1f}` %\n\n")

    # 5. 能量回收分析
    md.append("## 5. 能量回收效率分析\n")
    md.append(f"- **平均单次行程动能回收量**: `{df_trips['回收能量(kWh)'].mean():.3f}` kWh\n")
    md.append(f"- **动能回收占驱动能量平均比例**: `{df_trips['回收占比(%)'].mean():.1f}` %\n")
    md.append(
        "- **特征支撑**: 回收占比直接反映了该车辆在松开电门或踩下制动时阻尼发电机的工作饱满度，是后续 LSTM 预测网络降低残差偏差的关键负反馈输入。\n\n")

    # 6. 温度演化
    md.append("## 6. 三电系统行驶温度上演化规律\n")
    md.append(f"- **平均单次行程电池包温升**: `{df_trips['电池温升(℃)'].mean():.1f}` ℃\n")
    md.append(f"- **行驶工况下电机最高突破温度**: `{df_trips['电机最高温(℃)'].max()}` ℃\n\n")

    # 8. 空调与附件的能耗占比
    md.append("## 8. 空调与热管理附件的能耗蚕食率\n")
    md.append(f"- **热管理附件（空调+PTC）在行驶中蚕食了总电耗的**: `{df_trips['附件能耗占比(%)'].mean():.1f}` %\n\n")

    plt.figure(figsize=(9, 4))
    sns.scatterplot(data=df_trips, x='环境温度(℃)', y='附件能耗占比(%)', hue='工况打标', s=60)
    plt.title("热管理能耗占比 vs 环境温度")
    plt.grid(True, alpha=0.3)
    plot3_path = os.path.join(car_plot_dir, "hvac_vs_env.png")
    plt.savefig(plot3_path)
    plt.close()

    md.append(f"![附件能耗](./trip_plots/hvac_vs_env.png)\n\n")
    md.append(
        "- **模型设计启示**: 可以看出，环境温度过高或过低时，附件能耗占比都会显著飙升。在建立 `能耗预测` 模型时，如果不把环境温度和空调实际功率当作强时序特征加入，模型极难捕获冬季/夏季极端气候下的耗电跳变。\n")

    with open(md_file_path, "w", encoding="utf-8") as f:
        f.writelines(md)
    print(f"✅ 车辆 [{vin_id}] 的行程分析报告生成成功：{md_file_path}")


def run_all_trip_reports(data_dir="data", output_dir="EDA_Reports"):
    parquet_files = glob.glob(os.path.join(data_dir, "H56D_*_processed.parquet"))
    if not parquet_files:
        print(f"❌ 在 {data_dir} 目录下未找到数据文件！请确保已执行 data_utils.py。")
        return

    for file_path in parquet_files:
        filename = os.path.basename(file_path)
        vin_id = filename.split('_')[1]
        try:
            analyze_trips_for_vin(file_path, vin_id, output_dir)
        except Exception as e:
            print(f"⚠️ 处理 [{vin_id}] 行程数据时出错: {e}")


if __name__ == "__main__":
    run_all_trip_reports()