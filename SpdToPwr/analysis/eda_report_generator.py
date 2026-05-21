import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体，防止图表中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def generate_eda_for_single_vin(parquet_path, vin_id, output_dir="EDA_Reports"):
    """
    为单辆车生成完整的数据分析报告和配套图表
    """
    print(f"\n📝 开始生成车辆 [{vin_id}] 的 EDA 报告...")
    df = pd.read_parquet(parquet_path)

    # 为该车辆创建专属的图表和报告文件夹
    car_report_dir = os.path.join(output_dir, str(vin_id))
    car_plot_dir = os.path.join(car_report_dir, "plots")
    ensure_dir(car_plot_dir)

    md_file_path = os.path.join(car_report_dir, f"EDA_数据分析报告_VIN_{vin_id}.md")

    # 初始化 Markdown 内容
    md = [f"# 岚图 VOYAH 单车三电数据 EDA 分析报告 (VIN: {vin_id})\n"]
    md.append("> 本报告由系统自动生成，旨在为深度学习模型（能耗/温度预测）提供数据先验特征支撑。\n\n")

    # ==========================================
    # 1. 基础描述统计
    # ==========================================
    md.append("## 1. 基础描述统计\n")
    md.append(f"- **总数据量**: {len(df)} 行\n")

    md.append("### 1.1 数据缺失率概况 (Top 10)")
    missing_rate = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    md.append("\n" + missing_rate.head(10).to_frame(name="缺失率(%)").to_markdown() + "\n\n")

    md.append("### 1.2 核心数值变量画像")
    core_cols = ['车速', 'SOC', '电池包总电压', '电池包总电流', '前电机机械功率', '最高电芯温度', '加速踏板开度']
    available_core = [c for c in core_cols if c in df.columns]
    desc_stats = df[available_core].describe().T[['mean', 'std', 'min', '50%', 'max']]
    desc_stats['极差'] = desc_stats['max'] - desc_stats['min']
    md.append("\n" + desc_stats.to_markdown() + "\n\n")

    # ==========================================
    # 2. 分布形态分析
    # ==========================================
    md.append("## 2. 分布形态分析\n")
    skewness = df[available_core].skew().to_frame(name="偏度 (Skewness)")
    md.append("### 2.1 偏度分析 (大于0为右偏，小于0为左偏)\n")
    md.append(skewness.to_markdown() + "\n\n")

    # 画分布图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    if '车速' in df.columns:
        sns.histplot(df['车速'].dropna(), kde=True, ax=axes[0], color='skyblue').set_title("车速分布")
    if '前电机机械功率' in df.columns:
        sns.histplot(df['前电机机械功率'].dropna(), kde=True, ax=axes[1], color='salmon').set_title("前电机功率分布")
    if 'SOC' in df.columns:
        sns.boxplot(y=df['SOC'].dropna(), ax=axes[2], color='lightgreen').set_title("SOC箱线图")
    plt.tight_layout()
    dist_plot_path = os.path.join(car_plot_dir, "distribution.png")
    plt.savefig(dist_plot_path)
    plt.close()

    md.append(f"![分布图](./plots/distribution.png)\n\n")

    # ==========================================
    # 3. 相关性矩阵
    # ==========================================
    md.append("## 3. 特征相关性分析\n")
    corr_cols = ['车速', '前电机机械功率', '电池包总电流', '最高电芯温度', '加速踏板开度', '空调EDC实际功率',
                 '环境温度']
    available_corr = [c for c in corr_cols if c in df.columns]

    if len(available_corr) > 1:
        corr_matrix = df[available_corr].corr(method='pearson')
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title(f"特征相关性热力图 (VIN: {vin_id})")
        plt.tight_layout()
        corr_plot_path = os.path.join(car_plot_dir, "correlation.png")
        plt.savefig(corr_plot_path)
        plt.close()

        md.append(f"![相关性热力图](./plots/correlation.png)\n\n")

    # ==========================================
    # 4. 时间序列基础特征
    # ==========================================
    md.append("## 4. 时间序列特征\n")
    df['时间轴'] = pd.to_datetime(df['时间轴'])
    time_diff = df['时间轴'].diff().dt.total_seconds().dropna()

    md.append(f"- **总时间跨度**: `{df['时间轴'].min()}` 至 `{df['时间轴'].max()}`\n")
    md.append(f"- **采样间隔 (中位数)**: `{time_diff.median()}` 秒\n\n")

    # ==========================================
    # 5. 工况分组对比
    # ==========================================
    md.append("## 5. 驾驶行为与工况对比\n")
    if '驾驶模式' in df.columns:
        md.append("### 5.1 不同驾驶模式下的动力平均特征\n")
        mode_grp = df.groupby('驾驶模式')[['前电机机械功率', '车速', '加速踏板开度']].mean().round(2)
        md.append(mode_grp.to_markdown() + "\n\n")

    # ==========================================
    # 6. 效率与衍生能耗指标计算
    # ==========================================
    md.append("## 6. 效率与衍生指标 (Feature Engineering)\n")
    df_derived = df.copy()

    # 瞬时能耗 (限制车速>5)
    df_derived['瞬时能耗率(kWh/km)'] = np.where(df_derived['车速'] > 5,
                                                df_derived['前电机机械功率'] / df_derived['车速'], np.nan)
    # 总输入电功率与附件占比
    df_derived['总输入电功率(kW)'] = (df_derived['电池包总电压'] * df_derived['电池包总电流']) / 1000
    if '座舱PTC实际功率' in df_derived.columns and '空调EDC实际功率' in df_derived.columns:
        df_derived['PTC功率占比(%)'] = (df_derived['座舱PTC实际功率'] / df_derived['总输入电功率(kW)']).replace(
            [np.inf, -np.inf], np.nan) * 100
        df_derived['空调功率占比(%)'] = (df_derived['空调EDC实际功率'] / df_derived['总输入电功率(kW)']).replace(
            [np.inf, -np.inf], np.nan) * 100

    derived_cols = ['瞬时能耗率(kWh/km)', '总输入电功率(kW)', 'PTC功率占比(%)', '空调功率占比(%)']
    available_derived = [c for c in derived_cols if c in df_derived.columns]

    md.append("### 6.1 衍生指标分布特征\n")
    md.append(df_derived[available_derived].describe().T[['mean', 'min', 'max']].to_markdown() + "\n\n")

    # 写入文件
    with open(md_file_path, "w", encoding="utf-8") as f:
        f.writelines(md)

    print(f"✅ 车辆 [{vin_id}] 分析完毕！报告已保存至: {md_file_path}")


# ==========================================
# 批处理主控：扫描所有车辆并生成报告
# ==========================================
def run_all_eda_reports(data_dir="data", output_dir="EDA_Reports"):
    ensure_dir(output_dir)
    parquet_files = glob.glob(os.path.join(data_dir, "H56D_*_processed.parquet"))

    if not parquet_files:
        print(f" 在 {data_dir} 目录下未找到数据文件！请先运行 data_utils.py。")
        return

    print(f" 发现了 {len(parquet_files)} 辆车的独立数据，准备批量生成分析报告...")

    for file_path in parquet_files:
        filename = os.path.basename(file_path)
        # 提取 VIN，例如从 H56D_6993_processed.parquet 提取出 6993
        vin_id = filename.split('_')[1]

        try:
            generate_eda_for_single_vin(file_path, vin_id, output_dir)
        except Exception as e:
            print(f"⚠️ 生成车辆 [{vin_id}] 报告时发生错误: {str(e)}")

    print(f"\n🎉 所有车辆的 EDA 分析报告均已生成完毕！请前往 [{output_dir}/] 目录查看。")


if __name__ == "__main__":
    run_all_eda_reports(data_dir="data", output_dir="EDA_Reports")