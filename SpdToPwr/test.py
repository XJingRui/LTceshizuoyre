import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

from configsP.Bilstm_attention import BaseConfig
from data_utils import create_training_dataset_from_shards, MODULE_FEATURES
from model_improved import EVPowerImprovedModel

warnings.filterwarnings('ignore')

# 配置 Matplotlib 中文字体与高清晰度
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# ===================== ⚙️ 核心配置与导入 =====================



def run_evaluation():
    config = BaseConfig()

    # ⚙️ 测试目标配置（必须与 train_bilstm_a.py 中一致）
    target_vin = "6993"
    business_module = "power_prediction"  # "power_prediction" 或 "temp_prediction"
    shards_dir = "processedData"

    input_dim = len(MODULE_FEATURES[business_module]["features"])
    target_names = MODULE_FEATURES[business_module]["targets"]
    output_dim = len(target_names)

    # 路径锁定
    model_path = os.path.join("models", f"best_model_{business_module}_{target_vin}.pth")
    test_results_dir = "test_results"
    os.makedirs(test_results_dir, exist_ok=True)

    print(f"\n🔍 启动模型测试评估引擎 | 车辆: {target_vin} | 业务: {business_module}")
    print("-" * 70)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ 找不到训练好的模型权重: {model_path}，请先运行 train.py！")

    # 1. 动态获取测试集数据加载器与标签的反归一化器（scaler_y）
    print("⏳ 正在组装测试集数据...")
    _, _, test_loader, scaler_y = create_training_dataset_from_shards(
        data_dir=shards_dir,
        vin_id=target_vin,
        module=business_module,
        seq_len=config.seq_len,
        batch_size=config.batch_size
    )

    # 2. 实例化模型并加载权重
    model = EVPowerImprovedModel(
        input_dim=input_dim,
        hidden_dim=config.hidden_dim,
        seq_len=config.seq_len
    ).to(config.device)

    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model.eval()
    print("✅ 模型权重加载成功！开始进行全量推理...")

    # 3. 收集推理结果
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(config.device)
            # 模型输出的是归一化后的数据
            preds = model(x_batch).cpu().numpy()
            trues = y_batch.numpy()

            all_preds.append(preds)
            all_trues.append(trues)

    all_preds = np.vstack(all_preds)
    all_trues = np.vstack(all_trues)

    # 4. 🔥 关键步骤：反归一化！将数据还原回真实的物理量纲（千瓦 kW 或 摄氏度 ℃）
    preds_real = scaler_y.inverse_transform(all_preds)
    trues_real = scaler_y.inverse_transform(all_trues)

    print("\n" + "=" * 30 + " 📊 物理性能指标报告 " + "=" * 30)

    # 5. 计算并输出各项真实物理指标 + 工程满量程百分比误差
    for i, target_name in enumerate(target_names):
        p = preds_real[:, i]
        t = trues_real[:, i]

        mae = mean_absolute_error(t, p)
        rmse = np.sqrt(mean_squared_error(t, p))
        r2 = r2_score(t, p)

        # 💡 新增：工程界常用的满量程百分比误差 (nMAE)
        val_range = np.max(t) - np.min(t)
        if val_range == 0: val_range = 1e-5  # 防止除以 0
        nMAE_percent = (mae / val_range) * 100

        unit = "kW" if "功率" in target_name else "℃"
        print(f"🎯 目标: 【{target_name}】")
        print(f"   ➤ MAE  (平均绝对误差): {mae:.4f} {unit}")
        print(f"   ➤ RMSE (均方根误差)  : {rmse:.4f} {unit}")
        print(f"   ➤ R²   (决定系数)    : {r2:.4f}")
        print(f"   ➤ F.S. (满量程误差)  : {nMAE_percent:.2f}%  <-- 业务方最爱看")
        print("-" * 60)

    # 6. 绘图：同时保存【全量测试集宏观图】与【局部500步微观图】
    def draw_comparison_plot(time_steps, true_data, pred_data, title_suffix, filename_suffix):
        fig, axes = plt.subplots(output_dim, 1, figsize=(15, 4 * output_dim), squeeze=False)
        for i, target_name in enumerate(target_names):
            ax = axes[i, 0]
            ax.plot(time_steps, true_data[:, i], label='真实值 (Ground Truth)', color='black', linewidth=1.5, alpha=0.8)
            ax.plot(time_steps, pred_data[:, i], label='模型预测 (Prediction)', color='red', linewidth=1.5,
                    linestyle='--', alpha=0.9)

            unit = "kW" if "功率" in target_name else "℃"
            ax.set_ylabel(f"{target_name} ({unit})", fontsize=12)
            ax.set_title(f"[{title_suffix}] 预测对比曲线: {target_name} (VIN: {target_vin})", fontsize=14)
            ax.legend(loc='upper right')
            ax.grid(True, linestyle='--', alpha=0.5)

        plt.xlabel("序列时间步 (Time Steps)", fontsize=12)
        plt.tight_layout()

        plot_save_path = os.path.join(test_results_dir, f"predict_{filename_suffix}_{business_module}_{target_vin}.png")
        plt.savefig(plot_save_path)
        plt.close()
        print(f"📈 {title_suffix}已保存至: {plot_save_path}")

    # 画图 A：100% 测试集全景图（看全局衰减、漂移趋势）
    draw_comparison_plot(np.arange(len(preds_real)), trues_real, preds_real, "全量时序全景图", "FULL")

    # 画图 B：截取 500 步微观图（看瞬态跟随性能，急加减速）
    plot_len = min(500, len(preds_real))
    draw_comparison_plot(np.arange(plot_len), trues_real[:plot_len], preds_real[:plot_len], "局部动态跟随放大图",
                         "ZOOM")


if __name__ == "__main__":
    run_evaluation()