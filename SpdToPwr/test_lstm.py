# test_lstm.py
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

from configsP.lstm_config import LSTMConfig
from data_utils import create_training_dataset_from_shards, MODULE_FEATURES
from model_lstm import LSTMPredictor


def run_evaluation():
    config = LSTMConfig()

    target_vin = "6993"
    business_module = "power_prediction"  # "power_prediction" 或 "temp_prediction"
    shards_dir = "processedData"

    input_dim = len(MODULE_FEATURES[business_module]["features"])
    target_names = MODULE_FEATURES[business_module]["targets"]
    output_dim = len(target_names)

    model_path = os.path.join("models", f"best_model_lstm_{business_module}_{target_vin}.pth")
    test_results_dir = "test_results_lstm"
    os.makedirs(test_results_dir, exist_ok=True)

    print(f"\n🔍 启动 LSTM 模型测试评估引擎 | 车辆: {target_vin} | 业务: {business_module}")
    print("-" * 70)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ 找不到训练好的模型权重: {model_path}，请先运行 train_lstm.py！")

    print("⏳ 正在组装测试集数据...")
    _, _, test_loader, scaler_y = create_training_dataset_from_shards(
        data_dir=shards_dir,
        vin_id=target_vin,
        module=business_module,
        seq_len=config.seq_len,
        batch_size=config.batch_size
    )

    model = LSTMPredictor(
        input_dim=input_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        output_dim=output_dim
    ).to(config.device)

    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model.eval()
    print("✅ LSTM 模型权重加载成功！开始进行全量推理...")

    all_preds = []
    all_trues = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(config.device)
            preds = model(x_batch).cpu().numpy()
            trues = y_batch.numpy()

            all_preds.append(preds)
            all_trues.append(trues)

    all_preds = np.vstack(all_preds)
    all_trues = np.vstack(all_trues)

    preds_real = scaler_y.inverse_transform(all_preds)
    trues_real = scaler_y.inverse_transform(all_trues)

    print("\n" + "=" * 30 + " 📊 LSTM 物理性能指标报告 " + "=" * 30)

    for i, target_name in enumerate(target_names):
        p = preds_real[:, i]
        t = trues_real[:, i]

        mae = mean_absolute_error(t, p)
        rmse = np.sqrt(mean_squared_error(t, p))
        r2 = r2_score(t, p)

        val_range = np.max(t) - np.min(t)
        if val_range == 0:
            val_range = 1e-5
        nMAE_percent = (mae / val_range) * 100

        unit = "kW" if "功率" in target_name else "℃"
        print(f"🎯 目标: 【{target_name}】")
        print(f"   ➤ MAE  (平均绝对误差): {mae:.4f} {unit}")
        print(f"   ➤ RMSE (均方根误差)  : {rmse:.4f} {unit}")
        print(f"   ➤ R²   (决定系数)    : {r2:.4f}")
        print(f"   ➤ F.S. (满量程误差)  : {nMAE_percent:.2f}%")
        print("-" * 60)

    def draw_comparison_plot(time_steps, true_data, pred_data, title_suffix, filename_suffix):
        fig, axes = plt.subplots(output_dim, 1, figsize=(15, 4 * output_dim), squeeze=False)
        for i, target_name in enumerate(target_names):
            ax = axes[i, 0]
            ax.plot(time_steps, true_data[:, i], label='真实值 (Ground Truth)', color='black', linewidth=1.5, alpha=0.8)
            ax.plot(time_steps, pred_data[:, i], label='LSTM 预测 (Prediction)', color='red', linewidth=1.5,
                    linestyle='--', alpha=0.9)

            unit = "kW" if "功率" in target_name else "℃"
            ax.set_ylabel(f"{target_name} ({unit})", fontsize=12)
            ax.set_title(f"[LSTM] [{title_suffix}] 预测对比曲线: {target_name} (VIN: {target_vin})", fontsize=14)
            ax.legend(loc='upper right')
            ax.grid(True, linestyle='--', alpha=0.5)

        plt.xlabel("序列时间步 (Time Steps)", fontsize=12)
        plt.tight_layout()

        plot_save_path = os.path.join(test_results_dir,
                                      f"predict_{filename_suffix}_{business_module}_{target_vin}.png")
        plt.savefig(plot_save_path)
        plt.close()
        print(f"📈 {title_suffix}已保存至: {plot_save_path}")

    # 全量测试集全景图
    draw_comparison_plot(np.arange(len(preds_real)), trues_real, preds_real, "全量时序全景图", "FULL")

    # 局部 500 步微观图
    plot_len = min(500, len(preds_real))
    draw_comparison_plot(np.arange(plot_len), trues_real[:plot_len], preds_real[:plot_len], "局部动态跟随放大图",
                         "ZOOM")


if __name__ == "__main__":
    run_evaluation()
