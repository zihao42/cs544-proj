import pandas as pd
import matplotlib.pyplot as plt
import os

# 日志文件路径
log_file_path = "/media/data1/ningtong/wzh/projects/Key-Sentence/lightning_logs/version_0/metrics.csv"

# 图表输出目录
output_dir = "/media/data1/ningtong/wzh/projects/Key-Sentence/charts"
os.makedirs(output_dir, exist_ok=True)

# 平滑曲线函数
def smooth_curve(values, weight=0.9):
    smoothed = []
    last = values[0]
    for val in values:
        smoothed_val = last * weight + (1 - weight) * val
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

# 生成图表
def generate_loss_chart_with_epoch_lines(log_file_path, output_dir):
    # 加载日志数据
    data = pd.read_csv(log_file_path)

    # 提取有效的训练和验证损失数据
    train_loss = data[["step", "train_loss", "epoch"]].dropna()
    val_loss = data[["epoch", "val_loss"]].dropna()

    # 平滑训练损失
    train_loss["train_loss_smoothed"] = smooth_curve(train_loss["train_loss"].tolist())

    # 获取每个 epoch 的结束 step（每个 epoch 的最后一个 step）
    epoch_ends = train_loss.groupby("epoch")["step"].max().values

    # 绘制训练损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss["step"], train_loss["train_loss_smoothed"], label="Train Loss (Smoothed)", color="blue")

    # 添加每个 epoch 的竖线
    for i, epoch_end in enumerate(epoch_ends):
        if i + 1 == 4:  # 第四个 epoch
            plt.axvline(x=epoch_end, color="green", linestyle="-", alpha=0.9, label="Early Stop")
            plt.text(epoch_end + 200, 0.05, "Early Stop", color="green", fontsize=10, va="center")
        else:
            plt.axvline(x=epoch_end, color="red", linestyle="--", alpha=0.7, label=f"Epoch End (Step {epoch_end})" if epoch_end == epoch_ends[0] else "")

    # 图表设置
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve with Epoch Boundaries")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "training_loss_curve_with_epoch_lines.png"))
    plt.close()

    # 绘制验证损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(val_loss["epoch"], val_loss["val_loss"], label="Validation Loss", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "validation_loss_curve.png"))
    plt.close()

    print(f"Charts saved in {output_dir}")

# 调用函数生成图表
generate_loss_chart_with_epoch_lines(log_file_path, output_dir)
