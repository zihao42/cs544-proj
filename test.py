import os
import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pytorch_lightning as pl
from sklearn.metrics import precision_score, recall_score, accuracy_score

# 设置显卡（如果有多个显卡，设置 CUDA_VISIBLE_DEVICES 或 torch.cuda.set_device）
os.environ["CUDA_VISIBLE_DEVICES"] = "5, 6, 7"  # 使用第0块显卡（根据需求修改）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义模型
class KeySentenceModel(pl.LightningModule):
    def __init__(self, model_name="bert-base-uncased", lr=2e-5):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1)
        self.lr = lr
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

# 检查和加载权重函数
def load_model_with_checkpoints(model_path):
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)

    model = KeySentenceModel()
    model.load_state_dict(state_dict, strict=True)
    print("Model loaded successfully.")
    return model

# 加载测试数据
def load_test_data(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

# 预测函数
def predict_label(model, tokenizer, sentence, threshold=0.65):
    model.eval()

    encoding = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        score = torch.sigmoid(logits).squeeze().item()

    predicted_label = 1 if score >= threshold else 0
    return predicted_label, score

# 测试函数
def test_model(test_data_path, model_path, threshold=0.65):
    # 加载模型和分词器
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = load_model_with_checkpoints(model_path).to(device)

    # 加载测试数据
    test_data = load_test_data(test_data_path)

    # 初始化统计变量
    y_true = []
    y_pred = []

    print("\nStarting the test process...\n")

    # 逐条测试
    for i, entry in enumerate(test_data, 1):
        sentence = entry["sentence"]
        true_label = entry["label"]

        # 获取预测结果
        pred_label, score = predict_label(model, tokenizer, sentence, threshold)

        # 保存结果
        y_true.append(true_label)
        y_pred.append(pred_label)

        # 计算当前指标
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)

        # 实时输出结果
        print(f"Sentence {i}:")
        print(f"  Content         : {sentence}")
        print(f"  True Label      : {true_label}")
        print(f"  Predicted Label : {pred_label} (Score: {score:.2f})")
        print(f"  Current Accuracy: {accuracy:.2f}")
        print(f"  Current Precision: {precision:.2f}")
        print(f"  Current Recall   : {recall:.2f}")
        print("-" * 50, flush=True)  # 确保即时输出

    print("\nTest complete. Final Results:")
    print(f"  Final Accuracy  : {accuracy:.2f}")
    print(f"  Final Precision : {precision:.2f}")
    print(f"  Final Recall    : {recall:.2f}")

if __name__ == "__main__":
    # 配置路径
    test_data_path = "/media/data1/ningtong/wzh/projects/Key-Sentence/test_data.json"
    model_path = "/media/data1/ningtong/wzh/projects/Key-Sentence/checkpoints/key-sentence-epoch=08-val_loss=0.02.ckpt"

    # 开始测试
    test_model(test_data_path, model_path)