

import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pytorch_lightning as pl

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

    # 加载保存的检查点
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    state_dict = checkpoint.get("state_dict", checkpoint)

    # 初始化模型
    model = KeySentenceModel()
    model.load_state_dict(state_dict, strict=True)  # 严格加载检查点中的所有权重
    print("Model loaded successfully with state dict keys:")
    print(model.state_dict().keys())
    return model

# 分句函数
def split_sentence(text):
    sentences = text.replace('\n', '').replace('\xa0', ' ').split('.')
    return [sent.strip() + '.' for sent in sentences if sent.strip()]

# 预测函数
def predict_key_sentences(model, tokenizer, case_text):
    """
    对一个案例（case_text）进行预测，返回每个句子的分数。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    sentences = split_sentence(case_text)
    results = []

    for sent in sentences:
        encoding = tokenizer(
            sent,
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
        results.append((sent, score))

    return results

# 主函数
if __name__ == "__main__":
    # 路径设置
    model_path = "/media/data1/ningtong/wzh/projects/Key-Sentence/checkpoints/key-sentence-epoch=08-val_loss=0.02.ckpt"  # 根据你的保存路径调整
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # 加载模型
    model = load_model_with_checkpoints(model_path)

    # 示例输入
    test_case = (
        "Alexei Navalny, speaker of Russian. Anna Politkovskaya spoke the language Russian. References\n\nCategory:Neighbourhoods in Dhanbad The native language of Frantz Fanon is Russian. Later renovation is attributed to the architect Lucínio Cruz. Frantz Fanon spoke the language Russian. Louis Antoine de Saint-Just, speaker of French. Montesquieu, speaker of French. The mother tongue of Frantz Fanon is."
    )

    # 调用预测函数
    predictions = predict_key_sentences(model, tokenizer, test_case)

    # 打印结果
    print("Predicted Key Sentences with Scores:")
    for sent, score in predictions:
        print(f"{sent} ({score:.2f})")
