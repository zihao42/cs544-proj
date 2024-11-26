import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from sklearn.model_selection import train_test_split

# 分句函数
def split_sentence(text):
    sentences = text.replace('\n', '').replace('\xa0', ' ').split('.')
    return [sent.strip() + '.' for sent in sentences if sent.strip()]

# 自定义数据集
class KeySentenceDataset(Dataset):
    def __init__(self, cases, tokenizer, max_len=512):
        self.cases = cases
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]
        sentences = [s["sentence"] for s in case["sentences"]]
        labels = [s["label"] for s in case["sentences"]]

        encoding = self.tokenizer(
            sentences,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"],  # [num_sentences, seq_len]
            "attention_mask": encoding["attention_mask"],  # [num_sentences, seq_len]
            "labels": torch.tensor(labels, dtype=torch.float),  # [num_sentences]
        }

# 模型定义
class KeySentenceModel(pl.LightningModule):
    def __init__(self, model_name="bert-base-uncased", lr=2e-5):
        super().__init__()
        self.save_hyperparameters()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, 1)
        self.lr = lr
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask):
        # 展平张量
        batch_size, num_sentences, seq_len = input_ids.shape
        input_ids = input_ids.view(-1, seq_len)  # [batch_size * num_sentences, seq_len]
        attention_mask = attention_mask.view(-1, seq_len)  # [batch_size * num_sentences, seq_len]

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.pooler_output  # [batch_size * num_sentences, hidden_size]

        logits = self.classifier(hidden_states).squeeze(-1)  # [batch_size * num_sentences]
        logits = logits.view(batch_size, num_sentences)  # 恢复到 [batch_size, num_sentences]
        return logits

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        logits = self(input_ids, attention_mask)
        loss = self.loss_fn(logits, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        logits = self(input_ids, attention_mask)
        loss = self.loss_fn(logits, labels)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

# 数据预处理
def preprocess_data(input_path, output_train, output_test):
    with open(input_path, "r") as f:
        data = json.load(f)

    cases = []
    for item in data:
        sentences = split_sentence(item["sentence"])
        key_sentences = [
            item.get("key_sentence1", ""),
            item.get("key_sentence2", ""),
            item.get("key_sentence3", ""),
        ]
        labels = [1 if s in key_sentences else 0 for s in sentences]
        cases.append({"sentences": [{"sentence": s, "label": l} for s, l in zip(sentences, labels)]})

    train_cases, test_cases = train_test_split(cases, test_size=0.2, random_state=42)
    with open(output_train, "w") as f:
        json.dump(train_cases, f)
    with open(output_test, "w") as f:
        json.dump(test_cases, f)

# 主函数
def main(gpus):
    train_data_path = "train_data.json"
    test_data_path = "test_data.json"
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_dataset = KeySentenceDataset(
        cases=json.load(open(train_data_path, "r")), tokenizer=tokenizer
    )
    val_dataset = KeySentenceDataset(
        cases=json.load(open(test_data_path, "r")), tokenizer=tokenizer
    )

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)

    model = KeySentenceModel()

    # 添加 ModelCheckpoint 回调，保存每个 epoch 的权重
    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints",  # 保存权重的目录
        filename="key-sentence-{epoch:02d}-{val_loss:.2f}",  # 文件名模板
        save_top_k=-1,  # 保存所有 epoch
        mode="min",  # 监控 val_loss，选择较低值为最佳
        monitor="val_loss",  # 监控指标
        save_last=True,  # 保存最新的检查点
    )

    # 添加 EarlyStopping 和 LearningRateMonitor 回调
    lr_monitor = LearningRateMonitor(logging_interval="step")
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min",
    )

    # 定义 Trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=gpus,
        max_epochs=10,
        callbacks=[checkpoint_callback, lr_monitor, early_stop_callback],
        log_every_n_steps=10,
    )

    # 开始训练
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    preprocess_data("multi_counterfact_4distractors_new.json", "train_data.json", "test_data.json")
    # 指定 GPU，例如：[0] 表示使用 0 号 GPU；[0, 1] 表示使用 0 和 1 号 GPU
    main(gpus=[5, 6, 7])
