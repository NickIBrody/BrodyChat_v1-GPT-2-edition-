import torch
from transformers import GPT2LMHeadModel, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset

# Загрузка модели
model_name = "gpt2-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name)

# Загрузка данных из файла
with open("data/brody_train.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]

dataset = Dataset.from_dict({"text": lines})
tokenized_ds = dataset.map(lambda x: tokenizer(x["text"], truncation=True, max_length=128), batched=True)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Настройки обучения v2.0
training_args = TrainingArguments(
    output_dir="./BrodyChat_v2",
    num_train_epochs=30,
    per_device_train_batch_size=2,
    logging_steps=10,
    save_strategy="no",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_ds,
)

trainer.train()
model.save_pretrained("./BrodyChat_Final")
tokenizer.save_pretrained("./BrodyChat_Final")
