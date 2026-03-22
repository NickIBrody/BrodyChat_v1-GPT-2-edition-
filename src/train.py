import torch
from transformers import AutoTokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

#  GPT-2
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Путь к файлу с данными
train_path = "data/brody_train.txt"

def load_dataset(path, tokenizer):
    return TextDataset(tokenizer=tokenizer, file_path=path, block_size=128)

train_dataset = load_dataset(train_path, tokenizer)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Настройки обучения
training_args = TrainingArguments(
    output_dir="./BrodyChat_Model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)


trainer.train()
model.save_pretrained("./BrodyChat_Final")
tokenizer.save_pretrained("./BrodyChat_Final")
print("Обучение завершено!")
