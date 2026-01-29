from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from llm.llm_init import load_base_model
from llm.lora_setup import apply_lora

model, tokenizer = load_base_model()
model = apply_lora(model)

dataset = load_dataset("json", data_files="llm_training_data.json")

def tokenize(batch):
    text = batch["instruction"] + "\n" + batch["input"] + "\n" + batch["output"]
    return tokenizer(text, truncation=True, padding="max_length", max_length=512)

dataset = dataset.map(tokenize, batched=True)

training_args = TrainingArguments(
    output_dir="./lora_output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"]
)

trainer.train()
model.save_pretrained("./lora_output")
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "./lora_output",
    device_map="auto",
    load_in_4bit=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
reward = (
    env_reward
    + 0.2 * llm_coherence_score
    + 0.2 * genre_consistency_score
)
