import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import PeftModel

# ============================================================
# CONFIG
# ============================================================

BASE_MODEL = "SicariusSicariiStuff/Impish_Mind_8B"

# Stage 1 adapter
OLD_ADAPTER = "./impish-hindi-test"

# Stage 2 output
OUTPUT_DIR = "./impish-hindi-chat"

DATA_FILE = "train.jsonl"

MAX_LENGTH = 2048


# ============================================================
# TOKENIZER
# ============================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(OLD_ADAPTER)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer vocabulary:", len(tokenizer))


# ============================================================
# DATASET
# ============================================================

print("\nLoading conversation dataset...")

dataset = load_dataset(
    "json",
    data_files=DATA_FILE,
    split="train",
)

print("Training conversations:", len(dataset))


# ============================================================
# FORMAT CONVERSATIONS
# ============================================================

def format_chat(example):
    messages = example["messages"]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return {"text": text}


print("Applying chat template...")

dataset = dataset.map(
    format_chat,
    remove_columns=dataset.column_names,
)


# ============================================================
# TOKENIZATION
# ============================================================

print("Tokenizing conversations...")


def tokenize(example):
    result = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )

    result["labels"] = result["input_ids"].copy()

    return result


dataset = dataset.map(
    tokenize,
    remove_columns=["text"],
    desc="Tokenizing",
)


# Remove examples that somehow became empty
dataset = dataset.filter(
    lambda x: len(x["input_ids"]) > 0
)

print("Final training examples:", len(dataset))


# ============================================================
# 4-BIT CONFIG
# ============================================================

print("\nLoading base model in 4-bit...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


# ============================================================
# BASE MODEL
# ============================================================

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.bfloat16,
    quantization_config=bnb_config,
    device_map="auto",
)

model.config.use_cache = False


# ============================================================
# LOAD STAGE 1 LORA
# ============================================================

print("\nLoading Stage 1 LoRA adapter...")

model = PeftModel.from_pretrained(
    model,
    OLD_ADAPTER,
    is_trainable=True,
)

model.print_trainable_parameters()


# ============================================================
# TRAINING
# ============================================================

print("\nStarting conversation fine-tuning...\n")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    num_train_epochs=3,

    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,

    learning_rate=1e-5,

    bf16=True,

    logging_steps=1,

    save_strategy="epoch",

    optim="paged_adamw_8bit",

    lr_scheduler_type="cosine",
   

    weight_decay=0.01,

    gradient_checkpointing=True,

    report_to="none",

    remove_unused_columns=False,
)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)


trainer.train()


# ============================================================
# SAVE
# ============================================================

print("\nSaving conversation LoRA adapter...")

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\nTraining complete!")
print("Saved to:", OUTPUT_DIR)
