import os
from transformers import (
    GPT2Tokenizer, 
    GPT2LMHeadModel, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

# 1. MINIMAL Training Data (just enough to learn the pattern)
base_data = [
    "Ingredients: Egg, Onion, Oil. Recipe: Heat oil, fry onions, add eggs and scramble.",
    "Ingredients: Egg, Onion. Recipe: Cook onions in butter, pour eggs and scramble.",
    "Ingredients: Rice, Tomato, Salt. Recipe: Cook rice with chopped tomatoes and salt.",
    "Ingredients: Rice, Tomato. Recipe: Boil rice, add tomato paste and simmer.",
    "Ingredients: Milk, Sugar, Tea. Recipe: Boil milk, add tea leaves and sugar.",
    "Ingredients: Bread, Butter. Recipe: Toast bread and spread butter evenly.",
    "Ingredients: Chicken, Spice, Oil. Recipe: Fry chicken in oil with mixed spices.",
]

# Only repeat 15 times (instead of 40) = 105 samples total
data = base_data * 15

# Write to file
print("📝 Creating training data...")
with open("recipes.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(data))

print(f"✅ Created {len(data)} training examples")

# 2. Load Model
print("🤖 Loading model...")
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
model = GPT2LMHeadModel.from_pretrained("distilgpt2")

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# 3. Prepare Dataset
print("📊 Preparing dataset...")

def tokenize_function(examples):
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=64,  # Reduced from 128 to 64
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

with open("recipes.txt", "r", encoding="utf-8") as f:
    texts = [line.strip() for line in f if line.strip()]

dataset = Dataset.from_dict({"text": texts})
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

print(f"✅ Dataset ready: {len(tokenized_dataset)} samples")

# 4. Data Collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 5. ULTRA-LIGHT Training Config
training_args = TrainingArguments(
    output_dir="./temp_results",
    overwrite_output_dir=True,
    num_train_epochs=2,              # Only 2 epochs (was 3)
    per_device_train_batch_size=8,   # Larger batches = faster (was 4)
    save_steps=1000,                 # Save less often
    save_total_limit=1,              # Keep only 1 checkpoint
    logging_steps=50,                # Log less often
    learning_rate=5e-5,
    warmup_steps=50,                 # Reduced warmup
    fp16=False,                      # Disable mixed precision (can cause issues)
    dataloader_num_workers=0,        # No parallel workers
)

# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
)

# 7. Train
print("🚀 Starting Fast Training (2-3 minutes)...")
trainer.train()

# 8. Save
print("💾 Saving model...")
model.save_pretrained("./recipe_model")
tokenizer.save_pretrained("./recipe_model")

print("✅ DONE! Model ready at: ./recipe_model")
