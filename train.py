import os
import shutil
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# 1. Create Dummy Data
data = [
    "Ingredients: Egg, Onion, Oil. Recipe: Scramble eggs with fried onions and salt.",
    "Ingredients: Rice, Tomato, Salt. Recipe: Cook rice with tomato paste and spices.",
    "Ingredients: Milk, Sugar, Tea. Recipe: Boil milk with tea leaves and sugar.",
    "Ingredients: Bread, Butter. Recipe: Toast bread and spread butter evenly.",
    "Ingredients: Chicken, Spice, Oil. Recipe: Fry chicken in oil with mixed spices."
]
with open("recipes.txt", "w") as f:
    f.write("\n".join(data))

# 2. Load Model
model_name = "distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

# 3. Prepare Dataset
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="recipes.txt",
    block_size=128
)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 4. Train
# WE SAVE TO /content/temp_model (Local VM) to avoid Drive lag
training_args = TrainingArguments(
    output_dir="/content/temp_results", 
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500, # Don't save intermediate steps to drive
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

print("🚀 Starting Fine-Tuning...")
trainer.train()

# 5. Save Locally First (Fast)
print("💾 Saving model locally...")
model.save_pretrained("/content/temp_recipe_model")
tokenizer.save_pretrained("/content/temp_recipe_model")

# 6. Move to Drive (Explicit copy)
print("📂 Moving to Google Drive folder...")
target_dir = "./recipe_model"
if os.path.exists(target_dir):
    shutil.rmtree(target_dir)
shutil.copytree("/content/temp_recipe_model", target_dir)

print("✅ DONE! Model is ready.")
