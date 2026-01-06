import os
from transformers import (
    GPT2Tokenizer, 
    GPT2LMHeadModel, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

# 1. Create Better Training Data
base_data = [
    "Ingredients: Egg, Onion, Oil. Recipe: Heat oil in a pan, fry chopped onions until golden, add beaten eggs and scramble until cooked.",
    "Ingredients: Egg, Onion. Recipe: Whisk eggs with salt, cook diced onions in butter, pour eggs over onions and scramble gently.",
    "Ingredients: Rice, Tomato, Salt. Recipe: Cook rice in salted water, add chopped tomatoes and simmer until rice is tender and fluffy.",
    "Ingredients: Rice, Tomato. Recipe: Boil rice until half done, stir in tomato paste and cook covered until rice absorbs all flavors.",
    "Ingredients: Milk, Sugar, Tea. Recipe: Boil milk in a pot, add tea leaves and sugar, simmer for 5 minutes and strain into cups.",
    "Ingredients: Milk, Tea. Recipe: Brew strong tea in boiling water, add hot milk and sweeten to taste with sugar or honey.",
    "Ingredients: Bread, Butter. Recipe: Toast bread slices until golden brown, spread softened butter evenly while still warm.",
    "Ingredients: Bread. Recipe: Slice bread and toast in a toaster or pan until crispy and lightly browned on both sides.",
    "Ingredients: Chicken, Spice, Oil. Recipe: Marinate chicken pieces in mixed spices, heat oil and fry until golden and cooked through.",
    "Ingredients: Chicken, Spice. Recipe: Rub chicken with spice blend, bake at 180C for 40 minutes until tender and juicy.",
    "Ingredients: Egg, Oil. Recipe: Heat oil in a pan, crack eggs directly into hot oil and fry until whites are set and yolks are runny.",
    "Ingredients: Onion, Oil. Recipe: Slice onions thinly, fry in hot oil until caramelized and deep golden brown in color.",
    "Ingredients: Rice, Salt. Recipe: Rinse rice thoroughly, cook in salted boiling water until grains are separate and fluffy.",
    "Ingredients: Chicken, Oil. Recipe: Cut chicken into pieces, shallow fry in hot oil until skin is crispy and meat is cooked.",
    "Ingredients: Milk, Sugar. Recipe: Heat milk gently, stir in sugar until dissolved, serve warm or chilled as a sweet drink.",
]

# Expand dataset - repeat 40 times for better training
data = base_data * 40

# Write to file
print("📝 Creating training data file...")
with open("recipes.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(data))

print(f"✅ Created {len(data)} training examples")

# 2. Load Model and Tokenizer
print("🤖 Loading model...")
model_name = "distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set pad token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# 3. Prepare Dataset using the new method (not deprecated TextDataset)
print("📊 Preparing dataset...")

def tokenize_function(examples):
    # Tokenize the text
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
        padding="max_length",
        return_tensors=None
    )
    # Clone input_ids to labels for language modeling
    result["labels"] = result["input_ids"].copy()
    return result

# Read the file and create dataset
with open("recipes.txt", "r", encoding="utf-8") as f:
    texts = [line.strip() for line in f if line.strip()]

# Create a dataset from the texts
dataset = Dataset.from_dict({"text": texts})

# Tokenize the dataset
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

print(f"✅ Dataset prepared with {len(tokenized_dataset)} samples")

# 4. Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False
)

# 5. Training Arguments
training_args = TrainingArguments(
    output_dir="./temp_results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    logging_steps=10,
    learning_rate=5e-5,
    warmup_steps=100,
    weight_decay=0.01,
)

# 6. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
)

# 7. Train
print("🚀 Starting Fine-Tuning...")
trainer.train()

# 8. Save Model
print("💾 Saving model...")
target_dir = "./recipe_model"

model.save_pretrained(target_dir)
tokenizer.save_pretrained(target_dir)

print(f"✅ DONE! Model is ready at: {target_dir}")
