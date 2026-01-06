import os
import shutil
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# 1. Create Dummy Data
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

# Expand dataset - repeat 30 times for better training
data = base_data * 30
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
