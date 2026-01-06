from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = FastAPI()

# 1. Load our Fine-Tuned Model
model_path = "./recipe_model"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# 2. Define Input Format
class IngredientsInput(BaseModel):
    ingredients: str

# 3. Define the Endpoint
@app.post("/generate_recipe")
def generate_recipe(data: IngredientsInput):
    prompt = f"Ingredients: {data.ingredients}. Recipe:"
    
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    # Improved generation parameters
    outputs = model.generate(
        inputs, 
        max_length=80,           # Reduced from 100 to prevent long outputs
        min_length=30,           # Ensure minimum length
        num_return_sequences=1, 
        no_repeat_ngram_size=3,  # Increased to prevent more repetition
        do_sample=True,          # Enable sampling for variety
        top_k=50,                # Limit token choices
        top_p=0.95,              # Nucleus sampling
        temperature=0.7,         # Control randomness
        repetition_penalty=1.2,  # Penalize repetition
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the recipe part (stop at first period after recipe starts)
    if "Recipe:" in text:
        recipe_part = text.split("Recipe:", 1)[1]
        # Get first sentence or up to next "Ingredients:" if it appears
        if "Ingredients:" in recipe_part:
            recipe_part = recipe_part.split("Ingredients:")[0]
        text = f"Ingredients: {data.ingredients}. Recipe:{recipe_part.strip()}"
    
    return {"response": text}
