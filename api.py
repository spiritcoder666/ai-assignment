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
    
    # Generate output
    outputs = model.generate(
        inputs, 
        max_length=100, 
        num_return_sequences=1, 
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": text}
