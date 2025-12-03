import streamlit as st
import json

st.set_page_config(page_title="Recipe Bot", page_icon="🍳", layout="wide")

RECIPES_DATABASE = [
    {
        "name": "Egg Bhurji",
        "ingredients": ["egg", "onion", "tomato", "chili", "oil", "salt"],
        "instructions": "1. Heat oil\\n2. Sauté onions\\n3. Add tomatoes\\n4. Scramble eggs\\n5. Mix and serve",
        "prep_time": "15 min",
        "servings": "2"
    },
    {
        "name": "Masala Omelette",
        "ingredients": ["egg", "onion", "chili", "coriander", "salt"],
        "instructions": "1. Beat eggs\\n2. Add veggies\\n3. Cook in pan\\n4. Serve hot",
        "prep_time": "10 min",
        "servings": "2"
    },
    {
        "name": "Fried Rice",
        "ingredients": ["rice", "egg", "onion", "carrot", "soy sauce"],
        "instructions": "1. Cook rice\\n2. Stir fry veggies\\n3. Add rice\\n4. Mix with soy sauce",
        "prep_time": "25 min",
        "servings": "3"
    }
]

class RecipeBot:
    def __init__(self, recipes):
        self.recipes = recipes
    
    def find_recipes(self, ingredients):
        ingredients = [i.lower().strip() for i in ingredients]
        matches = []
        
        for recipe in self.recipes:
            recipe_ings = [i.lower() for i in recipe["ingredients"]]
            matched = sum(1 for ing in ingredients if any(ing in r for r in recipe_ings))
            match_pct = (matched / len(ingredients)) * 100 if ingredients else 0
            
            if match_pct > 0:
                matches.append({"recipe": recipe, "match": match_pct})
        
        matches.sort(key=lambda x: x["match"], reverse=True)
        return matches

if 'bot' not in st.session_state:
    st.session_state.bot = RecipeBot(RECIPES_DATABASE)

st.title("🍳 Recipe Chatbot")
st.markdown("Get recipe suggestions based on your ingredients!")

ingredients_input = st.text_area("Enter ingredients (comma-separated):", 
                                  placeholder="egg, onion, tomato")

if st.button("🔍 Find Recipes"):
    if ingredients_input:
        ingredients = [i.strip() for i in ingredients_input.split(',')]
        matches = st.session_state.bot.find_recipes(ingredients)
        
        if matches:
            st.success(f"Found {len(matches)} recipe(s)!")
            
            for match in matches:
                recipe = match["recipe"]
                with st.expander(f"{recipe['name']} - {match['match']:.0f}% Match"):
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.markdown("**Instructions:**")
                        st.text(recipe["instructions"])
                    with col2:
                        st.metric("Prep Time", recipe["prep_time"])
                        st.metric("Servings", recipe["servings"])
                        st.markdown("**Ingredients:**")
                        for ing in recipe["ingredients"]:
                            st.text(f"• {ing}")
        else:
            st.warning("No matches found! Try: egg, onion, rice")
