# 🤖 AI Assignment: Name Matcher & Local LLM Recipe Bot

## 📌 Project Overview
This solution implements two AI modules as per the assignment requirements:

1.  **Task 1: Name Matcher**
    * **Goal:** Find similar names from a dataset.
    * **Tech:** **FuzzyWuzzy** (Levenshtein Distance) and **FAISS** (Vector Search).
2.  **Task 2: Recipe Chatbot**
    * **Goal:** Generate recipes from ingredients using a custom AI model.
    * **Tech:** **DistilGPT-2** (Fine-Tuned), **FastAPI** (Backend), **Streamlit** (Frontend).

---

## 🌐 Live Demo
You can access the live frontend deployment here:
👉 **[Streamlit App Live Demo](https://ai-assignment-bqofuapcd5cqrxrnqtnozr.streamlit.app/)**

*(Note: Task 1 works fully on the cloud. Task 2 requires the Local API Server and will show a connection message if accessed via this cloud link.)*

---

## ⚡ Quick Start: Run on Google Colab (Recommended)
Since Task 2 involves a Local LLM and a Backend API, the easiest way to test the full project is via Google Colab.

**Step 1:** Open a new [Google Colab Notebook](https://colab.research.google.com/).
**Step 2:** Copy and paste the following code block into a cell and run it.

```python
# --- MASTER RUN SCRIPT FOR COLAB ---
import os
import subprocess
import time

# 1. Clone the Repository
if not os.path.exists("ai-assignment"):
    !git clone [https://github.com/spiritcoder666/ai-assignment.git](https://github.com/spiritcoder666/ai-assignment.git)
os.chdir("ai-assignment")

# 2. Install Dependencies
print("⏳ Installing Dependencies...")
!pip install -r requirements.txt -q
!npm install -g localtunnel -q

# 3. Fine-Tune the Model (Task 2)
print("🧠 Fine-Tuning Model (This takes ~2 mins)...")
!python train.py

# 4. Start Backend (FastAPI) & Frontend (Streamlit)
print("🚀 Starting Servers...")
# Kill old processes
!pkill uvicorn
!pkill streamlit

# Start API in background
subprocess.Popen(["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"])
time.sleep(10) # Wait for API to start

# Start Streamlit in background
subprocess.Popen(["streamlit", "run", "app.py"])

# 5. Create Public Tunnel
print("\n✅ APP IS RUNNING!")
print("1. Copy this IP address:")
!curl ipv4.icanhazip.com
print("\n2. Click the link below and paste the IP:")
!npx localtunnel --port 8501
```

---

## 🛠️ Local Installation (Windows/Linux)
If you prefer running it locally, follow these steps:

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/spiritcoder666/ai-assignment.git](https://github.com/spiritcoder666/ai-assignment.git)
    cd ai-assignment
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 How to Run Locally

### Task 1: Name Matcher
Run the main application:
```bash
streamlit run app.py
```
*Navigate to "Name Matcher" in the sidebar to test.*

### Task 2: Recipe Bot (Local LLM)
This task requires a **Local API Server** to be running.

**Step 1: Fine-Tune the Model**
Train the AI model on the recipe dataset (creates `./recipe_model` folder):
```bash
python train.py
```

**Step 2: Start the Backend (API)**
Open a terminal and run the FastAPI server:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

**Step 3: Start the Frontend (UI)**
Open a **new** terminal and run:
```bash
streamlit run app.py
```

---

## 🧪 Verification & Sample Outputs

| Task | Input | Expected Output | Logic |
| :--- | :--- | :--- | :--- |
| **Name Matcher** | `Geeta` | **Best:** `Geetha` (Score: 90+) <br> **Matches:** `Gita`, `Geetu` | Fuzzy Logic + Vector Embeddings |
| **Recipe Bot** | `Egg, Onion` | "Scramble eggs with fried onions and salt..." | Fine-Tuned LLM Inference |

---

## ⚠️ Important Note for Cloud Deployment
**Task 2 (Recipe Bot)** relies on a **Local API Server** (`localhost:8000`).
* **On Local Machine / Colab:** The feature works fully.
* **On Streamlit Cloud:** You will see a **"Connection Failed"** message. This is expected behavior because the public cloud server cannot access the Local API running on your private machine.

## 📂 Project Structure
* `app.py`: Frontend Entry Point.
* `api.py`: FastAPI Backend for Model Inference.
* `train.py`: Training Script for Fine-Tuning.
* `pages/`: Application Modules.
