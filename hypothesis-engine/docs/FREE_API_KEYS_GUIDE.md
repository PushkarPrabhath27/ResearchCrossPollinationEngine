# 🔑 Complete FREE API Keys Setup Guide

## Overview: What API Keys Do You Need?

Good news! **Almost everything is FREE!** Here's the complete breakdown:

| Service | Cost | API Key Required? | What It Does |
|---------|------|-------------------|--------------|
| **Ollama** | ✅ FREE | No | AI brain (runs locally on your computer) |
| **arXiv** | ✅ FREE | No | Computer science & physics papers |
| **PubMed/NCBI** | ✅ FREE | Just email | Medical & biology papers |
| **Semantic Scholar** | ✅ FREE | Yes (free) | 200M+ papers with citations |
| **OpenAlex** | ✅ FREE | No | 250M+ papers, completely open |
| **OpenAI** | 💰 Paid | Yes | Alternative AI (if you want faster) |

**Recommended FREE Setup: Use Ollama** (no API key needed, runs on your computer)

---

## 📋 STEP-BY-STEP: Getting Each API Key

---

### 🧠 OPTION A: Ollama (Recommended - 100% FREE)

Ollama lets you run AI models locally on your computer. It's completely free and private!

#### Step 1: Download Ollama

1. Open your browser
2. Go to: **https://ollama.com/download**
3. Click **"Download for Windows"**
4. Wait for download (~100MB)

#### Step 2: Install Ollama

1. Find the downloaded file: `OllamaSetup.exe`
2. Double-click to install  
3. Follow the installation wizard (just click Next)
4. Wait for installation to complete

#### Step 3: Download an AI Model

1. Open **Command Prompt** or **PowerShell**
2. Type this command and press Enter:

```powershell
ollama pull llama2
```

3. Wait for the download (~4GB) - this takes 10-30 minutes depending on internet speed
4. You'll see progress like:
```
pulling manifest 
pulling 8934d96d3f08... 100% |████████████████| 3.8 GB
```

#### Step 4: Verify Ollama is Working

In Command Prompt, type:
```powershell
ollama run llama2 "Hello, are you working?"
```

You should get a response from the AI. That means it's working!



#### Step 5: Configuration for Project

In your `.env` file, set:
```
LLM_PROVIDER=ollama
LLM_MODEL=llama2
```

**That's it! No API key needed!**

---

### 💳 OPTION B: OpenAI (Paid, but Faster)

If you want faster responses and don't mind paying ~$5-10 for testing:

#### Step 1: Create OpenAI Account

1. Go to: **https://platform.openai.com/signup**
2. Click **"Create account"**
3. Sign up with Google, Microsoft, or Email
4. Verify your email if required

#### Step 2: Add Payment Method (Required for API)

1. After logging in, go to: **https://platform.openai.com/account/billing**
2. Click **"Add payment method"**
3. Add a credit/debit card
4. Set a spending limit (recommend $10 max to start)

#### Step 3: Get Your API Key

1. Go to: **https://platform.openai.com/api-keys**
2. Click **"Create new secret key"**
3. Give it a name like "hypothesis-engine"
4. Click **"Create secret key"**
5. **IMPORTANT: Copy the key immediately!** (it starts with `sk-`)
6. You won't be able to see it again!

#### Step 4: Configuration for Project

In your `.env` file, set:
```
OPENAI_API_KEY=sk-your-actual-key-here
LLM_PROVIDER=openai
LLM_MODEL=gpt-4-turbo-preview
```

---

### 📚 Semantic Scholar API (FREE)

Semantic Scholar gives access to 200+ million research papers.

#### Step 1: Request Free API Key

1. Go to: **https://www.semanticscholar.org/product/api**
2. Scroll down and click **"Request API Key"** or **"Get Started"**
3. Fill out the form:
   - **Name**: Your name
   - **Email**: Your email
   - **Organization**: "Independent Researcher" or your university
   - **Use Case**: "Research hypothesis generation tool"
4. Click Submit

#### Step 2: Wait for Email

- You'll receive an email within 1-2 business days
- The email will contain your API key

#### Step 3: Configuration for Project

In your `.env` file, add:
```
SEMANTIC_SCHOLAR_API_KEY=your-semantic-scholar-key
```

**Note:** You can skip this initially - the project works without it, just with rate limits.

---

### 🏥 PubMed/NCBI (FREE - Just Email Required)

PubMed is free to use. They just need an email for identification.

#### Step 1: Get API Key (Optional but Recommended)

1. Go to: **https://www.ncbi.nlm.nih.gov/account/**
2. Click **"Log in"** or **"Register"**
3. Create a free NCBI account
4. After logging in, go to: **https://www.ncbi.nlm.nih.gov/account/settings/**
5. Scroll to **"API Key Management"**
6. Click **"Create an API Key"**
7. Copy the key

#### Step 2: Configuration for Project

In your `.env` file, add:
```
ENTREZ_EMAIL=your-actual-email@gmail.com
NCBI_API_KEY=your-ncbi-api-key-here
```

**Note:** The email is required, but the API key is optional (just gives higher rate limits).

---

### 📖 OpenAlex (FREE - No Key Needed!)

OpenAlex is completely free and open. No registration required!

#### Configuration

Just add your email for "polite" pool (higher rate limits):

In your `.env` file, add:
```
OPENALEX_EMAIL=your-actual-email@gmail.com
```

That's it! No API key needed!

---

### 📄 arXiv (FREE - No Key Needed!)

arXiv is completely open. No API key required.

No configuration needed!

---

## 📁 YOUR .env FILE: Complete Configuration

Now let's set up your `.env` file with everything:

### Step 1: Create the .env File

1. Open File Explorer
2. Navigate to: `C:\Users\pushk\OneDrive\Documents\LangchainProject\hypothesis-engine`
3. Find the file `.env.example`
4. Right-click → Copy
5. Right-click in empty space → Paste
6. Rename the copy from `.env.example - Copy` to `.env`

### Step 2: Edit the .env File

Right-click on `.env` → Open with → Notepad

Replace the contents with this (fill in your actual values):

```env
# ============================================
# SCIENTIFIC HYPOTHESIS ENGINE CONFIGURATION
# ============================================

# ----- LLM CONFIGURATION -----
# Choose ONE option:

# OPTION 1: Ollama (FREE - Recommended)
LLM_PROVIDER=ollama
LLM_MODEL=llama2
# Leave OPENAI_API_KEY empty if using Ollama

# OPTION 2: OpenAI (Paid - Faster)
# LLM_PROVIDER=openai
# LLM_MODEL=gpt-4-turbo-preview
OPENAI_API_KEY=

# ----- DATABASE CONFIGURATION -----
CHROMA_PERSIST_DIR=./data/embeddings
METADATA_DB_PATH=./data/metadata.db

# PostgreSQL (Docker will handle this)
POSTGRES_USER=hypothesis_user
POSTGRES_PASSWORD=hypothesis_password
POSTGRES_DB=hypothesis_engine
DATABASE_URL=postgresql://hypothesis_user:hypothesis_password@localhost:5432/hypothesis_engine

# ----- API CONFIGURATION -----
API_HOST=0.0.0.0
API_PORT=8000

# ----- RESEARCH API KEYS -----

# PubMed/NCBI (Required: your email, Optional: API key)
ENTREZ_EMAIL=YOUR_EMAIL@gmail.com
NCBI_API_KEY=

# Semantic Scholar (Optional - higher rate limits with key)
SEMANTIC_SCHOLAR_API_KEY=

# OpenAlex (Just email for polite pool)
OPENALEX_EMAIL=YOUR_EMAIL@gmail.com

# ----- EMBEDDING MODEL -----
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# ----- RATE LIMITING -----
MAX_REQUESTS_PER_MINUTE=30

# ----- AGENT SETTINGS -----
LLM_TEMPERATURE=0.7
MAX_TOKENS=2000
```

### Step 3: Save the File

Press **Ctrl + S** to save, then close Notepad.

---

## ✅ QUICK CHECKLIST: Minimum Requirements

For the project to work with **100% FREE** options:

| Item | Status | Action |
|------|--------|--------|
| ☐ Ollama installed | Required | Download from ollama.com |
| ☐ Ollama model downloaded | Required | Run `ollama pull llama2` |
| ☐ .env file created | Required | Copy from .env.example |
| ☐ Email added to .env | Required | Replace YOUR_EMAIL@gmail.com |
| ☐ Docker running | Required | Already done! ✅ |
| ☐ Semantic Scholar key | Optional | Get from website (1-2 day wait) |
| ☐ NCBI API key | Optional | Get from NCBI account |

---

## 🚀 NEXT STEPS: After Getting API Keys

Once you have your `.env` file set up, continue with:

### Step 1: Open Command Prompt in Project Folder

1. Open File Explorer
2. Go to: `C:\Users\pushk\OneDrive\Documents\LangchainProject\hypothesis-engine`
3. Click in address bar, type `cmd`, press Enter

### Step 2: Create Virtual Environment

```powershell
python -m venv venv
```

### Step 3: Activate Virtual Environment

```powershell
venv\Scripts\activate
```

You should see `(venv)` at the start of your prompt.

### Step 4: Install Dependencies

```powershell
pip install -r requirements.txt
```

Wait 5-10 minutes for everything to install.

### Step 5: Start Docker Services

```powershell
docker-compose up -d
```

### Step 6: Start the API Server

```powershell
cd src\api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Step 7: Open New Terminal for Frontend

Open a NEW command prompt in the project folder, then:

```powershell
venv\Scripts\activate
cd frontend
streamlit run app.py
```

### Step 8: Open the Application

Go to: **http://localhost:8501**

---

## ❓ FAQ

### Q: Do I NEED OpenAI?
**A:** No! Use Ollama for free. It's just slower but works great.

### Q: What if I skip Semantic Scholar key?
**A:** The app still works, just with lower rate limits for that source.

### Q: Can I add keys later?
**A:** Yes! Just edit the `.env` file and restart the application.

### Q: Is my data safe?
**A:** With Ollama, everything runs locally. Nothing is sent to the cloud.

---

## 📞 Need Help?

If you get stuck on any step, let me know:
1. Which step you're on
2. What error message you see
3. Screenshots if possible

I'll help you through it!

---

*Guide Version: 1.0 | December 2024*
*For: Scientific Hypothesis Cross-Pollination Engine*
