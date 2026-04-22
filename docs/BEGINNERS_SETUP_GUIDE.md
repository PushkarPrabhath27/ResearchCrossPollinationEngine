# 🚀 Complete Beginner's Guide: Running the Scientific Hypothesis Engine

**For users with NO prior programming experience**

This guide will walk you through EVERY step needed to run this project on your Windows computer. Don't worry if you've never done this before - just follow each step carefully!

---

## 📋 TABLE OF CONTENTS

1. [Prerequisites - What You Need](#1-prerequisites)
2. [Installing Python](#2-installing-python)
3. [Installing Required Software](#3-installing-required-software)
4. [Setting Up the Project](#4-setting-up-the-project)
5. [Configuring API Keys](#5-configuring-api-keys)
6. [Starting the Databases](#6-starting-the-databases)
7. [Running the Application](#7-running-the-application)
8. [Using the Application](#8-using-the-application)
9. [Troubleshooting Common Issues](#9-troubleshooting)
10. [Next Steps](#10-next-steps)

---

## 1. Prerequisites - What You Need {#1-prerequisites}

Before starting, make sure you have:

✅ **A Windows computer** (Windows 10 or 11)
✅ **Internet connection**
✅ **Administrator access** to install software
✅ **About 5GB of free disk space**
✅ **About 30-45 minutes** for setup

**Optional but recommended:**
- An OpenAI API key ($5-10 credit is enough to start)
- OR Ollama installed for free local AI (explained below)

---

## 2. Installing Python {#2-installing-python}

Python is the programming language this project uses.

### Step 2.1: Download Python

1. Open your web browser
2. Go to: **https://www.python.org/downloads/**
3. Click the big yellow button that says **"Download Python 3.12.x"**
4. Wait for the file to download (it's about 25MB)

### Step 2.2: Install Python

1. Find the downloaded file (usually in your Downloads folder)
2. Double-click **python-3.12.x-amd64.exe**
3. **⚠️ IMPORTANT: Check the box that says "Add Python to PATH"** ← This is crucial!
4. Click **"Install Now"**
5. Wait for installation to complete
6. Click **"Close"**

### Step 2.3: Verify Python Installation

1. Press **Windows Key + R** on your keyboard
2. Type **cmd** and press Enter
3. In the black window that opens, type:
   ```
   python --version
   ```
4. Press Enter
5. You should see something like: `Python 3.12.1`

**If you see an error**, try closing and reopening the command window, or restart your computer.

---

## 3. Installing Required Software {#3-installing-required-software}

### Step 3.1: Install Docker Desktop (Required for Databases)

Docker helps run the database services.

1. Go to: **https://www.docker.com/products/docker-desktop/**
2. Click **"Download for Windows"**
3. Run the installer
4. Follow the installation prompts (keep default options)
5. **Restart your computer when asked**
6. After restart, Docker Desktop should start automatically

### Step 3.2: Verify Docker

1. Look for the Docker whale icon in your system tray (bottom right corner)
2. It should say "Docker Desktop is running"

**If Docker asks about WSL 2:**
- Click "Install WSL 2" and follow the prompts
- This is needed for Docker to work properly

### Step 3.3: Install Visual Studio Code (Recommended Editor)

This is optional but makes working with the project much easier.

1. Go to: **https://code.visualstudio.com/**
2. Download and install
3. When prompted, check "Add to PATH"

---

## 4. Setting Up the Project {#4-setting-up-the-project}

### Step 4.1: Open Terminal in Project Folder

1. Open **File Explorer**
2. Navigate to: `C:\Users\pushk\OneDrive\Documents\LangchainProject\hypothesis-engine`
3. Click in the address bar at the top
4. Type **cmd** and press Enter

A command window will open in the project folder.

### Step 4.2: Create a Virtual Environment

In the command window, type these commands one at a time:

```batch
python -m venv venv
```

Wait for it to complete (no output is normal).

### Step 4.3: Activate the Virtual Environment

```batch
venv\Scripts\activate
```

You should see `(venv)` appear at the start of your command line. This means you're now in an isolated Python environment.

### Step 4.4: Install Project Dependencies

This will install all the libraries the project needs:

```batch
pip install -r requirements.txt
```

**This will take 5-10 minutes.** You'll see lots of text scrolling - that's normal!

If you see any red text with "ERROR", don't panic. Contact for help with the specific error message.

### Step 4.5: Install Additional Dependencies

Some dependencies might need to be installed separately:

```batch
pip install langchain>=0.1.0 langchain-community>=0.0.20 langchain-openai chromadb sentence-transformers
pip install arxiv biopython beautifulsoup4 lxml pypdf
pip install fastapi uvicorn streamlit plotly pandas
pip install httpx aiohttp pydantic-settings
pip install sqlalchemy psycopg2-binary networkx
pip install pytest
```

---

## 5. Configuring API Keys {#5-configuring-api-keys}

### Step 5.1: Create Your Configuration File

1. In the project folder, find the file called `.env.example`
2. Make a copy of it (right-click → Copy, then right-click → Paste)
3. Rename the copy to `.env` (just remove ".example")

### Step 5.2: Edit the Configuration

1. Right-click on `.env`
2. Choose "Open with" → "Notepad" (or VS Code if installed)
3. You'll see content like this:

```
# OpenAI API (optional - can use Ollama instead)
OPENAI_API_KEY=your_key_here

# Database
CHROMA_PERSIST_DIR=./data/embeddings
METADATA_DB_PATH=./data/metadata.db

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Email for PubMed API (required by NCBI)
ENTREZ_EMAIL=your_email@example.com

# LLM Configuration
LLM_PROVIDER=openai
LLM_MODEL=gpt-4-turbo-preview
```

### Step 5.3: Get an OpenAI API Key (Option A - Paid but Easy)

1. Go to: **https://platform.openai.com/signup**
2. Create an account or sign in
3. Go to: **https://platform.openai.com/api-keys**
4. Click **"Create new secret key"**
5. Copy the key (it starts with `sk-`)
6. Paste it in your `.env` file replacing `your_key_here`

**Cost**: About $0.01-0.05 per hypothesis generation

### Step 5.4: OR Use Ollama for Free Local AI (Option B - Free but Slower)

If you don't want to pay for OpenAI:

1. Go to: **https://ollama.ai/**
2. Download and install Ollama for Windows
3. Open a command prompt and type:
   ```
   ollama pull llama2
   ```
4. In your `.env` file, change:
   ```
   LLM_PROVIDER=ollama
   LLM_MODEL=llama2
   ```

### Step 5.5: Add Your Email for PubMed

Replace `your_email@example.com` with your actual email address. PubMed requires this but doesn't send you anything.

### Step 5.6: Save the File

Press **Ctrl + S** to save, then close the file.

---

## 6. Starting the Databases {#6-starting-the-databases}

### Step 6.1: Start Docker Services

Make sure Docker Desktop is running (check for whale icon in system tray).

In your command window (in the project folder), type:

```batch
docker-compose up -d
```

This starts:
- **PostgreSQL**: Stores paper metadata
- **ChromaDB**: Stores paper embeddings for AI search

Wait until you see "Creating... done" messages.

### Step 6.2: Verify Databases Are Running

```batch
docker ps
```

You should see entries for `postgres` and possibly `chromadb`.

---

## 7. Running the Application {#7-running-the-application}

You'll need **TWO command windows** - one for the API server and one for the frontend.

### Step 7.1: Start the API Server (Window 1)

In your current command window:

```batch
cd src\api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
```

**Leave this window running!**

### Step 7.2: Open a Second Command Window (Window 2)

1. Open a NEW command window in the project folder
2. Activate the virtual environment again:
   ```batch
   venv\Scripts\activate
   ```

### Step 7.3: Start the Frontend (Window 2)

```batch
cd frontend
streamlit run app.py
```

You should see:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

### Step 7.4: Open the Application

1. Your browser should open automatically
2. If not, open your browser and go to: **http://localhost:8501**

🎉 **You should now see the Scientific Hypothesis Cross-Pollination Engine!**

---

## 8. Using the Application {#8-using-the-application}

### Step 8.1: The Main Interface

You'll see:
- A **text box** to enter your research question
- **Settings** on the left sidebar
- **Tabs** for different functions

### Step 8.2: Generate Your First Hypothesis

1. In the text box, type a research question, for example:
   ```
   How can machine learning techniques be used to improve early detection of Alzheimer's disease?
   ```

2. Adjust settings if desired:
   - **Primary Field**: Leave as "auto-detect" or select "biology"
   - **Year Range**: Keep defaults
   - **Max Hypotheses**: Start with 5

3. Click **"🚀 Generate Hypotheses"**

4. Wait 1-3 minutes (the AI is thinking!)

5. View your results:
   - Each hypothesis shows a title, scores, and description
   - Click to expand and see more details

### Step 8.3: Understanding the Scores

- **Novelty Score (0-1)**: How original is this idea?
- **Feasibility Score (0-1)**: How practical to implement?
- **Impact Potential (0-1)**: How significant if successful?

### Step 8.4: Searching Papers

1. Go to the **"Search Papers"** tab
2. Enter a search query
3. View semantically similar papers from the database

### Step 8.5: Downloading Results

1. Go to **"View Results"** tab
2. Click **"Download Results (JSON)"**
3. Save for your records

---

## 9. Troubleshooting Common Issues {#9-troubleshooting}

### Problem: "Python is not recognized"

**Solution**: 
- Restart your computer
- Reinstall Python, making sure to check "Add to PATH"

### Problem: "pip is not recognized"

**Solution**:
```batch
python -m pip install --upgrade pip
```

### Problem: Docker won't start

**Solution**:
- Ensure virtualization is enabled in BIOS
- Try running Docker Desktop as Administrator
- Ensure Hyper-V is enabled (Windows Features)

### Problem: "Module not found" errors

**Solution**:
```batch
pip install [module_name]
```

### Problem: API returns errors

**Solution**:
- Check your `.env` file has correct API key
- Ensure Docker containers are running: `docker ps`
- Check the API window for error messages

### Problem: "Connection refused" on localhost

**Solution**:
- Ensure the API server is running in Window 1
- Try accessing: http://127.0.0.1:8000/health
- Check if another program is using port 8000

### Problem: Very slow hypothesis generation

**Solution**:
- This is normal for the first run
- With Ollama, expect 2-5 minutes
- With OpenAI, expect 30-90 seconds

---

## 10. Next Steps {#10-next-steps}

### Stopping the Application

To stop everything:
1. Go to Window 1 (API) and press **Ctrl+C**
2. Go to Window 2 (Frontend) and press **Ctrl+C**
3. Stop Docker: `docker-compose down`

### Starting Again Later

Every time you want to use the application:

1. Open command window in project folder
2. Activate venv: `venv\Scripts\activate`
3. Start Docker: `docker-compose up -d`
4. Start API: `cd src\api && uvicorn main:app --reload`
5. (New window) Start frontend: `cd frontend && streamlit run app.py`

### Adding More Papers to the Database

To ingest papers for better results:

```python
# Run this in Python
from src.ingestion.arxiv_fetcher import ArxivFetcher

fetcher = ArxivFetcher()
papers = fetcher.fetch_papers(query="machine learning", max_results=100)
print(f"Fetched {len(papers)} papers")
```

### Getting Help

- Check the `docs/` folder for more documentation
- Review error messages carefully
- Search the error message online

---

## 📊 Quick Reference Commands

| Task | Command |
|------|---------|
| Activate virtual environment | `venv\Scripts\activate` |
| Start Docker services | `docker-compose up -d` |
| Stop Docker services | `docker-compose down` |
| Start API server | `uvicorn src.api.main:app --reload` |
| Start frontend | `streamlit run frontend/app.py` |
| Check Docker status | `docker ps` |
| Install missing package | `pip install [package_name]` |
| Run tests | `pytest tests/` |

---

## 🎓 Congratulations!

You've successfully set up and run a sophisticated AI-powered research tool!

**What you've accomplished:**
- ✅ Installed Python and required software
- ✅ Set up a professional development environment
- ✅ Configured API keys and databases
- ✅ Launched a full-stack AI application
- ✅ Generated your first research hypotheses

**Remember:**
- The more papers in the database, the better the results
- Keep your API keys secret
- Save interesting hypotheses for your research

---

*Guide created for the Scientific Hypothesis Cross-Pollination Engine*
*Version 1.0 - December 2024*
