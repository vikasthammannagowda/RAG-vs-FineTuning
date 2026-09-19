# RAG-vs-FineTuning

# Step 0 - Verify Python Installation

Open **Command Prompt** and verify that Python is installed:

```bash
python --version
```

You should see output similar to:

```text
Python 3.11.9
```

If Python is not installed, download and install the latest version from https://www.python.org/downloads/.

During installation, make sure **Add Python to PATH** is selected.

---

# Step 1 - Install Ollama and Download Phi-3 Mini

1. Open your browser and go to:

   https://ollama.com

2. Click **Download** and install the Windows version.

3. After installation completes, open a new **Command Prompt** and verify the installation:

```bash
ollama --version
```

You should see the installed Ollama version.

4. Download the Phi-3 Mini model:

```bash
ollama pull phi3:mini
```

The model download is approximately **2.5 GB**. A stable internet connection is recommended. The model only needs to be downloaded once.

5. Verify the model works:

```bash
ollama run phi3:mini
```

Try a simple prompt such as:

```text
What is a manga?
```

If the model responds, the installation was successful.

---

# Step 2 - Create a Virtual Environment

Navigate to the project directory:

```bash
cd path\to\anime_llm_demo
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment:

```bash
venv\Scripts\activate
```

After activation, your command prompt should look similar to:

```text
(venv) C:\Users\YourName\anime_llm_demo>
```

The `(venv)` prefix indicates that the virtual environment is active and ready for package installation.
