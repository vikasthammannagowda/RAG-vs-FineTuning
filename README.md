# RAG-vs-FineTuning

# Project Overview

This project demonstrates two common techniques used to customize Large Language Models (LLMs):

1. Retrieval-Augmented Generation (RAG)
2. Fine-Tuning

Both approaches use the same base model, Phi-3 Mini, but modify its behavior in different ways.

The goal of this project is to make the model consistently include Japanese manga references in its responses and to compare how RAG and Fine-Tuning achieve that objective.

---

## Learning Objectives

By completing this guide, you will learn how to:

- Run a local LLM using Ollama.
- Chat with the base Phi-3 Mini model.
- Add external knowledge using Retrieval-Augmented Generation (RAG).
- Fine-tune a model using custom training data.
- Compare the strengths and weaknesses of RAG and Fine-Tuning.
- Convert a fine-tuned model into a standalone Ollama model.
- Upload a fine-tuned model to Hugging Face.

---

## Project Workflow

The guide is organized into the following sections:

### Step 0
Verify Python is installed and ready to use.

### Step 1
Install Ollama and download Phi-3 Mini.

### Step 2
Create a virtual environment and test the base model.

### Step 3
Run the RAG implementation and observe how retrieved knowledge influences responses.

### Step 4
Fine-tune Phi-3 Mini using custom manga-focused training datasets.

### Step 4A
Convert the fine-tuned model into a standalone Ollama model.

---

## Understanding the Comparison

Throughout this guide, you will compare three versions of the same model:

### Base Model

The original Phi-3 Mini model with no modifications.

### RAG-Enhanced Model

The original Phi-3 Mini model augmented with an external knowledge source at runtime.

### Fine-Tuned Model

A modified version of Phi-3 Mini trained on manga-specific examples.

By comparing the responses from all three systems, you will gain a practical understanding of when RAG is appropriate, when Fine-Tuning is beneficial, and the trade-offs associated with each approach.

# Hardware Requirements

This project was developed and tested using the following hardware:

- 32 GB System RAM
- NVIDIA RTX 4070 Laptop GPU (8 GB VRAM)
- Windows 11
- Python 3.11+
- Internet connection for downloading models and packages

---

## Minimum Requirements

For the RAG portion:

- 8 GB RAM
- Modern multi-core CPU
- Approximately 5 GB free disk space

For the Fine-Tuning portion:

- NVIDIA GPU with CUDA support
- 8 GB VRAM minimum
- 16 GB RAM minimum
- Approximately 15 GB free disk space

---

## Recommended Requirements

For the smoothest experience:

- 32 GB RAM
- NVIDIA RTX 4070 Laptop GPU (8 GB VRAM) or better
- 20+ GB free disk space
- Stable broadband internet connection

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


You can continue the conversation naturally, and Phi-3 Mini will generate responses based on your prompts.

To exit the chat session, type:

```text
/bye
```

or press:

```text
Ctrl + D
```

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

---

# Step 3 - Test Retrieval-Augmented Generation (RAG)

In this step, you will test a simple Retrieval-Augmented Generation (RAG) implementation. Unlike the base Phi-3 Mini model, the RAG system retrieves information from an external knowledge file before generating a response. This allows the model to incorporate additional information without modifying its original weights.

---

## Install the Dependencies

Make sure your virtual environment is activated.

Install the required Python packages:

```bash
pip install ollama sentence-transformers numpy
```

These packages provide:

- **ollama**: Connects to the locally running Phi-3 Mini model.
- **sentence-transformers**: Generates vector embeddings for retrieval.
- **numpy**: Supports vector calculations and similarity searches.

---

## Run the RAG Application

Navigate to the project directory containing the RAG script and run:

```bash
python rag_chat.py
```

After the script starts, you will see a chat prompt where you can begin interacting with the RAG-enhanced model.

---

## Test the System

Try prompts such as:

```text
Recommend a manga for someone who likes adventure stories.
```

```text
What manga would you suggest for fans of psychological thrillers?
```

```text
Tell me about a manga with strong themes of friendship and perseverance.
```

Notice that the responses now include references from the external knowledge source used by the RAG system.

---

## What to Observe

As you chat with the model, pay attention to the following:

- The model includes manga and anime references more consistently.
- Responses contain information retrieved from the knowledge file.
- New information can be added by updating the source document without retraining the model.
- The underlying Phi-3 Mini model remains unchanged.

This demonstrates a key advantage of RAG: **the model can access new knowledge without requiring fine-tuning or retraining**.

---

## Expected Outcome

Compared to the baseline Phi-3 Mini model from Step 2A, the RAG-enhanced version should produce responses that are more likely to mention manga and anime-related content because relevant information is retrieved and supplied to the model at runtime.

In the next step, we will compare this approach with fine-tuning, where the model itself is trained to naturally produce manga-focused responses.

# Step 4 - Fine-Tune Phi-3 Mini

In this step, you will fine-tune Phi-3 Mini using a manga-focused training dataset. Unlike RAG, which retrieves information from an external source at runtime, fine-tuning teaches the model to incorporate manga references directly into its responses.

This project includes multiple training datasets of different sizes so you can observe how the amount of training data affects the model's behavior.

---

## Experiment with Different Dataset Sizes

Before starting training, open `finetune.py` and locate the following line:

```python
DATA_FILE = "training_data.jsonl"
```

Update it to point to one of the available training datasets:

### Small Dataset (31 Samples)

```python
DATA_FILE = "training_data_31.jsonl"
```

### Medium Dataset (94 Samples)

```python
DATA_FILE = "training_data_94.jsonl"
```

### Large Dataset (285 Samples)

```python
DATA_FILE = "training_data_285.jsonl"
```

Run the complete fine-tuning process with each dataset and compare the results.

As the number of training examples increases, observe whether:

- Manga references become more consistent.
- Responses become more detailed.
- The model follows the desired style more reliably.
- Recommendations become more relevant.
- Response quality improves overall.

This comparison helps demonstrate one of the key concepts of fine-tuning: **training data quantity and quality directly influence model behavior**.

---

## Step 4.1 - Create a Dedicated Virtual Environment

Create a separate environment for fine-tuning:

```bash
python -m venv venv_ft
```

Activate it:

```bash
venv_ft\Scripts\activate
```

Your prompt should now display:

```text
(venv_ft)
```

---

## Step 4.2 - Install PyTorch with CUDA Support

Install the CUDA-enabled version of PyTorch:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

CUDA support allows training to run on your NVIDIA GPU, which is significantly faster than using only the CPU.

---

## Step 4.3 - Verify GPU Access

Confirm that PyTorch can access your GPU:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Expected output:

```text
True
```

**Do not continue until this command returns `True`.**

If the result is `False`, verify:

- Your NVIDIA drivers are installed.
- CUDA-compatible hardware is available.
- PyTorch was installed using the CUDA package above.

---

## Step 4.4 - Install Fine-Tuning Libraries

Install the required libraries:

```bash
pip install transformers peft trl datasets accelerate bitsandbytes
```

These packages provide:

- **transformers**: Loads and trains language models.
- **peft**: Implements efficient fine-tuning methods such as LoRA.
- **trl**: Supports supervised fine-tuning workflows.
- **datasets**: Loads and processes training datasets.
- **accelerate**: Optimizes training performance.
- **bitsandbytes**: Enables memory-efficient training.

---

## Step 4.5 - Run Fine-Tuning

Start the training process:

```bash
python finetune.py
```

During training, you may see:

- Dataset loading
- Model initialization
- Training progress
- Loss values
- Checkpoint creation

Depending on your hardware and dataset size, training time may vary.

Once training completes, the fine-tuned model will be saved to the configured output directory.

---

## Step 4.6 - Chat with the Fine-Tuned Model

After training finishes, launch the chat interface:

```bash
python chat_finetuned.py
```

Try the same prompts used in the baseline and RAG tests:

```text
Recommend a manga for someone who enjoys adventure stories.
```

```text
Suggest a psychological thriller manga.
```

```text
What manga would you recommend to fans of Naruto?
```

---

## What to Observe

Compare the responses from:

1. Base Phi-3 Mini
2. RAG-Enhanced Phi-3 Mini
3. Fine-Tuned Phi-3 Mini

Pay attention to:

- Consistency of manga references
- Quality of recommendations
- Response fluency
- Domain-specific knowledge
- Ability to answer without external retrieval

You should notice that the fine-tuned model naturally incorporates manga-related content even without accessing an external knowledge source. This is because the behavior has been learned during training rather than supplied at runtime through retrieval.


# Step 4A - Run the Fine-Tuned Model with Ollama (Optional)

If you would like to use your fine-tuned model directly through Ollama, you can merge the LoRA adapter into the base model and convert it into a standalone GGUF model. This allows you to chat with the model exactly like you do with the original Phi-3 Mini model, without needing any Python scripts.

---

## Step 4A.1 - Merge the Adapter into the Base Model

Run the merge script:

```bash
python merge.py
```

This creates a folder named:

```text
anime_model_merged
```

The folder contains a complete standalone version of your fine-tuned model.

---

## Step 4A.2 - Install the Conversion Tool

Install the required package:

```bash
pip install llama-cpp-python
```

Clone the llama.cpp repository:

```bash
git clone https://github.com/ggerganov/llama.cpp
```

Navigate into the repository:

```bash
cd llama.cpp
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Step 4A.3 - Convert the Model to GGUF Format

Run the conversion script:

```bash
python convert_hf_to_gguf.py ../anime_model_merged --outfile ../anime_model_merged/anime-phi3.gguf
```

After the conversion completes, you should have:

```text
anime_model_merged/
└── anime-phi3.gguf
```

The GGUF format is optimized for local inference and is supported by Ollama.

---

## Step 4A.4 - Create an Ollama Modelfile

Create a file named:

```text
Modelfile
```

Add the following content:

```text
FROM ./anime_model_merged/anime-phi3.gguf
```

Save the file in your project directory.

---

## Step 4A.5 - Register the Model with Ollama

Return to your project directory:

```bash
cd ..
```

Create an Ollama model:

```bash
ollama create anime-phi3 -f Modelfile
```

Ollama will import the GGUF model and register it under the name:

```text
anime-phi3
```

---

## Step 4A.6 - Run the Fine-Tuned Model

Launch the model:

```bash
ollama run anime-phi3
```

You can now chat with your fine-tuned model directly in the terminal:

```text
>>> Recommend a manga for fans of adventure stories.
```

```text
>>> Suggest a psychological thriller manga.
```

```text
>>> What manga would you recommend to someone who enjoyed Naruto?
```

---

## Why Use This Approach?

Benefits of converting the model for Ollama include:

- No Python scripts required for chatting.
- Easy distribution of the fine-tuned model.
- Consistent command-line experience.
- Simple deployment on other systems running Ollama.
- Works similarly to the original `phi3:mini` model.

At this point, your fine-tuned model behaves just like the base Phi-3 Mini model from the user's perspective, except that it has learned to incorporate manga-focused knowledge and recommendations based on your training data.