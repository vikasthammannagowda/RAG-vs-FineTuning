# RAG-vs-FineTuning

# Step 0 - Verify Python Installation
2
 
3
Open **Command Prompt** and verify that Python is installed:
4
 
5
```bash
6
python --version
7
```
8
 
9
You should see output similar to:
10
 
11
```text
12
Python 3.11.9
13
```
14
 
15
If Python is not installed, download and install the latest version from https://www.python.org/downloads/.
16
 
17
During installation, make sure **Add Python to PATH** is selected.
18
 
19
---
20
 
21
# Step 1 - Install Ollama and Download Phi-3 Mini
22
 
23
1. Open your browser and go to:
24
 
25
https://ollama.com
26
 
27
2. Click **Download** and install the Windows version.
28
 
29
3. After installation completes, open a new **Command Prompt** and verify the installation:
30
 
31
```bash
32
ollama --version
33
```
34
 
35
You should see the installed Ollama version.
36
 
37
4. Download the Phi-3 Mini model:
38
 
39
```bash
40
ollama pull phi3:mini
41
```
42
 
43
The model download is approximately **2.5 GB**. A stable internet connection is recommended. The model only needs to be downloaded once.
44
 
45
5. Verify the model works:
46
 
47
```bash
48
ollama run phi3:mini
49
```
50
 
51
Try a simple prompt such as:
52
 
53
```text
54
What is a manga?
55
```
56
 
57
If the model responds, the installation was successful.
58
 
59
---
60
 
61
# Step 2 - Create a Virtual Environment
62
 
63
Navigate to the project directory:
64
 
65
```bash
66
cd path\to\anime_llm_demo
67
```
68
 
69
Create a virtual environment:
70
 
71
```bash
72
python -m venv venv
73
```
74
 
75
Activate the virtual environment:
76
 
77
```bash
78
venv\Scripts\activate
79
```
80
 
81
After activation, your command prompt should look similar to:
82
 
83
```text
84
(venv) C:\Users\YourName\anime_llm_demo>
85
```
86
 
87
The `(venv)` prefix indicates that the virtual environment is active and ready for package installation.
