# VectorDB — Python Port

A fully working **Vector Database** built from scratch in Python with a web UI.  
Implements **HNSW**, **KD-Tree**, and **Brute Force** search algorithms side-by-side, plus a **RAG pipeline** powered by a local LLM via Ollama.



---

## Project Structure

```
VectorDB/
├── main.py      ← Python backend (HNSW, KD-Tree, BruteForce, REST API, RAG)
├── index.html   ← Frontend (unchanged from C++ version)
└── README.md    ← This file
```

---

## Prerequisites

1. **Python 3.9+**
2. **Ollama** (for real embeddings + RAG)

---

## Setup

### Step 1 — Install Python dependencies

```bash
pip install flask requests
```

### Step 2 — Install Ollama

1. Download from **https://ollama.com** and install
2. Pull the two required models:

```bash
ollama pull nomic-embed-text   # ~274 MB — embedding model
ollama pull llama3.2           # ~2 GB  — language model
```

### Step 3 — Run

```bash
python main.py
```

You should see:
```
=== VectorDB Engine (Python) ===
http://localhost:8080
20 demo vectors | 16 dims | HNSW+KD-Tree+BruteForce
Ollama: ONLINE
  embed model: nomic-embed-text  gen model: llama3.2
```

Open **http://localhost:8080** in your browser.

---

## REST API

Identical to the C++ version — see the original README for the full reference.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/search?v=f1,f2,...&k=5&metric=cosine&algo=hnsw` | K-NN search |
| `POST` | `/insert` | Insert a demo vector |
| `DELETE` | `/delete/:id` | Delete by ID |
| `GET` | `/items` | List all demo vectors |
| `GET` | `/benchmark?v=...&k=5&metric=cosine` | Compare all 3 algorithms |
| `GET` | `/hnsw-info` | HNSW graph structure |
| `GET` | `/stats` | Database statistics |
| `POST` | `/doc/insert` | Embed and store document |
| `GET` | `/doc/list` | List stored documents |
| `DELETE` | `/doc/delete/:id` | Delete document chunk |
| `POST` | `/doc/ask` | RAG: retrieve + generate |
| `GET` | `/status` | Ollama status and model info |

---

## Use a Smaller/Faster LLM

If `llama3.2` is too slow, switch to the 1B model:

```bash
ollama pull llama3.2:1b
```

Then edit `main.py`:
```python
self.gen_model = "llama3.2:1b"
```

---

## Common Issues

| Problem | Fix |
|---|---|
| `Ollama: OFFLINE` | Run `ollama serve` in a terminal |
| `ModuleNotFoundError: flask` | Run `pip install flask requests` |
| Port 8080 in use | Change `port=8080` in `main.py` to another port |
| Embedding takes forever | Ollama is downloading model on first use, wait ~2 min |

---

## License

MIT
