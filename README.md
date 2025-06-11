# Multi-Hop RAG Evaluation

Compare how well different RAG systems answer multi-step questions using original vs. summarized documents.

## What This Does

- Tests 3 approaches: LLM-only, Single-hop RAG, Multi-hop RAG
- Compares original documents vs. summarized documents
- Measures answer quality with F1 scores

## Quick Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

### 2. Setup Ollama (Recommended)

```bash
# Install Ollama from https://ollama.ai/
# Then pull a model:
ollama pull qwen3:8b
```

### 3. Prepare Data

Make sure you have:
- `data/multihoprag_corpus.txt` 
- `data/MultiHopRAG.json`
- `data_summary/multihoprag_corpus_summary.csv`

## Run Evaluation

### Quick Test (5 questions)
```bash
python run_evaluation.py
```

### Full Test (50 questions)
Edit `run_evaluation.py` and change the last line:
```python
if __name__ == "__main__":
    main()  # Change from quick_test()
```

Then run:
```bash
python run_evaluation.py
```


## Troubleshooting

**Ollama not working?**
```bash
ollama serve
ollama list  # Check if model is downloaded
```

**NLTK error?**
```bash
python -c "import nltk; nltk.download('punkt')"
```

**Want to use OpenAI instead?**
Create `.env` file:
```
OPENAI_API_KEY=your-key-here
```

## Using Different LLMs

Edit `run_evaluation.py` around line 45:

```python
# For Ollama (default)
llm = LLMWrapper(model_identifier="qwen3:8b", llm_type="ollama")

# For OpenAI  
llm = LLMWrapper(model_identifier="gpt-4-turbo", llm_type="openai")
``` 