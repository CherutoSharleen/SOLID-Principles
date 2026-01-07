# SOLID Retrieval Pipeline

## What it teaches

- Dependency inversion via `Tokenizer` and `Retriever` contracts
- Single-responsibility separation between tokenization and retrieval
- Local search over a synthetic corpus
- Swappable implementations (boolean overlap and BM25 ranking)
- Transparent overlap explanations for the boolean mode

## Run

```bash
cd examples/retrieval_pipeline
python app.py
```

## Test

```bash
cd examples/retrieval_pipeline
python -m unittest discover -s tests -p "test_*.py"
```

## Components

- `contracts.py`: protocol contracts
- `boolean_retriever.py`: basic overlap retriever implementation
- `bm25_retriever.py`: BM25 implementation using the same interfaces
- `corpus.py`: synthetic documents
- `app.py`: CLI entry point
- `explain.py`: query-term explanation helper

## Run with BM25 mode

```bash
cd examples/retrieval_pipeline
python app.py
# choose "bm25" when prompted
```

## Data note

All documents are fictional/synthetic.
