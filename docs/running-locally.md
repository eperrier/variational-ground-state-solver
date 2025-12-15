# Running locally (no Docker)

## Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Ensure `PYTHONPATH` includes the `src/` directory (or install the package in editable mode):

```bash
export PYTHONPATH=$(pwd)/src
```

## Run the app

```bash
streamlit run app/streamlit_app.py
```

Then open the URL printed by Streamlit (typically `http://localhost:8501`).

## GPU (local)

If you want GPU acceleration without Docker, you must install a CUDA-enabled PyTorch build compatible with your system. Refer to the official PyTorch install instructions.

## Output directory

By default the app writes to:

- `1 - RESEARCH/PSITest/`

To change it:

```bash
export PSITEST_OUTPUT_DIR="/path/to/your/output/root"
```
