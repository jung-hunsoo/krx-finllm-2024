# 2024 KRX 금융 sLLM 경진대회

## Instllation

1) Create and activate a new virtual environment

    ```bash
    python3 -m venv .venv
    . .venv/bin/activate
    ```

2) Install dependencies

    ```bash
    pip install -U -q --no-cache-dir -r requirements.txt
    ```

3) Create environment variables file

    ```bash
    # .env.secret
    
    OPENAI_API_KEY=sk-...
    HF_TOKEN=hf_...
    ```

## Basic usage

1) Parse external data files

    `python parse_raw.py -d DATASET_ID`
    e.g. `python parse_raw.py -d aihub_71782 -f -r`

2) Generate Q&As

    `python generate_qnsa.py -m MODEL_NAME -t TASK_NAME`

3) Merge generated

    `python merge_generated.py`
