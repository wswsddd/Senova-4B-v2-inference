# Senova-4B-v2

Senova-4B-v2 uses a revised training mixture of approximately 1.8 million
high-quality examples with option-set soft labels and adds Read Twice inference;
base model: Qwen/Qwen3.5-4B. Model weights are Apache-2.0; inference code is MIT.

- [Model weights](https://huggingface.co/WangSenQWQ/Senova-4B-v2)
- [Public scores.json](https://huggingface.co/datasets/WangSenQWQ/Senova-4B-v2-decision-index-results/blob/main/runs/senova_4b_v2_readtwice/scores.json)
- Public Decision Index 0.3: **67.27**, complete=true, 140178/140178 scoreable
  requests successful.

The uploaded repository includes `release.json` with exact model/results
revisions. Use the exact inference commit linked in the submission entry.

## 1. Install and download

Use Linux x86_64, Python 3.12 and an NVIDIA driver compatible with the CUDA 13.0
Torch wheel. Single-GPU target: one RTX PRO 6000 (96 GB). From this repository:

```bash
bash scripts/install_env.sh .venv
source .venv/bin/activate
python -m pip check
MODEL_REVISION=$(python -c 'import json; print(json.load(open("release.json"))["model_revision"])')
hf download WangSenQWQ/Senova-4B-v2 --revision "$MODEL_REVISION" --local-dir ./model
```

Backend, HTTP frontend and evaluation client share the same environment.
Core dependencies are pinned in `requirements.txt`: SGLang 0.5.15,
Torch 2.11.0 (observed build 2.11.0+cu130), Transformers 5.12.1,
sglang-kernel 0.4.4 and FlashInfer 0.6.12. These are core version pins, not a
complete transitive lockfile. Model and results repositories require no login
when public.

## 2. Start inference

Terminal 1, from the repository root:

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 bash scripts/serve_model.sh ./model
```

Wait for backend readiness:

```bash
curl --noproxy '*' -f http://127.0.0.1:8000/health
```

Terminal 2, from the same repository:

```bash
source .venv/bin/activate
python -m decision_index.systemone_server \
  --tokenizer ./model --backend-url http://127.0.0.1:8000/v1 \
  --host 127.0.0.1 --port 8008
```

Terminal 3, from the same repository:

```bash
source .venv/bin/activate
curl --noproxy '*' -f http://127.0.0.1:8008/health
python scripts/smoke_test.py --url http://127.0.0.1:8008
```

The frontend health response includes `"read_twice": true` and
`"enable_thinking": false`. Read Twice is fixed on by the frontend; no request
flag or extra evaluator option is needed. The smoke test uses synthetic data.

Both services bind to localhost and have no authentication. Run the evaluator
on the same host, or use an SSH tunnel. Do not expose them directly to the
public internet. Ensure localhost bypasses HTTP proxies:

```bash
export no_proxy=localhost,127.0.0.1,::1 NO_PROXY=localhost,127.0.0.1,::1
```

## 3. Run private evaluation

Keep the maintainers' private evaluator and scoring unchanged. Configure its
System One HTTP adapter with:

```text
--engine http --option base_url=http://127.0.0.1:8008 --model Senova-4B-v2
```

Pass each private case's original `state` and `questions` to
`POST http://127.0.0.1:8008/v1/systemone`. Do not send gold answers,
`expected`, metadata or `_evaluation`. No public suite directory is required.

This executable synthetic example shows the wire contract:

```python
import httpx

payload = {
    "model": "Senova-4B-v2",
    "state": "The kettle is red.",
    "questions": {
        "color": {
            "type": "choice",
            "instructions": "What color is the kettle?",
            "criteria": {"red": "Red", "blue": "Blue"},
        },
        "is_red": {"type": "noul", "instructions": "The kettle is red."},
    },
}
with httpx.Client(timeout=600, trust_env=False) as client:
    response = client.post("http://127.0.0.1:8008/v1/systemone", json=payload)
    response.raise_for_status()
    answers = response.json()["answers"]
    print(answers["color"]["choice"], answers["color"]["probabilities"])
    print(answers["is_red"]["noul"])  # P(true)
```

Replace the synthetic state/questions with private inputs. Each answer is
indexed by the original question key. A choice returns the original option
key and a probability for every supplied option. A noul returns P(true).
Instructions may be a string or object. Use those values with the official
private scorers; this server does not know private gold labels or scoring rules.

## 4. Exact inference settings and limits

- Read Twice: repeat the complete rendered user input twice, separated by
  `\n\n`, within one user message. Do not repeat the system message.
  One next-token inference per question, not two separate model calls.
- Thinking is fixed off; native SGLang `/generate`, temperature=0,
  max_new_tokens=1. Candidate logprobs are softmax-normalized across all
  candidates and mapped back to the original keys.
- Single GPU, TP=1, DP=1, one running backend request, serialized frontend
  decisions. Checkpoint dtype bfloat16 via dtype=auto, no quantization.
- Context limit: 262144 tokens including the requested output token, measured
  **after** Read Twice and chat-template rendering. No truncation or option
  filtering. Memory fraction 0.8; default SGLang CUDA graphs.
- Choice and noul types only; text/JSON state, no image input. Up to 588
  distinct verified single-token uppercase labels for this tokenizer.
  HTTP body limit: 16 MiB. Each choice requires at least two candidates.
- HTTP 413: declared capacity failure; 422: invalid/unsupported input;
  502: backend failure. GET /health returns 503 until backend readiness.
- Prompt version:
  `di-unified-letter-options-v4-ragtruth-train-aligned-read-twice-user-v1`.
  Scoring version: `sglang-next-token-v1`.
  Full transformations are in `decision_index/engines/sglang_engine.py`.

For official latency measurement, use one process and one request at a time on
one RTX PRO 6000. Maintainers measure mean, median and P80 against the
1000 ms requirement; public-run parallel timings are separate.

## 5. Public evaluation record

Public suite 0.3: index 67.27; complete=true; all 140178 scoreable requests
successful. This is not the private Full score. Original model/run identifiers
are preserved in results; Senova-4B-v2 is the release name.

Historical public run: eight endpoints, 288 client workers, TP=2 / DP=4 per
endpoint, context 262144, memory fraction 0.8, max running requests=32 per
endpoint, disabled CUDA graphs, Read Twice on, thinking off, unified-letter
labels, native logprobs. Original environment.json accompanies the predictions.

Public reproduction requires rebuilding/importing the official public suite
as documented in UPSTREAM_README.md, then using the same frontend:

```bash
python -m decision_index pipeline --engine http \
  --option base_url=http://127.0.0.1:8008 --model Senova-4B-v2 \
  --suite-dir suite-0.3 --edition 0.3 --compact \
  --out runs/senova_4b_v2_reproduction
```

Tests without loading weights:

```bash
python -m unittest test_systemone_server test_sglang_engine -v
```
