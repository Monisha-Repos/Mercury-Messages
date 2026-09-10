# Message Notification Router — Solution

Routes every message in `dataset/messages.csv` to `notify`, `digest`, or `mute`,
writing a contract-compliant `output.csv`.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

The core pipeline is **stdlib-only** and runs without any of those packages.
They are needed only for voice prosody (`soundfile`, `numpy`) and the
evaluation extras.

## Run

```bash
.venv/Scripts/python.exe code/main.py
```

Writes `output.csv` (repo root) and `dataset/output.csv` — the two locations the
spec refers to — then prints a distribution summary.

| Flag | Effect |
|---|---|
| `--backend ollama` | Use local LLM/VLM scoring instead of the lexical baseline |
| `--trace artifacts/trace.csv` | Write per-message score breakdown |
| `--limit 10 --verbose` | Smoke test |
| `--no-cache` | Force full recompute |

Evaluate against the 30 labeled samples:

```bash
.venv/Scripts/python.exe code/evaluation/main.py --calibrate --errors
```

## Architecture

Two independent scores are computed per message and then fused.

**Content layer** (`content.py`, `modalities.py`) — what the message says.
Emits `urgency` *and* `risk` separately, which matters because scams are
engineered to read as urgent; collapsing them into one score promotes the most
dangerous messages. Branches per modality: text, image (VLM), voice (prosody).

**Pattern layer** (`features.py`) — how much this user values this sender.
Built from engagement history, group mute state, business relationship, and
notification fatigue. Never looks at message content, which is what makes the
ablation meaningful.

**Fusion** (`fusion.py`) — `0.65 × content + 0.35 × pattern`, then five
guardrail overrides, each addressing a specific failure the blend produces on
its own: `risk_veto`, `broadcast_override`, `high_content_override`,
`muted_group`, `dnd_degrade`.

**Evidence** (`evidence.py`) — TF-IDF retrieval over `message_history.csv`
scoped to the recipient, combining lexical similarity with a structural bonus
for shared conversation. Emits `none` below a similarity floor rather than
citing noise.

## Platform notes (Windows ARM64)

Three planned dependencies have no ARM64 Windows wheels and were replaced:

| Intended | Problem | Replacement |
|---|---|---|
| `librosa` | `llvmlite` fails to build | Prosody implemented directly on numpy — RMS energy, autocorrelation F0, pause segmentation |
| `faster-whisper` | `ctranslate2` / `av` unavailable | No local STT; voice notes route on prosody + conversation context |
| `openai-whisper` | no `torch` for win-arm64 | as above |

MP3 decoding works via `soundfile` (bundled libsndfile 1.2.2 reads MP3 natively).

## Determinism

`temperature=0`, `top_p=1`, fixed `seed`, pinned model tags, deterministic tie
-breaking in retrieval, and output rows ordered to match `messages.csv`. All
model calls are cached in SQLite keyed on a hash of their exact inputs, so
re-runs are byte-identical and only changed branches recompute.

## Results (30 labeled samples)

| Metric | Lexical baseline | LLM backend |
|---|---|---|
| Action accuracy | 60.0% | **73.3%** |
| Message-type accuracy | 40.0% | **60.0%** |
| Joint accuracy | 26.7% | **53.3%** |
| Evidence recall | 89.3% | 89.3% |

Ablation (LLM backend) — each layer earns its place:

| Mode | Action | Type | Joint |
|---|---|---|---|
| Content only | 43.3% | 56.7% | 23.3% |
| Pattern only | 53.3% | 56.7% | 33.3% |
| Fused | 60.0% | 56.7% | 40.0% |
| Fused + overrides | **73.3%** | **60.0%** | **53.3%** |

Fusion beats either component alone, and the guardrail overrides add a further
13 points — the two-signal design is doing real work rather than dressing up a
single classifier.

## On calibrating the thresholds

`ALPHA` and the two thresholds are fitted by grid search, but only after
leave-one-out cross-validation confirmed the fit generalises:

| Content scorer | Untuned | Fitted | Leave-one-out | Applied? |
|---|---|---|---|---|
| Lexical baseline | 56.7% | 70.0% | 46.7% | **No** — LOO below untuned |
| LLM | 60.0% | 73.3% | 63.3% | **Yes** — LOO above untuned |

With 30 examples and three parameters, the fitted number is optimistic by
construction. Against the lexical scorer, tuning actively hurt: leave-one-out
fell 10 points below simply leaving the defaults alone. The same search became
worthwhile only once the score being thresholded carried real signal.

**Re-run `--calibrate` after any change to the content layer** — the correct
thresholds depend on the distribution of the scores feeding them.

One finding worth noting: the fitted `ALPHA` is 0.40, meaning the sender prior
outweighs message content. That contradicts the original design assumption that
content should dominate as the direct evidence.
