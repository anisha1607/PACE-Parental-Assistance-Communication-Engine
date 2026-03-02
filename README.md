# PACE — Parental Assistance Communication Engine

PACE helps parents turn an **observed situation** into a calm, de-escalated message they can send to their teen using Non-Violent Communication (NVC).

The app uses a **safety guard + LLM pipeline**:
1. A BART zero-shot classifier filters unsafe or out-of-scope requests
2. Any LLM of your choice generates the empathetic response
3. A distress detection backstop intercepts crisis situations before the LLM and routes them to specialised safety prompts
4. A lightweight post-check blocks harmful output

---

##  What the App Does

**Input:**
One or two sentences describing a parent-teen situation you observed.

**Output:**
A short empathetic coaching response that:
- acknowledges the parent's feelings
- infers the teen's underlying needs
- produces a ready-to-send de-escalated message
- optionally asks one gentle reflective question

**Automatic refusal for:**
- spying / hacking / monitoring requests
- legal advice
- medical diagnosis
- harmful or adversarial content

**Distress detection:**
If the parent's message contains crisis signals (self-harm, suicidal language, substance use, school refusal, physical threats), the app bypasses coaching and responds with appropriate support resources instead.

---

## Tech

- **Backend:** FastAPI
- **Frontend:** Simple HTML/JS
- **Pre-LLM guard:** BART MNLI zero-shot classifier (`valhalla/distilbart-mnli-12-1`)
- **Distress backstop:** Regex + optional ML classifiers (ELECTRA suicidality, BERT severity)
- **Generator LLM:** Your choice — Groq, Anthropic, OpenAI, Ollama, or any OpenAI-compatible endpoint
- **Judge LLM:** Independent model for eval harness rubric scoring (configurable separately)
- **Package manager:** uv

---

## Setup

**1. Copy `.env.example` to `.env` and fill in your keys**

```bash
cp .env.example .env
```

Open `.env` and uncomment the provider you want to use. See `.env.example` for all options.

**2. Install dependencies**

```bash
uv sync
```

**3. Run the app**

```bash
uv run uvicorn app.main:app --port 8000
```

Open http://localhost:8000

---

## Supported LLM Providers

Configure in `.env` — no code changes needed:

| Provider | `LLM_PROVIDER` | Key needed |
|---|---|---|
| Groq | `groq` | `GROQ_API_KEY` |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` |
| OpenAI | `openai` | `OPENAI_API_KEY` |
| Ollama (local) | `ollama` | none |
| Any OpenAI-compatible | `openai_compatible` | optional |

The **judge LLM** (used in eval harness rubric scoring) can be set independently via `JUDGE_PROVIDER` and `JUDGE_MODEL` — allowing a different model to evaluate responses than the one that generated them.

---

## Running the Eval Harness

```bash
uv run python scripts/evaluate.py
```

This runs all 30 test cases and prints:
- Pass/fail per test with metric type used
- Category-wise pass rates (in-domain, out-of-scope, adversarial, rubric)
- Overall pass rate

**Test categories:**

| Category | Cases | Metric |
|---|---|---|
| in-domain | 10 | MaaJ golden-ref (keyword presence) |
| out-of-scope | 5 | Deterministic (refusal detection) |
| adversarial | 5 | Deterministic (refusal detection) |
| rubric | 10 | MaaJ rubric (LLM judge, scores 1-5) |

---

## Live URL

🔗 **https://pace-680681190998.us-central1.run.app/**

---

## GCP Deployment

- Deploy the FastAPI app on Cloud Run
- Set all env vars (API keys, `LLM_PROVIDER`, `LLM_MODEL`) in Cloud Run → Edit → Variables & Secrets
- Never upload `.env` to GitHub — use Cloud Run environment variables instead
- The BART guard runs on CPU (latency ~1-2s on first request, cached after)