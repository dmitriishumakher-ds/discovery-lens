# 🔍 Discovery Lens

**Discovery Lens** is a Streamlit app for product managers that turns raw qualitative discovery data: user interviews, app reviews, support tickets, usability notes, into a structured, evidence-backed [Opportunity-Solution Tree (OST)](https://www.producttalk.org/2023/12/opportunity-solution-trees/).

You set a product goal, upload your discovery files, and the tool clusters insights, frames opportunities in [Jobs-to-be-Done (JTBD)](https://jtbd.info/2-what-is-jobs-to-be-done-jtbd-796b82081cca) language, scores them with deterministic ODI signals, and shows exactly which source quotes justify each decision.

> Built as a capstone project for the [Neue Fische / Spiced Academy](https://www.neuefische.de/en/bootcamp/data-science-and-ai) Data Science & AI bootcamp. Runs entirely on CPU, no GPU, no database, no authentication required.

---

## Live demo

🚀 **[Try it on Hugging Face Spaces](https://huggingface.co/spaces/YOUR_SPACE_URL)** ← _update after deployment_

---

## How it works

```
Upload files → Chunk → Embed → Cluster → Score → LLM frames OST → Results
```

1. **Set your goal** — Enter your product name and a goal statement. A lightweight LLM gate checks that your goal is specific enough (measurable metric, user segment, timeframe) and offers a rewrite suggestion if it isn't.

2. **Upload your discovery files** — PDF, DOCX, CSV, or TXT. Tag each file with its source type (interview, review, support ticket, usability test, social, or internal notes). Optionally paste or upload a context block (OKRs, roadmap constraints) — it's injected into the LLM prompt so opportunities are both user-evidenced and strategically feasible.

3. **Pipeline runs** — Your text is chunked into 80-token sliding windows, embedded with `all-MiniLM-L6-v2`, and clustered with BERTopic + HDBSCAN. Each cluster is scored deterministically — no LLM involved in scoring.

4. **Get your OST** — The LLM (Groq) frames each cluster as a JTBD opportunity and proposes solutions with assumption risk ratings. Three independent scores are shown per opportunity so you know what to act on, what to validate, and what to monitor.

---

## Conceptual background

**Jobs-to-be-Done (JTBD)** is a framework for understanding user motivation. Instead of describing what users do, it captures why — framed as: _"When I [situation], I want to [motivation], so I can [outcome]."_ This format separates the user's underlying need from any specific solution, making it easier to evaluate whether a feature actually addresses the right problem.

**Opportunity-Solution Trees (OST)** are a product discovery structure popularised by Teresa Torres. The tree starts from a product goal, branches into opportunities (unmet user needs), and then into solutions and the assumptions those solutions rest on. The structure forces PMs to stay grounded in evidence before jumping to solutions, and makes the reasoning behind prioritisation decisions explicit and traceable.

**Outcome-Driven Innovation (ODI)** is a methodology by Tony Ulwick that scores opportunities using importance (how much users care about a need) and satisfaction (how well current solutions meet it). The core insight is that the most valuable opportunities are those where importance is high and satisfaction is low, the underserved needs. In its original form, ODI uses structured customer surveys to collect these scores.

Discovery Lens takes a computational approximation of this mechanic: importance is derived from cluster size relative to the total corpus, and satisfaction is derived from sentiment analysis of the chunks in each cluster. This is not a reproduction of the ODI methodology, it is an NLP-based proxy that captures the same signal from unstructured text without requiring structured surveys. The tradeoffs are acknowledged: sentiment is a coarser measure of satisfaction than direct survey responses, and cluster size is a coarser measure of importance than stated importance ratings. The approach is designed to be useful for rapid synthesis of existing discovery data, not to replace rigorous ODI research.

---

## Scoring system

Each opportunity card shows three scores, all computed deterministically by the pipeline (never by the LLM):

| Score | Formula | What it answers |
|---|---|---|
| `odi_score` | `importance × (1 − satisfaction)` | How underserved is this need? |
| `evidence_robustness` | `(source_type_diversity × 0.65) + (importance × 0.35)` | How robustly evidenced across source types? |
| `priority_score` | `[(odi_score × 0.60) + (evidence_robustness × 0.40)] × max(goal_relevance, 0.20)` | What should you act on first? |

Opportunities are also labelled automatically:

| Label | Condition |
|---|---|
| **Act** | `odi_score ≥ 0.10` and `evidence_robustness ≥ 0.40` |
| **Validate** | `odi_score ≥ 0.10` and `evidence_robustness < 0.40` |
| **Monitor** | `odi_score < 0.10` and `evidence_robustness ≥ 0.40` |
| **Deprioritise** | `odi_score < 0.10` and `evidence_robustness < 0.40` |

---

## Tech stack

| Layer | Choice |
|---|---|
| App framework | Streamlit |
| Deployment | Hugging Face Spaces (free tier: 2 vCPU, 16 GB RAM) |
| LLM | Groq — `llama-3.3-70b-versatile` (fallback: `llama-3.1-8b-instant`) |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` (CPU-only, 384 dims) |
| Clustering | BERTopic + HDBSCAN |
| Sentiment | `lxyuan/distilbert-base-multilingual-cased-sentiments-student` |
| Visualisation | Plotly |
| File parsing | pypdf, python-docx, pandas |

---

## Supported source types

| Type | Covers |
|---|---|
| `interview` | User interviews, contextual inquiry, diary studies |
| `review` | App store, G2, Capterra, ProductHunt reviews |
| `ticket` | Support tickets, CS chat transcripts |
| `usability` | Usability test notes, session recording summaries |
| `social` | Reddit threads, Twitter/X, community forums |
| `internal` | Sales call notes, CS/AM notes, escalation reports |

---

## Tips for best results

### Goal statement
The tool validates your goal before running the pipeline. A goal passes if it meets at least 2 of these 3 criteria but meeting all 3 produces noticeably better-framed opportunities:

* **Measurable metric** — names a concrete outcome, metric, or count (e.g. _"reduce churn by 15%"_, _"increase activation rate"_)
* **User segment** — identifies who the goal is for (e.g. _"for first-time users"_, _"for enterprise admins"_)
* **Timeframe or scope** — sets a boundary (e.g. _"by Q3"_, _"in the onboarding flow"_)

A weak goal like _"improve the product"_ will produce generic JTBD statements. A goal like _"increase 30-day retention among new mobile users by 20% before Q4"_ will produce opportunities that are specific, comparable, and directly actionable.
If the validator flags your goal, take the rewrite suggestion seriously, it costs 10 seconds and meaningfully improves output quality throughout the whole pipeline.

### Source diversity
`evidence_robustness` is scored against all 6 supported source types, regardless of how many you actually upload. A cluster that appears in only one source type (e.g. only app reviews) will score at most ~0.35 on evidence robustness, the size component, even if that cluster is very large. Uploading data from 3 or more distinct source types is where the score starts to differentiate meaningfully.

**The practical implication**: if you only have one source type available (e.g. a batch of support tickets), `odi_score` is still fully valid and useful, it captures unmet need independently of source diversity. Use it as your primary sort key in that case, and treat `evidence_robustness` scores as a known limitation rather than signal.

**As a rough guide**: 3+ source types gives the scoring system enough diversity signal to be meaningful. Mixing at least one qualitative source (interviews, usability) with at least one volume source (reviews, tickets) tends to produce the strongest cluster differentiation.

### Context block
The optional context block (OKRs, roadmap constraints, team capacity) is injected into the LLM prompt after the cluster evidence. It does not affect scoring, scores are always computed from the data alone. What it does affect is how the LLM frames solutions and assumption risk: a solution that requires new infrastructure will be flagged as higher risk if you've stated that no new infrastructure spend is approved.

**Keep it factual and specific**. Vague context (_"we want to grow"_) adds noise. Specific constraints (_"3-engineer team, Q3 hard deadline, no new third-party integrations"_) help the LLM generate assumptions that are actually useful to stress-test.

### File quality
The pipeline chunks text into 80-token windows. Very short files (under ~10 sentences) may not produce enough chunks to form their own cluster and will instead blend into other clusters. Very long files are fine, they just produce more chunks, which increases their weight in importance scores.
PDFs with complex layouts (multi-column, heavy tables, scanned images) may extract poorly. Plain text, DOCX, and well-structured PDFs give the cleanest results.

---

## Local setup

**Prerequisites:** Python 3.10+, a free [Groq API key](https://console.groq.com/)

```bash
git clone https://github.com/YOUR_ORG/discovery-lens.git
cd discovery-lens

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# open .env and add your key: GROQ_API_KEY=your_key_here

streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

> **Note:** First run downloads the `all-MiniLM-L6-v2` embedding model and the lxyuan sentiment model (~500 MB total). Subsequent runs use the cached versions.

---

## Constraints

- Single-user, single-project session — no authentication
- No database — all state lives in `st.session_state`
- All ML runs CPU-only

---

## Team

Built by [Lucas](https://github.com/Lookus22), [Mengda](https://github.com/XX2026),[Asma](https://github.com/asmajarrar2025-creator), and [Dmitrii](https://github.com/dmitriishumakher-ds) as part of the [Neue Fische / Spiced Academy](https://www.neuefische.de/en/bootcamp/data-science-and-ai) Data Science & AI bootcamp.

---

## License

[MIT](LICENSE)