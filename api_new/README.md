# Quiver Signals

Investor-facing stock screener over Quiver alt-data (congressional trading, insider filings,
government contracts, lobbying, Reddit attention), powered by a machine-learning ranking
engine that is validated out-of-sample under the hood and honest about when its signals are —
and aren't — better than chance.

**Research tool, not investment advice.**

## Quickstart (one command)
```bash
./run.sh
```
Creates an isolated `.venv`, installs pinned dependencies, runs a health check, and launches
at http://localhost:8501. Safe to re-run any time; it reuses the venv and never touches conda
base. `./run.sh --check` does everything except launch.

Manual alternative:
```bash
python -m pip install -U -r requirements.txt
python check_setup.py        # optional: 30-second environment doctor
streamlit run app.py
```

First screen asks for your Quiver API key (validated before anything loads), or click
**Explore with sample data** — sample mode needs no key or internet and runs automatically.

## Using it
Sidebar has exactly two inputs: the tickers you care about (blank = auto-scan the most
politically-active names) and an investment horizon. Press **Find signals**. Results only
recompute when you press the button — tweaking settings just shows a hint until you do.

Tabs: **📊 Top Picks & Signals** (ranked names with plain-English reasons) · **🔍 Deep Dive**
(why each name ranks where it does, straight from the Quiver data) · **🛡️ Risk & Reliability**
(how trustworthy today's signals are, in plain English) · **📓 Methodology**.

The **Signal Quality** badge (High / Medium / Low) is computed from real out-of-sample results
plus a scrambled-data baseline. Buy/avoid framing unlocks only when signals genuinely beat
random; otherwise the dashboard says so and stays in research mode. On real disclosure data,
**Low is the common, expected outcome** — that's the tool being honest, not broken.

## Ship-readiness features
- **Input validation** — tickers are cleaned, deduped, capped (60) with clear feedback on
  anything ignored.
- **Typed errors with fix hints** — expired API key routes back to the connect screen;
  network, missing yfinance, empty universe, and no-price-data each get specific messages.
- **Per-tab error boundaries** — one broken view can't take down the app.
- **Staged progress** — you see each step (pull → prices → features → train & stress-test).
- **Provenance** — "data as of", which sources loaded, which failed, which tickers were
  dropped for lacking price history.
- **Versioned joblib cache** — instant re-runs; old cache files can never crash a new build;
  "Refresh latest data" bypasses once.
- **Native dark theme** (`.streamlit/config.toml`) matching the ink-navy/amber identity;
  telemetry off.

## Files
| File | Role |
|---|---|
| `app.py` | Investor-facing Streamlit app (gate, pipeline, dashboard). |
| `ml.py` | Purged walk-forward CV, OOS metrics, shuffled-label null, selection. |
| `features.py` | Point-in-time features, forward-excess labels, yfinance + sample providers. |
| `viz.py` | Plotly figures + shared template. |
| `cache.py` | Joblib cache (fingerprint keys, TTL, memoize). |
| `quiver_client.py` | Slim Quiver client + key validation. |
| `check_setup.py` | Environment doctor (deps, the starlette pitfall, engine smoke test). |
| `run.sh` | One-command venv + install + doctor + launch. |
| `.streamlit/config.toml` | Native theme + settings. |

## Troubleshooting
- **`ImportError: ... starlette.middleware.gzip`** — old starlette under new streamlit.
  `python -m pip install -U starlette` (run.sh/requirements now prevent this).
- **`streamlit: command not found`** — deps went into a different interpreter. Use `./run.sh`,
  or `python -m pip install ...` with the interpreter you'll launch from.
- **Wrong environment installs (conda base vs venv)** — `./run.sh` sidesteps this entirely by
  always using its own `.venv`. `python check_setup.py` prints exactly which interpreter and
  versions are live.
- **All live sources fail** — usually the key (you'll be routed back to reconnect) or the
  network; sample mode always works offline.
