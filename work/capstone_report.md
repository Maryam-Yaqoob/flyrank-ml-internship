# Capstone Report — Growth / Recovery / Momentum Prediction

- **Author:** Maryam Yaqoob
- **Lane:** Freestyle — Growth / Recovery / Momentum Prediction
- **Repo:** https://github.com/Maryam-Yaqoob/flyrank-ml-internship
- **Date:** 2026-08-09

## 0. Abstract

Can a (client, content) page's first-half-of-month search activity flag, before the month ends,
that it is on track to lose more than 20% of its impressions in the second half? Using
`fact_content_daily_performance` (month=2026-03) from the FlyRank ML Internship warehouse —
9,841,378 rows across 55 clients, aggregated into 151,981 (client, content) pairs with a 32.7%
base decline rate — this project compares a transparent CTR-vs-position rule (Precision@10 =
50.0%, not client-held-out) against a Logistic Regression / Random Forest pair trained on five
first-half-only features and evaluated on clients neither model ever trained on. The
client-held-out model reaches a measured Precision@10 of **40.0%** (Logistic Regression), against
a 32.7% base rate. The output is a ranked, reason-coded review queue — decision support for a
content/SEO team with a fixed review budget, not an automated action or a claim about search
engine mechanics.

## 1. Problem framing

**Decision supported:** given a fixed content-review budget, which (client, content) pages
should a content/SEO lead check *this sprint*, before a decline is visible in an end-of-month
report? **Unit of analysis:** one (client, content) pair's activity within March 2026, split
into a first-half feature window (Mar 1–15) and a second-half label window (Mar 16–31).
**Output:** a ranked queue, sorted by predicted probability of `is_declining_next_half`, each row
carrying a plain-language reason code and a suggested action (e.g. review title/meta description,
treat as low-confidence). **The human action:** a person reviews the page, not an automated
change. **Cost of a wrong call:** a false positive is a wasted review — cheap and recoverable
next sprint; a false negative is a real decline that goes unreviewed until it already shows up in
a monthly report — the more expensive direction, which is why the primary metric is Precision@10
(trust at the very top of the list) rather than raw accuracy.

**Why ML over a fixed rule:** a hand-written rule was tried first (`w04_baseline_score.ipynb`) —
"flag pages with real volume whose CTR is below what their position tier typically gets." It
beats the base rate (50.0% vs 32.7% Precision@10), but both signals it leans on came back
**MIXED** when tested directly (`w04_signal_audit.ipynb`): CTR did not decrease cleanly by
position tier, and decline rate did not decrease cleanly by volume tier. A single hand-written
AND/OR rule isn't cleanly separating the pattern — the concrete evidence for trying a model that
can combine several weak, non-monotonic signals instead of picking one.

## 2. Data safety

**Source:** FlyRank ML Internship warehouse (gated, Hugging Face, Parquet), queried directly via
DuckDB — never downloaded locally. **Table:** `fact_content_daily_performance`, partition
`month=2026-03` (a mid-panel month; the sealed final-month `_sample` table was never touched).
Grain confirmed unique (`report_date × client_hash_id × content_hash_id`, 0 duplicates) in
`w03_data_contract.ipynb`.

**Features used (5, all Mar 1–15 only):** `impressions_first_half`, `clicks_first_half`,
`ctr_first_half`, `avg_position_first_half`, `active_days_first_half` — all aggregated
exclusively from the already-elapsed feature window.

**Excluded on purpose:** `impressions_second_half` — the exact raw material the label is built
from (`w03_data_contract.ipynb`'s leakage trap: training with it included made the score jump
unrealistically, confirming it as label material, not a feature). Any pre-computed FlyRank
scoring column, so the model learns a pattern rather than reproducing FlyRank's own prior
judgment. `client_hash_id` / `content_hash_id` — pseudonymous identifiers used only to build the
client-grouped train/test split, never as model inputs.

**Public-safety confirmation:** no client-identifying details, raw tokens, or private query
output appear anywhere in `work/`.

## 3. Baseline

**The rule** (`w04_baseline_score.ipynb`): a page is worth reviewing for CTR if it already has
real search visibility (`impressions_first_half ≥ 50`) but its click-through rate sits below what
pages at its own position tier typically get. No fitted weights — a transparent, hand-coded
AND/OR rule, scored on the whole month (not client-held-out).

| Method | Evaluated on | Precision@10 |
|---|---|---|
| Base rate | Whole modeling population | 32.7% |
| Rule baseline | Whole month, not client-held-out | 50.0% |

The rule clears the base rate, but both signals it depends on tested as **MIXED**
(Section 1) — real evidence the pattern needs more than one hand-picked feature.

## 4. Model / analysis

**Method:** Logistic Regression first (a coefficient-per-feature model, readable in one sentence
per feature — the honest floor for "can a straight line through 5 features beat the rule?"),
then Random Forest (adds non-linear splits and feature interactions the rule can't express).

**Target:** `is_declining_next_half` = 1 if a pair's summed impressions in Mar 16–31 are more
than 20% lower than Mar 1–15, else 0 — an observed future-window outcome, never a same-window
bucket.

**Features:** the same five honest, first-half-only features listed in Section 2.

**Which model won on Precision@10 (client-held-out):** Logistic Regression, at **40.0%** —
tied with Random Forest on Precision@10, but LR was taken forward for the error-analysis read in
Section 5 (`w05_model.ipynb`). Full comparison from `w05_model.ipynb`:

| Method | Precision@10 | Precision@50 | ROC-AUC |
|---|---|---|---|
| Base rate (floor) | 31.0% | 31.0% | 0.500 |
| Week-4 rule baseline (refit train-only, held-out clients) | 10.0% | 8.0% | 0.538 |
| Logistic Regression | 40.0% | 28.0% | 0.554 |
| Random Forest | 40.0% | 40.0% | 0.582 |

**Worth naming honestly:** on this held-out comparison, Random Forest actually edges out Logistic
Regression on Precision@50 and ROC-AUC (it stays useful further down the ranked list, not just at
the very top). The two tie exactly on the primary metric (Precision@10), which is why Logistic
Regression — the simpler, fully-readable model — was the one carried into the error analysis
below. A future iteration with a larger held-out client set could reasonably prefer Random Forest
instead; the tie itself is a legitimate result, not a resolved one.

## 5. Evaluation

**Split:** grouped by `client_hash_id`, not random rows — 70% of clients → train, 30% → test
(`test_size=0.3`, `random_state=42`), so no client's pages appear on both sides. This closes the
leak a random row split would allow (one client's template/strategy quirks bleeding across the
split). Time is handled by the feature engineering itself — the label window is strictly after
the feature window for every row, train or test. Train: 110,403 rows across 30 clients (base rate
33.3%). Test: 41,578 rows across 14 held-out clients (base rate 31.0%).

| Method | Evaluated on | Precision@10 |
|---|---|---|
| Base rate | Test-set clients | 31.0%\* |
| Rule baseline (refit train-only, scored on test clients) | Client-held-out | 10.0% |
| Model (Logistic Regression vs Random Forest, best of the two) | Client-held-out | 40.0% |

\* Test-client-only base rate (31.0%) sits close to the whole-population figure (32.7%) — no
meaningful split imbalance.

**A validation-design honesty point:** the Week-4 rule's 50.0% Precision@10 (Section 3) was
scored on the whole month, not client-held-out. Refit on train clients only and scored on the
same held-out test clients as the model, the rule collapses to **10.0%** — well below even the
base rate. That gap is itself a finding: the rule's original 50.0% number does not generalize to
unseen clients, and comparing it directly to the model's 40.0% without noting this would overstate
the rule's real-world performance.

**Error analysis** (`w05_model.ipynb` Section 4, reading Logistic Regression — the model taken
forward from Section 4):

*Coefficients (direction + rough weight):*

| Feature | Coefficient |
|---|---|
| `clicks_first_half` | −0.556 |
| `impressions_first_half` | +0.300 |
| `active_days_first_half` | −0.270 |
| `avg_position_first_half` | −0.102 |
| `ctr_first_half` | +0.019 |

*Random Forest permutation importance (drop in ROC-AUC when a feature is shuffled), for
comparison:* `ctr_first_half` +0.0455, `avg_position_first_half` +0.0371,
`impressions_first_half` +0.0349, `active_days_first_half` +0.0232, `clicks_first_half` +0.0114.

*Where the model is most wrong* (top-decile prediction vs actual, by tier, n≥20) — worst cases
cluster in `low (<10 first-half clicks)` volume tier regardless of position tier, with error
rates from 39% up to **74.1%** for `no_data`-position / low-volume pairs — the model is least
reliable exactly where it has the least first-half signal to work with.

*3 concrete wrong cases* (flagged in the predicted top 10, did **not** actually decline):
- `content_8e1334d6…` — score 0.996, 58,553 first-half impressions, 0.00% CTR, position 4.6, 15
  active days. Looked at-risk on paper but held steady — likely a case the 5 features genuinely
  can't see (e.g. a mid-March content refresh not captured here).
- `content_42571554…` — score 0.857, 24,678 first-half impressions, 0.01% CTR, position 5.4, 15
  active days. Same pattern.
- `content_1bb7d17c…` — score 0.856, 27,300 first-half impressions, 0.03% CTR, position 5.1, 15
  active days. Same pattern.

## 6. Interpretation

**Signal audit** (`w04_signal_audit.ipynb`), tested directly on this slice:

| Test | Hypothesis | Verdict |
|---|---|---|
| 1 — CTR vs position tier | Better (lower) position → higher CTR | MIXED |
| 2 — Impression volume vs decline rate | More volume → less likely to decline | MIXED |
| 3 — Active-day coverage vs decline rate | More first-half coverage → different decline rate | MIXED — sparse (1–3d) coverage sits at a 5.9% decline rate, then partial/mostly/full (4–15d) all cluster around ~30%; the relationship is a step, not a clean monotonic trend |
| Flag-linked — position vs impressions (local) | Matches paper's portfolio-level health-score-vs-position pattern (r = −0.592)? | Directionally consistent but weak — r = −0.059 (n = 150,674) vs the paper's portfolio-level r = −0.592. Same negative direction, on a different value metric and a much narrower slice, but far weaker |

**What this means:** two of three signals a content team might assume hold true — "better
position always means better CTR," "more volume always means safer from decline" — came back
MIXED on this slice, not confirmed. Active-day coverage tells a similar story: a real but
non-monotonic step (sparse coverage clearly worse, but partial/mostly/full are indistinguishable
from each other), not a clean "more is always better" line. That's a real, useful finding on its
own: it's evidence against trusting any single-signal rule, and the reason a model that weighs
several imperfect signals together is worth the extra complexity over the hand-written rule —
which Section 5's comparison confirms (40.0% vs the rule's held-out 10.0%).

## 7. Recommendation

Ranked output from `w07_action_playbook.ipynb`'s exported queue
(`work/outputs/action_playbook_queue.csv`):

1. **Review `high_volume_at_risk`-flagged pages first** — most impressions at stake if the flag
   is right, so the highest expected-value use of limited review time.
2. **Treat `thin_coverage`-flagged pages (fewer than 5 active days in the feature window) as
   lower-confidence** — verify against daily-level data before acting; the underlying estimate is
   noisier.
3. **Never automate directly from this queue** — no auto-delete, de-index, or redirect. A human
   checks the live page, rules out a site-wide cause (redesign, migration, tracking change), and
   checks for a known seasonal dip before treating a flag as content-specific.

**Confidence and limits, stated plainly:** this is decision support for March 2026, this
warehouse slice, and this specific 20%-drop-vs-first-half target — not a general health score,
not a claim about search-engine mechanics, and not validated yet for other months or clients
outside this dataset (see Limitations, `capstone.ipynb` Section 5). The model is also least
reliable in exactly the low-first-half-signal cases flagged in Section 5's error analysis —
a real, stated limit, not a hidden one.

## 8. Reproducibility

**Re-run from a fresh clone:**
1. Clone the repo, open `work/notebooks/capstone.ipynb` in Colab (badge link in
   `work/README.md`).
2. Add your own `HF_TOKEN` (Read token, gated-repositories permission) to Colab's Secrets panel
   — never pasted into a cell.
3. Runtime → Run all. The notebook is self-contained: it rebuilds `feature_frame`, trains the
   model, and rebuilds the ranked queue inline in that session.

**Seeds:** `random_state=42` throughout (train/test client split, Logistic Regression, Random
Forest). **Environment:** `%pip -q install duckdb scikit-learn` inside each notebook; see
`requirements.txt` for the reference pipeline's pinned versions. **Sealed evaluation:** this
project does not claim a sealed holdout beyond the client-grouped test split described in
Section 5 — that split's build cell and the metrics it produces are both committed, so "evaluated
once, on held-out clients" is checkable from the repo.

## 9. Acknowledgments & data credit

Built on the FlyRank ML Internship dataset — [flyrank.ai](https://flyrank.ai).
