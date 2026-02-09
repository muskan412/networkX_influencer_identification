# Influencer Detection & Network Analytics 

This project builds an end-to-end workflow to **identify social influencers** using:
1) **Pairwise influencer classification** (Logistic Regression on relative feature differences), and  
2) **Twitter interaction network analysis** (NetworkX centrality + influencer ranking),  
and then **quantifies the financial value** of using an influencer model via an expected-profit simulation.

---

## What’s inside

### Part I — Predictors of influence (pairwise classification)
- **Goal:** Given a pair of users (A vs B), predict who is more influential.
- **Dataset:** Kaggle-style “Predict Who is More Influential” paired dataset with Twitter activity + network metrics.

**Modeling approach**
- **Feature engineering:**  
  - Log transform heavy-tailed metrics using `log1p(x)`  
  - Build relative features: `diff_feature = log(A_feature) - log(B_feature)`
- **Normalization:** MinMax scaling to `[0, 1]`
- **Model:** Logistic Regression (L2, `C=0.1`, `liblinear`)
- **Interpretability:** Coefficients used to identify strongest influence signals.

**Key results (from notebook)**
- Confusion matrix: `TN=2003, FP=695, FN=575, TP=2227 (N=5500)`
- Accuracy: **76.9%**
- Precision (class “A more influential”): **76.2%**
- Recall (class “A more influential”): **79.5%**

**Top predictors (qualitative)**
- `diff_listed_count` is the strongest signal (social curation / authority).
- `diff_follower_count` is next (reach still matters).
- Network position (`diff_degree`) matters more than pure posting volume.

---

### Part I-B — Financial value of the model (expected-profit lift)
- **Goal:** Convert predictive performance into business value.
- **Idea:** Use the model to target “true influencers” for a higher-impact campaign (two tweets).

**Assumptions (from notebook)**
- Profit per purchase: **$10**
- Buy probability (1 tweet): **0.01%** (`p1 = 0.0001`)
- Buy probability (2 tweets): **0.015%** (`p2 = 0.00015`)
- Fixed cost per user-pair: **$10** (e.g., $5 each)

**Scenarios**
1. **No analytics:** pay everyone, true influencer tweets once  
2. **Model-based:** if prediction is correct → influencer tweets twice; otherwise wasted spend on false positives / missed opportunities on false negatives  
3. **Perfect model:** always identifies influencers correctly; influencers tweet twice

**Key results (from notebook)**
- Expected net profit increases from **$5.32M** (no analytics) to **$7.42M** (model-based)  
- Incremental lift: **+$2.10M (~39.5%)**
- Perfect model: **$8.01M**
- Current model captures ~**78%** of attainable uplift (leaving ~**$0.59M** upside)

---

### Part II — Twitter network: parse → graph → centrality → influencer ranking
- **Goal:** Build a directed interaction network from tweets and identify influencers structurally.

**Tweet parsing → edge list**
We parse tweets into a Gephi-friendly edge list with interaction types:
- `Tweet`: self-loop (author → author)
- `RT`: amplification edge (author → retweeted user)
- `Reply`: author → replied-to user(s)
- `Mention`: author → mentioned user(s)

**Hard invariants enforced**
- Every author gets exactly one self-loop per tweet row.
- Every retweet has exactly one RT target (first @handle after “RT”).
- Every retweeted user must appear as a node (forced self-loop if never an author).

**Network metrics computed (NetworkX)**
- `in_degree`, `out_degree`, `total_degree`
- `betweenness`, `closeness`
- `pagerank`
- Each metric also min-max normalized to `[0, 1]`.

**Influencer ranking**
We reuse **Part I logistic coefficients** as weights to score users in the Part II network:
- Normalize features (with `log1p` for heavy-tailed counts)
- Influence score (log scale):  
  \[
  \text{score} = \sum_f \beta_f \cdot f_{norm}
  \]
- Convert to an odds-style index: `exp(score)` (ranking-only; not a calibrated probability)

We also include a **network-only** ranking variant that uses only:
- `in_degree`, `betweenness`, `closeness`

**Insights (from notebook)**
- Full scoring often emphasizes “platform authority” (listed + followers dominate).
- Network-only scoring surfaces two archetypes:
  - **Hubs** (high in-degree): conversation-orbit accounts
  - **Bridges** (high betweenness): cross-cluster connectors

---

