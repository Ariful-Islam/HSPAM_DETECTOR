# 📧 HSPAM Detector — SMS/Email Spam Classifier with Online Learning

A production-style **spam detection web app** built with FastAPI, scikit-learn, and MySQL.
It classifies incoming messages as **spam** or **ham**, explains *why* using per-token
contributions, and learns from user feedback through a persistent, reproducible
training pipeline.

Built as a **learning project** that mirrors how real spam filters (Rspamd,
SpamAssassin) actually work: a probabilistic classifier + a feedback loop +
frozen evaluation benchmarks.

---

## ✨ Features

- 🧠 **Naive Bayes classifier** trained on TF-IDF features (unigrams + bigrams)
- 🧹 **Custom text preprocessor** — normalizes URLs, emails, numbers, punctuation
- 📊 **Explainable predictions** — shows the top tokens pushing toward spam or ham
- 💾 **MySQL-backed dataset** — every labeled message lives in the database
- 🔁 **Feedback loop** — user corrections become future training data
- 🧪 **Frozen train/test split** — reproducible, leakage-free evaluation
- 📈 **Experiment tracking** — every training run logged with metrics & hyperparams
- 🔍 **Per-message test results** — query which test samples the model gets wrong
- 🌐 **Interactive web UI** — paste a message, get a verdict, flag corrections
- 📚 **Auto-generated API docs** — Swagger UI at `/docs`

---

## 🏗️ Architecture

```
┌──────────────┐    POST /predict     ┌──────────────────────┐
│              │  { message }         │                      │
│  Browser UI  │ ───────────────────► │      FastAPI         │
│              │                      │                      │
│              │ ◄─────────────────── │  ┌────────────────┐  │
│              │  { label, score,     │  │  ML Pipeline   │  │
│              │    top_tokens }      │  │ (TF-IDF + NB)  │  │
│              │                      │  └───────┬────────┘  │
│              │  POST /feedback      │          │           │
│              │  { message, label }  │          ▼           │
│              │ ───────────────────► │  ┌────────────────┐  │
└──────────────┘                      │  │     MySQL      │  │
                                      │  │  messages      │  │
                                      │  │  training_runs │  │
                                      │  │  test_preds    │  │
                                      │  └────────────────┘  │
                                      └──────────────────────┘
                                                ▲
                                                │
                                      ┌──────────────────┐
                                      │  train.py (CLI)  │
                                      │  fit → eval →    │
                                      │  save → log run  │
                                      └──────────────────┘
```

**Learning loop:**

```
User flags message  →  /feedback  →  messages.split='unassigned'
                                           │
                                           ▼
                                    python train.py
                                           │
                          ┌────────────────┴────────────────┐
                          ▼                                 ▼
                  train on split='train'         evaluate on split='test'
                          │                                 │
                          └────────► new TrainingRun ◄──────┘
                                          │
                                          ▼
                                 TestPrediction rows
                                 (per-message accuracy)
```

---

## 🧰 Tech Stack

| Layer            | Tool                                                    |
|------------------|---------------------------------------------------------|
| **Web framework**| [FastAPI](https://fastapi.tiangolo.com/)                |
| **Server**       | [Uvicorn](https://www.uvicorn.org/)                     |
| **ML**           | [scikit-learn](https://scikit-learn.org/) (TF-IDF + MultinomialNB) |
| **Database**     | MySQL 8 via SQLAlchemy + PyMySQL                        |
| **Templating**   | Jinja2                                                  |
| **Serialization**| joblib                                                  |

---

## 📁 Project Structure

```
HSPAM_DETECTOR/
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI app + routes
│   ├── config.py          # env-driven settings
│   ├── database.py        # SQLAlchemy engine + session
│   ├── models.py          # ORM: Message, TrainingRun, TestPrediction, PredictionLog
│   ├── schemas.py         # Pydantic request/response models
│   ├── ml.py              # TextPreprocessor + ModelStore + predict_with_explanation
│   └── split.py           # assign_splits_once + get_train/test_rows
├── templates/
│   └── index.html         # Web UI
├── static/
│   └── style.css          # Styling
├── seed.py                # Load initial labeled data into MySQL
├── train.py               # CLI: train, evaluate, log run
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python **3.10+**
- MySQL **8.x** running locally or accessible over the network

### 1. Clone the repository

```bash
git clone https://github.com/Ariful-Islam/HSPAM_DETECTOR.git
cd HSPAM_DETECTOR
```

### 2. Create a virtual environment

```bash
python -m venv env
source env/bin/activate        # macOS/Linux
# env\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up MySQL

Log in and run:

```sql
CREATE DATABASE spam_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'spam_user'@'%' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON spam_db.* TO 'spam_user'@'%';
FLUSH PRIVILEGES;
```

### 5. Configure environment

Change `.env` file with your environment details:

```env
DB_HOST=127.0.0.1
DB_PORT=3306
DB_USER=spam_user
DB_PASSWORD=your_password
DB_NAME=spam_db
MODEL_PATH=spam_filter.joblib
```

### 6. Create tables & seed data

```bash
python seed.py
```

This creates the schema and loads the initial labeled dataset into `messages`.
If you have the [SMS Spam Collection](https://www.kaggle.com/uciml/sms-spam-collection-dataset)
file (`SMSSpamCollection.csv`) in the project root, it will be used automatically.
Otherwise a small inline sample is used as a fallback.

### 7. Assign the frozen train/test split

```bash
python -c "from app.database import sessionLocal; \
           from app.split import assign_splits_once; \
           db = sessionLocal(); print(assign_splits_once(db)); db.close()"
```

Expected output:

```python
{'train': 4450, 'test': 1112, 'train_ham': 3850, 'train_spam': 600, 'test_ham': 963, 'test_spam': 149}
```

### 8. Train the model

```bash
python train.py
```

Sample output:

```
Training on 4450 rows (600 spam / 3850 ham)
Testing on  1112 rows
============================================================
Model version : v1728723456
Fit time      : 0.42s
============================================================
accuracy        : 0.9810
precision_spam  : 0.9534
recall_spam     : 0.8926
f1_spam         : 0.9220
f1_weighted     : 0.9799

Confusion Matrix (rows=true, cols=pred, order=ham,spam):
[[955   8]
 [ 16 133]]

✅ Run #1 saved to training_runs (1112 per-message predictions).
```

### 9. Start the API

```bash
env/bin/uvicorn app.main:app --reload --port 8000
```

Open:

- **Web UI:** http://127.0.0.1:8000/
- **Swagger docs:** http://127.0.0.1:8000/docs
- **Health check:** http://127.0.0.1:8000/health

---

## 🔌 API Reference

| Method | Endpoint        | Description                                    |
|--------|-----------------|------------------------------------------------|
| `GET`  | `/`             | Web UI                                         |
| `GET`  | `/health`       | Service + model status                         |
| `POST` | `/predict`      | Classify a message, return score + explanation |
| `POST` | `/feedback`     | Store a user-labeled message (`unassigned`)    |
| `GET`  | `/runs`         | List recent training runs                      |
| `GET`  | `/runs/{id}`    | Full detail of one run + per-message results   |
| `GET`  | `/stats`        | Dataset composition + counters                 |

### `POST /predict`

**Request:**

```json
{ "message": "WINNER!! You've won a FREE iPhone! Click http://bit.ly/xyz" }
```

**Response:**

```json
{
  "label": "spam",
  "spam_score": 0.9823,
  "ham_score": 0.0177,
  "top_tokens": [
    { "token": "url",     "weight": 2.1043 },
    { "token": "won",     "weight": 1.6521 },
    { "token": "winner",  "weight": 1.4210 },
    { "token": "free",    "weight": 0.9822 },
    { "token": "click",   "weight": 0.7710 }
  ]
}
```

### `POST /feedback`

**Request:**

```json
{ "message": "Limited time offer! Buy now!", "label": "spam" }
```

**Response:**

```json
{ "id": 5572, "label": "spam", "status": "stored", "split": "unassigned" }
```

The row enters the database as `unassigned` and is promoted to `train`
on the next `python train.py` run. **The test set is never touched.**

---

## 🧪 CLI Usage

### Seeding

```bash
python seed.py           # Creates tables and loads labeled messages
```

### Training

```bash
python train.py                        # Defaults
python train.py --alpha 0.5            # Stronger Laplace smoothing
python train.py --ngram 1 1            # Unigrams only (no bigrams)
python train.py --ngram 1 3            # Unigrams through trigrams
python train.py --min-df 1             # Include rare terms
python train.py --output models/nb_v2.joblib
```

Every run is recorded in `training_runs` with hyperparameters and metrics.

---

## 🔬 Experiments & Evaluation

Because every run is logged, you can compare models with SQL:

**Compare hyperparameter sweeps:**
```sql
SELECT id, hyperparams, f1_spam, f1_weighted, accuracy
FROM training_runs
ORDER BY f1_spam DESC
LIMIT 10;
```

**Track improvement as feedback accumulates:**
```sql
SELECT id, num_train, f1_spam, f1_weighted, created_at
FROM training_runs
ORDER BY id;
```

**Find "always wrong" test cases:**
```sql
SELECT m.content, m.label, COUNT(*) AS times_wrong
FROM test_predictions tp
JOIN messages m ON m.id = tp.message_id
WHERE tp.correct = 0
GROUP BY m.id
ORDER BY times_wrong DESC
LIMIT 10;
```

That last query surfaces **hard examples** — the cases the model consistently
misses. Exactly the ones worth studying when improving features.

---

## 🧠 How It Works

### 1. Preprocessing
Raw text is lowercased and normalized. URLs, emails, and numbers are replaced
with semantic placeholder tokens (`url`, `email`, `num`) — these become strong
spam indicators without exploding the vocabulary.

```
Input : "WINNER!! You've won £5000! Click http://bit.ly/xyz NOW!"
Output: "winner you ve won num click url now"
```

### 2. Vectorization (TF-IDF)
Each message becomes a sparse vector. Word importance is weighted by:

```
tfidf(t, d) = TF(t, d) × IDF(t)
```

Words that appear **often in this message** but **rarely everywhere else**
score highest. Unigrams + bigrams capture phrases like *"free money"*.

### 3. Classification (Multinomial Naive Bayes)
Bayes' theorem with the naive independence assumption:

```
P(spam | msg) ∝ P(spam) · Π P(wordᵢ | spam)
```

Laplace smoothing (`alpha`) prevents zero probabilities for unseen words.

### 4. Explainability
For each token in the message:

```
weight(token) = tfidf_value × [log P(word|spam) − log P(word|ham)]
```

Positive → pushes toward spam. Negative → pushes toward ham.
Shown in the UI under "Why?".

---

## 🗄️ Database Schema

| Table              | Purpose                                              |
|--------------------|------------------------------------------------------|
| `messages`         | Labeled data with `split` ∈ {train, test, unassigned}|
| `training_runs`    | One row per retrain, with metrics + hyperparams      |
| `test_predictions` | Per-message outcome on the test split                |
| `predictions`      | Live audit log of every served prediction            |

The `messages.split` column is the **source of truth** for train/test.
Once a message is assigned to `test`, it never enters training. This makes
every metric comparable across runs.

---

## 🛡️ Design Decisions

- **Frozen test split** — assigned once, never mutated. Prevents leakage and
  makes F1 comparable across retrains.
- **New feedback = `unassigned`** — user corrections only ever enter the
  *training* set, never the benchmark.
- **Pipeline everywhere** — sklearn `Pipeline` guarantees identical
  preprocessing at train and predict time.
- **CLI training, not API** — heavy work stays out of the web process,
  so long training doesn't block requests.
- **Explainable by default** — the UI shows *why* a verdict was made,
  building trust and helping debug misclassifications.

---

## 🗺️ Roadmap

- [ ] User accounts + per-user feedback caps (prevent poisoning)
- [ ] Sample weighting for user-sourced rows
- [ ] `ComplementNB` and `LogisticRegression` model comparison
- [ ] Threshold tuning endpoint (trade precision for recall)
- [ ] Learning-curve chart (F1 vs training size over time)
- [ ] Daily automated retrain via cron or APScheduler
- [ ] Docker + Docker Compose for one-command setup
- [ ] Postfix milter integration for live email filtering
- [ ] Alembic migrations instead of `create_all()`

---

## 🤝 Contributing

This is a learning project. Issues, PRs, and experiment reports are welcome.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-experiment`
3. Commit your changes: `git commit -m "Add ComplementNB comparison"`
4. Push and open a Pull Request

---

## 📜 License

MIT License — see `LICENSE` for details.

---

## 🙏 Acknowledgements

- [SMS Spam Collection Dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset)
  — UCI Machine Learning, Esther Kim
- [scikit-learn](https://scikit-learn.org/) for the ML primitives
- [FastAPI](https://fastapi.tiangolo.com/) for the web layer

---

## 📸 Screenshot

<p align="center">
  <img src="/docs/screenshot.png" alt="Spam detector UI showing a message classified as SPAM with 98% confidence" width="720">
  <br>
  <em>A spammy message classified with 98% confidence, with per-token explanations</em>
</p>

---

**Made with ☕ and curiosity.**