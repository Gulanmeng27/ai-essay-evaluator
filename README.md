# 📝 智评 ZhiPing — AI-Powered English Writing Assessment

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-MVP-orange.svg)]()

> **Comparing Algorithmic and Generative Approaches to L2 Writing Assessment**
> 
> A systematic comparison of feature-based Linear Regression vs LLM-based (DeepSeek) scoring on the ELLIPSE Corpus, plus a product-grade demo application.

---

## 🎯 Project Overview

**ZhiPing** tackles a real pain point in English education: writing feedback is slow, expensive, and inconsistent. 

| Problem | Our Solution |
|---------|-------------|
| Teacher takes 15 min per essay | AI scores in 5 seconds |
| Different teachers give different scores | Consistent rubric-based scoring |
| Students wait days for feedback | Instant, anytime feedback |
| Feedback is often vague ("be more specific") | Concrete: quotes errors → explains → gives improved version |

### Key Results

| Model | Pearson r | MSE |
|-------|-----------|-----|
| Linear Regression (5 features) | 0.4438 | 0.3494 |
| **DeepSeek LLM (temp=0.7)** | **0.6276** | **0.3200** |

> 🚀 **LLM outperforms traditional feature-based scoring by 41% in correlation with human scores.**

---

## 🏗️ Project Structure

```
ai-essay-evaluator/
├── src/
│   ├── feature_extraction.py    # Linguistic feature extraction (TTR, MTLD, etc.)
│   ├── model.py                 # Linear Regression training & evaluation
│   └── llm_client.py            # DeepSeek API client (scoring + feedback)
├── demo/
│   └── app.py                   # Gradio demo with radar chart visualization
├── docs/
│   ├── PRD.md                   # Product Requirements Document
│   └── competitive_analysis.md  # Competitive landscape analysis
├── data/
│   └── part1_predictions.csv    # Model predictions on 100-essay dev set
├── .env.example                 # API key template
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set API Key
```bash
cp .env.example .env
# Edit .env and add your DeepSeek API key
```

### 3. Run the Demo
```bash
python demo/app.py
```

### 4. Try It
Open the Gradio URL, paste an English essay, and get instant scoring + feedback.

---

## 📊 How It Works

### Dual Scoring Engine

```
Essay Text
    │
    ├──► Feature Extraction ──► Linear Regression ──► Score + Feature Importance
    │    (num_sent, MTLD, TTR,
    │     num_words, para_div)
    │
    └──► DeepSeek LLM ──► Holistic Score (1.0-5.0) + Detailed Feedback
         (Rubric-prompted,        (Grammar / Vocabulary / Organization / Content)
          temperature=0.7)
```

### Linguistic Features (Linear Regression)

| Feature | Coefficient | Interpretation |
|---------|-------------|----------------|
| `num_sent` | +0.0206 | More sentences → higher score |
| `MTLD` | +0.0182 | Greater lexical diversity → higher score |
| `TTR` | -0.0117 | Artifact of length control |
| `num_words` | -0.0003 | Minimal influence |
| `num_word_div_para` | -0.0006 | Minimal influence |

### LLM Feedback Dimensions

| Dimension | What It Evaluates |
|-----------|-------------------|
| Grammar & Sentence Structure | Error patterns, tense consistency, run-ons |
| Vocabulary Use | Word choice precision, collocations, register |
| Organization & Coherence | Logical flow, cohesion devices, paragraphing |
| Content & Ideas | Argument quality, evidence, persuasiveness |

---

## 🧪 Experimental Design

### Part 1: Automated Essay Scoring
- **Data**: ELLIPSE Corpus (6,500 ELL essays, human-rated)
- **Training**: 200 essays (random seed=42)
- **Evaluation**: 100 essays
- **Baselines**: Mean predictor, Random, Normalized word count
- **LLM Conditions**: DeepSeek-chat-v3 @ temperature = [0.0, 0.3, 0.7, 1.0]

### Part 2: Automated Writing Evaluation
- **5 essays**: low/mid/high proficiency × Chinese L1 learners
- **Models compared**: DeepSeek API vs Qwen2.5-1.5B (local)
- **Feedback accuracy**: 87.9% valid (51/58 items across 5 essays)

---

## 🎨 Product Vision

See `docs/PRD.md` for the full Product Requirements Document.

### Target Users
| User | Pain Point | Value Prop |
|------|-----------|------------|
| College English Teachers | 60 essays × 15 min = 15 hours | Batch scoring in minutes |
| IELTS Candidates | Practice writing without feedback | Instant, rubric-aligned feedback |
| International Schools | High expat teacher costs | AI-assisted scoring consistency |

### Business Model
- **C-end**: ¥29/month subscription
- **B-end**: ¥5,000-20,000/school/year SaaS
- **API**: ¥0.05 per essay evaluation

---

## 🔒 Privacy

We support **local model deployment** (Qwen2.5-1.5B) for schools that require student data to remain on-premise. See the Part 2 notebook for local model setup instructions.

---

## 📄 License

MIT License

---

## 👩‍💻 Author

- **Background**: B.A. in Chinese (Shenzhen University) + M.A. in Linguistics (City University of Hong Kong)
- **Domain**: Computer-Assisted Language Learning (CALL), NLP, AI in Education
- **This project**: Final project for CALL course, extended with product thinking

---

> 💡 *This project demonstrates both technical depth (feature engineering, LLM API integration, experimental design) and product thinking (PRD, competitive analysis, demo application) — bridging the gap between research and product in AI education.*
