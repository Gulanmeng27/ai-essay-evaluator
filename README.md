# 📝 智评 ZhiPing — AI-Powered English Writing Assessment

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-green.svg)]()
[![LangChain](https://img.shields.io/badge/Powered%20by-LangChain-orange.svg)](https://langchain.com)

> **Comparing Algorithmic and Generative Approaches to L2 Writing Assessment**
> 
> A production-grade AI writing evaluation system featuring Transformer architecture, Agentic Workflow, and RAG enhancement.

---

## 🎯 Project Overview

**ZhiPing** is an enterprise-level AI writing assessment platform that addresses real pain points in English education:

| Problem | Solution |
|---------|----------|
| Teacher takes 15 min per essay | AI scores in 5 seconds |
| Inconsistent grading | Rubric-aligned consistent scoring |
| Generic feedback | Personalized, actionable suggestions |
| Knowledge gaps | RAG-enhanced evidence-based evaluation |

### Key Results

| Model | Pearson r | MSE |
|-------|-----------|-----|
| Linear Regression (5 features) | 0.4438 | 0.3494 |
| **DeepSeek LLM (temp=0.7)** | **0.6276** | **0.3200** |

> 🚀 **LLM outperforms traditional scoring by 41% in correlation with human scores.**

---

## 🏗️ Project Structure

```
ai-essay-evaluator/
├── src/
│   ├── __init__.py
│   ├── feature_extraction.py    # Linguistic features (TTR, MTLD, etc.)
│   ├── model.py                 # Linear Regression model
│   ├── llm_client.py            # DeepSeek API client
│   ├── transformer_model.py     # Qwen2.5-7B Transformer wrapper
│   ├── agent.py                 # LangChain Agentic Workflow
│   ├── rag.py                   # RAG Knowledge Enhancement
│   └── api.py                   # FastAPI REST Service
├── demo/
│   └── app.py                   # Gradio demo with Agent/RAG support
├── docs/
│   ├── PRD.md                   # Product Requirements Document
│   └── competitive_analysis.md  # Competitive Analysis
├── knowledge_base/              # RAG knowledge documents
│   ├── scoring_rubric.txt       # ELLIPSE scoring standards
│   ├── grammar_rules.txt        # Grammar rules database
│   └── writing_tips.txt         # Writing best practices
├── data/
│   └── part1_predictions.csv    # Experimental results
├── .env.example                 # API key template
├── .gitignore
└── requirements.txt             # All dependencies
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

### 3. Run Demo
```bash
python demo/app.py
```

### 4. Run API Service
```bash
python -m uvicorn src.api:app --reload
```

---

## 🧠 Cutting-Edge Technologies

### 1. Transformer Architecture
- **Model**: Qwen2.5-7B-Instruct
- **Quantization**: 4-bit NF4 (bitsandbytes)
- **Fine-tuning**: LoRA with PEFT
- **Attention**: Flash Attention 2

```python
from src.transformer_model import TransformerEvaluator

evaluator = TransformerEvaluator("Qwen/Qwen2.5-7B-Instruct")
score = evaluator.score_essay(essay_text)
```

### 2. Agentic Workflow
- **Framework**: LangChain Structured Chat Agent
- **Tools**: Grammar analysis, vocabulary analysis, structure analysis, exercise generation
- **Memory**: Conversation Buffer Memory

```python
from src.agent import EssayEvaluationAgent

agent = EssayEvaluationAgent()
result = agent.evaluate(essay_text, proficiency="mid")
```

### 3. RAG Enhancement
- **Vector Store**: FAISS
- **Embeddings**: text-embedding-3-small
- **Knowledge Base**: ELLIPSE rubric, grammar rules, writing tips

```python
from src.rag import RAGEnhancer

rag = RAGEnhancer()
enhancement = rag.enhance_evaluation(essay_text)
```

### 4. REST API
- **Framework**: FastAPI
- **Features**: Single/batch evaluation, health checks, rubric endpoint
- **Production Ready**: Error handling, Pydantic validation

---

## 📊 Evaluation Pipeline

```
Essay Input
    │
    ├──► Feature Extraction ──► Linear Regression ──► Baseline Score
    │
    ├──► LLM Scoring ──► Holistic Score (1.0-5.0)
    │
    ├──► Agent Analysis ──► Grammar/Vocab/Structure/Content Analysis
    │
    └──► RAG Retrieval ──► Evidence-Based Feedback
            │
            ▼
    Comprehensive Evaluation Report
```

---

## 🎨 Product Vision

### Target Users
| User | Pain Point | Value Proposition |
|------|-----------|-------------------|
| College English Teachers | 60 essays × 15 min = 15 hours | Batch scoring in minutes |
| IELTS Candidates | Practice without feedback | Instant, rubric-aligned evaluation |
| International Schools | High teacher costs | AI-assisted consistent scoring |

### Business Model
- **C-end**: ¥29/month subscription
- **B-end**: ¥5,000-20,000/school/year SaaS
- **API**: ¥0.05 per essay evaluation

---

## 🔒 Privacy & Security

- **Local Deployment**: Qwen2.5-1.5B can run on-premise
- **API Key Protection**: Environment variable management
- **Data Encryption**: HTTPS/TLS for API communications

---

## 📄 License

MIT License

---

## 👩‍💻 Author

- **Background**: B.A. in Chinese (Shenzhen University) + M.A. in Linguistics (CityU)
- **Domain**: Computer-Assisted Language Learning (CALL), NLP, AI in Education
- **Skills**: Python, Transformers, LangChain, RAG, FastAPI, Gradio

---

> 💡 **Project Highlights for Interviews**:
> - ✅ Transformer architecture with LoRA fine-tuning
> - ✅ Agentic workflow with tool selection
> - ✅ RAG-enhanced knowledge retrieval
> - ✅ Production-grade API service
> - ✅ Full-stack demo application
> - ✅ Academic validation with ELLIPSE corpus
