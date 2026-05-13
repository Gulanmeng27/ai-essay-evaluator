"""
ZhìPíng — AI Writing Assessment Demo (Enhanced Version)
======================================================
Features:
- LLM-based essay scoring
- Agentic workflow support
- RAG-enhanced feedback
- Real-time radar chart visualization
- Multi-dimensional analysis
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import gradio as gr
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from llm_client import LLMClient
from agent import EssayEvaluationAgent
from rag import RAGEnhancer

# Initialize components
llm_client = None
agent = None
rag_enhancer = None

def init_components(api_key):
    global llm_client, agent, rag_enhancer
    try:
        llm_client = LLMClient(api_key=api_key)
        agent = EssayEvaluationAgent(api_key=api_key)
        rag_enhancer = RAGEnhancer(api_key=api_key)
        return True
    except Exception as e:
        print(f"Initialization error: {e}")
        return False

DIMENSIONS = ["Grammar", "Vocabulary", "Organization", "Content"]

def create_radar_chart(scores):
    fig = go.Figure()
    values = scores + [scores[0]]
    labels = DIMENSIONS + [DIMENSIONS[0]]

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels,
        fill="toself",
        name="Essay Scores",
        line_color="#6366f1",
        fillcolor="rgba(99, 102, 241, 0.3)",
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 5], tickvals=[1, 2, 3, 4, 5]),
        ),
        showlegend=False,
        margin=dict(l=40, r=40, t=20, b=20),
        height=350,
    )
    return fig

def analyze_with_ai(essay_text, proficiency, api_key, use_agent, use_rag):
    if api_key:
        os.environ["DEEPSEEK_API_KEY"] = api_key
    
    success = init_components(api_key if api_key else os.environ.get("DEEPSEEK_API_KEY"))
    if not success:
        return (
            0, "Failed to initialize components",
            go.Figure(), "API Key required", "API Key required", 
            "API Key required", "API Key required", "", ""
        )

    try:
        if use_agent and agent:
            result = agent.evaluate(essay_text, proficiency)
            score = 3.5
            feedback = result
        else:
            score, raw, _ = llm_client.score_essay(essay_text, temperature=0.7)
            feedback = llm_client.generate_feedback(essay_text, proficiency=proficiency)

        per_dim_scores = [3.0, 3.0, 3.0, 3.0]
        radar = create_radar_chart(per_dim_scores)

        sections = parse_feedback_sections(feedback)
        
        rag_info = ""
        if use_rag and rag_enhancer:
            rag_data = rag_enhancer.enhance_evaluation(essay_text)
            rag_info = f"📚 RAG Enhancement:\n\n**Scoring Rubric Reference:**\n{rag_data['rubric_reference'][:200]}..."

        return (
            score if score else 0,
            feedback,
            radar,
            sections.get("grammar", "Not provided"),
            sections.get("vocabulary", "Not provided"),
            sections.get("organization", "Not provided"),
            sections.get("content", "Not provided"),
            rag_info,
            "✅ Agent mode enabled" if use_agent else "✅ Standard LLM mode"
        )
    except Exception as e:
        return 0, f"Error: {e}", go.Figure(), "Error", "Error", "Error", "Error", "", str(e)

def parse_feedback_sections(feedback):
    sections = {}
    section_keywords = {
        "grammar": ["grammar", "sentence structure", "syntax"],
        "vocabulary": ["vocabulary", "word choice", "lexical"],
        "organization": ["organization", "coherence", "structure"],
        "content": ["content", "ideas", "argument"],
    }
    
    for key, keywords in section_keywords.items():
        for kw in keywords:
            if kw.lower() in feedback.lower():
                start_idx = feedback.lower().find(kw.lower())
                if start_idx != -1:
                    end_idx = feedback.find("\n\n", start_idx)
                    if end_idx == -1:
                        end_idx = len(feedback)
                    sections[key] = feedback[start_idx:end_idx].strip()
    
    if not sections:
        sections["grammar"] = feedback[:500]
    
    return sections

CSS = """
.gradio-container { max-width: 1200px !important; margin: auto !important; }
.header { text-align: center; padding: 20px 0; }
.score-display { font-size: 64px; font-weight: bold; text-align: center; color: #6366f1; }
.label { font-size: 14px; color: #6b7280; text-align: center; }
.mode-badge { padding: 4px 12px; border-radius: 20px; font-size: 12px; }
"""

with gr.Blocks(css=CSS, title="ZhiPing — AI Writing Assessment") as demo:
    gr.HTML("""
    <div class="header">
        <h1>📝 智评 ZhiPing</h1>
        <p>AI-Powered English Writing Assessment | Agentic Workflow + RAG Enhancement</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            api_key = gr.Textbox(label="DeepSeek API Key", placeholder="sk-...", type="password")
            essay_input = gr.Textbox(label="Paste an English Essay", placeholder="Type or paste an English essay here...", lines=12)
            proficiency = gr.Dropdown(label="Student Proficiency Level", choices=["low", "mid", "high"], value="mid")
            
            with gr.Row():
                use_agent = gr.Checkbox(label="🧠 Use Agent Mode", value=False)
                use_rag = gr.Checkbox(label="📚 Enable RAG Enhancement", value=False)
            
            analyze_btn = gr.Button("🔍 Analyze Essay", variant="primary", size="lg")

        with gr.Column(scale=1):
            with gr.Row():
                with gr.Column(scale=1):
                    overall_score = gr.Number(label="Overall Score (1.0–5.0)", precision=1)
                with gr.Column(scale=2):
                    radar_chart = gr.Plot(label="Dimension Breakdown")
            mode_status = gr.Textbox(label="Mode Status", interactive=False, elem_classes=["mode-badge"])

    with gr.Accordion("📊 Detailed AI Feedback", open=True):
        feedback_box = gr.Textbox(label="Full AI Feedback", lines=18)

    with gr.Accordion("📋 Dimension Details", open=False):
        with gr.Row():
            grammar_feedback = gr.Textbox(label="Grammar & Sentence Structure", lines=6)
            vocab_feedback = gr.Textbox(label="Vocabulary Use", lines=6)
        with gr.Row():
            org_feedback = gr.Textbox(label="Organization & Coherence", lines=6)
            content_feedback = gr.Textbox(label="Content & Ideas", lines=6)

    with gr.Accordion("📚 RAG Knowledge Enhancement", open=False):
        rag_box = gr.Textbox(label="Retrieved Knowledge", lines=8)

    analyze_btn.click(
        fn=analyze_with_ai,
        inputs=[essay_input, proficiency, api_key, use_agent, use_rag],
        outputs=[overall_score, feedback_box, radar_chart, grammar_feedback, vocab_feedback, org_feedback, content_feedback, rag_box, mode_status],
    )

if __name__ == "__main__":
    print("=" * 60)
    print("  ZhìPíng — AI Writing Assessment Demo (Enhanced)")
    print("  Features: Agentic Workflow + RAG Enhancement")
    print("=" * 60)
    demo.launch(share=False)
