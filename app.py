"""
智评 ZhìPíng — AI-Powered English Writing Assessment Demo
===========================================================
A demo application that:
1. Accepts an English essay input
2. Scores it on a 1.0-5.0 scale using DeepSeek LLM
3. Generates detailed feedback across 4 dimensions
4. Visualizes scores on a radar chart

Usage:
    pip install gradio plotly openai python-dotenv
    python demo/app.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import gradio as gr
except ImportError:
    print("Install gradio first: pip install gradio")
    sys.exit(1)

try:
    import plotly.graph_objects as go
except ImportError:
    print("Install plotly first: pip install plotly")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass

from llm_client import LLMClient


DIMENSIONS = ["Grammar", "Vocabulary", "Organization", "Content"]
DIMENSION_KEYS = ["grammar_score", "vocabulary_score", "organization_score", "content_score"]

client = None


def get_client():
    global client
    if client is None:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            return None
        client = LLMClient(api_key=api_key)
    return client


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


def analyze_with_ai(essay_text, proficiency, api_key_input):
    if api_key_input:
        os.environ["DEEPSEEK_API_KEY"] = api_key_input

    c = get_client()
    if c is None:
        return (
            0, "Please enter a valid DeepSeek API key.",
            go.Figure(),
            "Waiting for API key...",
            "Waiting for API key...",
            "Waiting for API key...",
            "Waiting for API key...",
        )

    try:
        score, raw, _ = c.score_essay(essay_text, temperature=0.7)
        feedback, _ = c.generate_feedback(essay_text, proficiency=proficiency)

        per_dim_scores = [3.0, 3.0, 3.0, 3.0]
        if score:
            import re
            dim_map = {
                "grammar": "grammar_score",
                "vocabulary": "vocabulary_score",
                "organization": "organization_score",
                "content": "content_score",
            }
            for dim in dim_map:
                pattern = rf"{dim}.*?([\d.]+)"
                match = re.search(pattern, feedback, re.IGNORECASE)
                if match:
                    val = float(match.group(1))
                    if 1 <= val <= 5:
                        idx = ["grammar", "vocabulary", "organization", "content"].index(dim)
                        per_dim_scores[idx] = val

        radar = create_radar_chart(per_dim_scores)

        sections = parse_feedback_sections(feedback)

        return (
            score if score else 0,
            feedback,
            radar,
            sections.get("grammar", "Not provided"),
            sections.get("vocabulary", "Not provided"),
            sections.get("organization", "Not provided"),
            sections.get("content", "Not provided"),
        )
    except Exception as e:
        return 0, f"Error: {e}", go.Figure(), "Error", "Error", "Error", "Error"


def parse_feedback_sections(feedback):
    sections = {}
    current_section = None
    current_text = []
    section_keywords = {
        "grammar": ["grammar", "sentence structure"],
        "vocabulary": ["vocabulary"],
        "organization": ["organization", "coherence"],
        "content": ["content", "ideas"],
    }

    for line in feedback.split("\n"):
        line_lower = line.lower().strip()
        for key, keywords in section_keywords.items():
            for kw in keywords:
                if kw in line_lower and (line_lower.startswith("#") or line_lower.startswith("**") or line_lower[0].isdigit()):
                    if current_section and current_text:
                        sections[current_section] = "\n".join(current_text).strip()
                    current_section = key
                    current_text = []
                    break
            if current_section:
                break
        else:
            if current_section:
                current_text.append(line)

    if current_section and current_text:
        sections[current_section] = "\n".join(current_text).strip()

    return sections


CSS = """
.gradio-container { max-width: 1100px !important; margin: auto !important; }
.header { text-align: center; padding: 20px 0; }
.score-display { font-size: 64px; font-weight: bold; text-align: center; color: #6366f1; }
.label { font-size: 14px; color: #6b7280; text-align: center; }
"""

with gr.Blocks(css=CSS, title="ZhiPing — AI Writing Assessment") as demo:
    gr.HTML("""
    <div class="header">
        <h1>📝 智评 ZhiPing</h1>
        <p>AI-Powered English Writing Assessment · LLM vs Traditional Feature-Based Scoring</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            api_key = gr.Textbox(
                label="DeepSeek API Key",
                placeholder="sk-...",
                type="password",
                value=os.environ.get("DEEPSEEK_API_KEY", ""),
            )
            essay_input = gr.Textbox(
                label="Paste an English Essay",
                placeholder="Type or paste an English essay here...",
                lines=12,
            )
            proficiency = gr.Dropdown(
                label="Student Proficiency Level",
                choices=["low", "mid", "high"],
                value="mid",
            )
            analyze_btn = gr.Button("🔍 Analyze Essay", variant="primary", size="lg")

        with gr.Column(scale=1):
            with gr.Row():
                with gr.Column(scale=1):
                    overall_score = gr.Number(label="Overall Score (1.0–5.0)", precision=1)
                with gr.Column(scale=2):
                    radar_chart = gr.Plot(label="Dimension Breakdown")

    with gr.Accordion("📊 Detailed AI Feedback", open=True):
        feedback_box = gr.Textbox(label="Full AI Feedback", lines=18)

    with gr.Accordion("📋 Dimension Details", open=False):
        with gr.Row():
            with gr.Column():
                grammar_feedback = gr.Textbox(label="Grammar & Sentence Structure", lines=6)
            with gr.Column():
                vocab_feedback = gr.Textbox(label="Vocabulary Use", lines=6)
        with gr.Row():
            with gr.Column():
                org_feedback = gr.Textbox(label="Organization & Coherence", lines=6)
            with gr.Column():
                content_feedback = gr.Textbox(label="Content & Ideas", lines=6)

    analyze_btn.click(
        fn=analyze_with_ai,
        inputs=[essay_input, proficiency, api_key],
        outputs=[overall_score, feedback_box, radar_chart, grammar_feedback, vocab_feedback, org_feedback, content_feedback],
    )

if __name__ == "__main__":
    print("=" * 60)
    print("  ZhìPíng — AI Writing Assessment Demo")
    print("  Starting Gradio interface...")
    print("=" * 60)
    demo.launch(share=False)
