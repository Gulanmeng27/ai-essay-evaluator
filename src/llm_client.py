import os
import time
from openai import OpenAI


ESSAY_SCORING_PROMPT = """You are an expert English language proficiency assessor. Your task is to evaluate an English Language Learner (ELL) essay holistically on a scale from 1.0 to 5.0, using 0.5 increments.

A holistic score reflects the overall effectiveness of the writing, considering vocabulary, grammar, sentence structure, organization, and coherence together, not as a simple average of separate traits.

Use the following detailed rubric to determine the score:

1.0 Minimal
- Vocabulary: Extremely limited; may rely on memorized phrases or single words.
- Grammar: Pervasive, basic errors that severely distort meaning.
- Sentences: Very short, fragmented, or repetitive; no sentence variety.
- Organization/Cohesion: No clear organization; ideas are disconnected.
- Overall: The text is extremely difficult to understand.

2.0 Emerging
- Vocabulary: Basic, high-frequency words; little variety or precision.
- Grammar: Frequent errors in simple structures; meaning is often unclear.
- Sentences: Mostly simple sentences; attempts at compound sentences usually contain errors.
- Organization/Cohesion: Limited logical order; minimal use of basic linking words often misused.
- Overall: The text conveys only a basic idea with significant effort from the reader.

3.0 Developing
- Vocabulary: Adequate for everyday topics; some attempts at less common words may be inaccurate.
- Grammar: Errors are frequent but generally do not obscure meaning; some control of basic structures.
- Sentences: A mix of simple and compound sentences; occasional attempts at complex sentences, though they may contain errors.
- Organization/Cohesion: A basic structure is present; uses some common cohesive devices but may be mechanical or inconsistent.
- Overall: The text is generally understandable despite noticeable errors and limited sophistication.

4.0 Proficient
- Vocabulary: A good range of vocabulary with some precision; occasional awkward word choices.
- Grammar: Good control of basic structures; some errors in complex structures but meaning is clear.
- Sentences: A mix of sentence structures with some success at complex sentences.
- Organization/Cohesion: Clear organization with logical progression of ideas; uses a range of cohesive devices.
- Overall: The text communicates effectively with occasional lapses.

5.0 Advanced
- Vocabulary: Wide range with precise and appropriate word choices.
- Grammar: Strong control of both basic and complex structures; very few errors.
- Sentences: Varied and sophisticated sentence structures used effectively.
- Organization/Cohesion: Well-organized with skillful use of cohesive devices.
- Overall: The text is clear, fluent, and demonstrates strong command of written English.

Output ONLY a single number (e.g., 3.0 or 3.5). Do not include any text, explanation, or punctuation."""


FEEDBACK_PROMPT = """You are a supportive university English writing teacher. Your student is a {proficiency}-level English language learner. Provide detailed, constructive feedback on their essay.

Structure your feedback in four dimensions:

1. Grammar and Sentence Structure: Identify error patterns, quote specific examples from the essay, explain the issue clearly, and provide improved versions.

2. Vocabulary Use: Point out imprecise or inappropriate word choices with quoted examples and better alternatives.

3. Organization and Coherence: Evaluate the essay's structure, logical flow, and use of cohesive devices.

4. Content and Ideas: Assess the quality of arguments, use of evidence, and overall persuasiveness.

Important guidelines:
- Begin by acknowledging strengths. Be specific about what the student did well.
- For each weakness, quote the exact text from the essay.
- Explain the issue in simple, clear language suitable for a language learner.
- Always provide a concrete improved version.
- End with 2-3 actionable tips the student can apply immediately.
- Keep an encouraging, supportive tone throughout."""


class LLMClient:
    def __init__(self, api_key=None, base_url="https://api.deepseek.com"):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY is required. Set it in .env or pass directly.")
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        self.model = "deepseek-chat"

    def score_essay(self, essay_text, temperature=0.7, max_tokens=10):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": ESSAY_SCORING_PROMPT},
                {"role": "user", "content": essay_text},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        raw = response.choices[0].message.content.strip()
        try:
            score = float(raw)
        except ValueError:
            import re
            match = re.search(r"[\d.]+", raw)
            score = float(match.group()) if match else None
        return score, raw, response.usage.completion_tokens

    def generate_feedback(self, essay_text, proficiency="mid", temperature=0.7):
        prompt = FEEDBACK_PROMPT.format(proficiency=proficiency)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": essay_text},
            ],
            max_tokens=1024,
            temperature=temperature,
        )
        return response.choices[0].message.content, response.usage.completion_tokens

    def score_essays_batch(self, essays, temperature=0.7):
        results = []
        for i, essay in enumerate(essays):
            score, raw, tokens = self.score_essay(essay, temperature)
            results.append({"score": score, "raw": raw, "tokens": tokens})
            time.sleep(1.0)
        return results
