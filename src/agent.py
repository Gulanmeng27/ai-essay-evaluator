"""
AI Writing Evaluation Agent (Simplified Version)
================================================
Agentic workflow for comprehensive essay evaluation
"""

import os
import json
import re


def analyze_grammar(essay: str) -> dict:
    """Analyze grammar errors in an essay"""
    grammar_issues = []
    
    patterns = [
        (r"\b(he|she|it)\b.*?\b(are|were)\b", "Subject-verb agreement error"),
        (r"\b(I|we|they|you)\b.*?\b(is|was)\b", "Subject-verb agreement error"),
        (r"\b(a|an)\s+(?=[aeiouAEIOU])", "Article error: 'a' before vowel"),
        (r"\ban\s+(?=[^aeiouAEIOU])", "Article error: 'an' before consonant"),
    ]
    
    for pattern, desc in patterns:
        matches = re.findall(pattern, essay, re.IGNORECASE)
        if matches:
            grammar_issues.append({"type": desc, "count": len(matches)})
    
    past_tense = re.findall(r"\b(was|were|did|had|went|said)\b", essay, re.IGNORECASE)
    present_tense = re.findall(r"\b(is|are|do|have|go|say)\b", essay, re.IGNORECASE)
    
    if len(past_tense) > 2 and len(present_tense) > 2:
        grammar_issues.append({"type": "Potential tense inconsistency", 
                              "past_count": len(past_tense), "present_count": len(present_tense)})
    
    return {"grammar_issues": grammar_issues, "error_count": len(grammar_issues)}


def analyze_vocabulary(essay: str) -> dict:
    """Analyze vocabulary usage in an essay"""
    words = essay.lower().split()
    if not words:
        return {"error": "Empty essay"}
    
    unique_words = set(words)
    common_words = ["the", "and", "is", "are", "of", "to", "in", "for", "on", "with", "a", "an"]
    common_count = sum(1 for w in words if w in common_words)
    
    ttr = round(len(unique_words) / len(words), 2)
    avg_word_length = round(sum(len(w) for w in words) / len(words), 2)
    common_ratio = round(common_count / len(words), 2)
    
    return {
        "total_words": len(words),
        "unique_words": len(unique_words),
        "ttr": ttr,
        "avg_word_length": avg_word_length,
        "common_word_ratio": common_ratio,
        "vocabulary_richness": "rich" if ttr > 0.5 else "moderate" if ttr > 0.3 else "limited"
    }


def analyze_structure(essay: str) -> dict:
    """Analyze essay structure and coherence"""
    paragraphs = [p.strip() for p in essay.split('\n') if p.strip()]
    sentences = re.split(r'[.!?]+', essay)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    coherence_words = ["however", "therefore", "moreover", "furthermore", "nevertheless", 
                      "in addition", "on the other hand", "for example", "in conclusion"]
    coherence_count = sum(1 for w in essay.lower().split() if w in coherence_words)
    
    return {
        "paragraph_count": len(paragraphs),
        "sentence_count": len(sentences),
        "avg_sentences_per_paragraph": round(len(sentences) / max(len(paragraphs), 1), 2),
        "avg_paragraph_length": round(len(essay) / max(len(paragraphs), 1), 2),
        "coherence_markers_count": coherence_count,
        "structure_quality": "well-organized" if len(paragraphs) >= 3 else "basic"
    }


def generate_practice_exercises(weaknesses: list, proficiency: str = "mid") -> list:
    """Generate personalized practice exercises"""
    exercises = []
    
    if 'grammar' in weaknesses:
        exercises.append({
            "type": "grammar",
            "exercise": "Rewrite sentences with correct subject-verb agreement",
            "examples": ["The group of students was/were studying.", "Neither the teacher nor the students was/were prepared."],
            "difficulty": proficiency
        })
    
    if 'vocabulary' in weaknesses:
        exercises.append({
            "type": "vocabulary",
            "exercise": "Replace basic words with more precise vocabulary",
            "examples": ["The food was very good.", "She said something interesting."],
            "difficulty": proficiency
        })
    
    if 'structure' in weaknesses:
        exercises.append({
            "type": "structure",
            "exercise": "Add coherence markers to improve flow",
            "examples": ["I like reading. I don't have much time.", "The test was hard. I passed."],
            "difficulty": proficiency
        })
    
    return exercises


class EssayEvaluationAgent:
    """Simplified essay evaluation agent without complex LangChain dependencies"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    
    def evaluate(self, essay_text, proficiency="mid"):
        """Evaluate an essay using multi-dimensional analysis"""
        grammar_result = analyze_grammar(essay_text)
        vocab_result = analyze_vocabulary(essay_text)
        structure_result = analyze_structure(essay_text)
        
        weaknesses = []
        if grammar_result["error_count"] > 0:
            weaknesses.append("grammar")
        if vocab_result.get("vocabulary_richness") == "limited":
            weaknesses.append("vocabulary")
        if structure_result.get("structure_quality") == "basic":
            weaknesses.append("structure")
        
        exercises = generate_practice_exercises(weaknesses, proficiency)
        
        score = self._calculate_score(grammar_result, vocab_result, structure_result)
        
        report = f"""
📝 Essay Evaluation Report
=========================

Overall Score: {score}/5.0

---

📚 Grammar Analysis
-------------------
Errors found: {grammar_result['error_count']}
{json.dumps(grammar_result['grammar_issues'], indent=2, ensure_ascii=False) if grammar_result['grammar_issues'] else 'No grammar errors detected'}

---

📖 Vocabulary Analysis
----------------------
Total words: {vocab_result['total_words']}
Unique words: {vocab_result['unique_words']}
Type-Token Ratio: {vocab_result['ttr']}
Vocabulary richness: {vocab_result['vocabulary_richness']}

---

🏛️ Structure Analysis
---------------------
Paragraphs: {structure_result['paragraph_count']}
Sentences: {structure_result['sentence_count']}
Coherence markers: {structure_result['coherence_markers_count']}
Structure quality: {structure_result['structure_quality']}

---

🎯 Personalized Exercises
-------------------------
{self._format_exercises(exercises) if exercises else 'No targeted exercises recommended'}

---

💡 Suggestions for improvement:
{self._generate_suggestions(weaknesses)}
"""
        
        return report
    
    def _calculate_score(self, grammar, vocab, structure):
        """Calculate overall score based on analysis results"""
        grammar_score = max(0, 5 - grammar["error_count"])
        vocab_score = 5 if vocab["vocabulary_richness"] == "rich" else 3 if vocab["vocabulary_richness"] == "moderate" else 1
        structure_score = 5 if structure["structure_quality"] == "well-organized" else 3
        
        avg_score = round((grammar_score + vocab_score + structure_score) / 3, 1)
        return min(5.0, max(1.0, avg_score))
    
    def _format_exercises(self, exercises):
        """Format exercises for display"""
        formatted = []
        for i, ex in enumerate(exercises, 1):
            formatted.append(f"{i}. **{ex['exercise']}**")
            formatted.append(f"   Examples: {', '.join(ex['examples'])}")
            formatted.append(f"   Difficulty: {ex['difficulty']}")
        return "\n".join(formatted)
    
    def _generate_suggestions(self, weaknesses):
        """Generate improvement suggestions based on weaknesses"""
        suggestions = []
        if 'grammar' in weaknesses:
            suggestions.append("• Review subject-verb agreement rules and practice regularly")
        if 'vocabulary' in weaknesses:
            suggestions.append("• Read more to expand vocabulary and learn collocations")
        if 'structure' in weaknesses:
            suggestions.append("• Learn to use coherence markers (however, therefore, moreover)")
        if not weaknesses:
            suggestions.append("• Your essay is well-written! Keep practicing to maintain this level.")
        return "\n".join(suggestions)


if __name__ == "__main__":
    agent = EssayEvaluationAgent()
    essay = """Technology has changed our lives. It make communication easier. People can talk to each other anytime, anywhere."""
    result = agent.evaluate(essay, proficiency="mid")
    print(result)
