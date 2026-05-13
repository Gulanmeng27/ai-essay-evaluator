"""
RAG Enhancer (Simplified Version)
=================================
Retrieval-Augmented Generation for essay evaluation
"""

import os
import json
import re


class RAGEnhancer:
    """Simplified RAG enhancer for essay evaluation"""
    
    def __init__(self, api_key=None, knowledge_base_path="knowledge_base"):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.knowledge_base = self._load_knowledge_base()
    
    def _load_knowledge_base(self):
        """Load scoring rubric and writing guidelines"""
        return {
            "rubric": {
                "1.0": "Basic: Severe difficulties in grammar, vocabulary, and coherence. Minimal sentence variety.",
                "2.0": "Limited: Frequent errors, basic vocabulary, weak structure.",
                "3.0": "Adequate: Some errors, reasonable vocabulary, basic structure.",
                "4.0": "Good: Few errors, varied vocabulary, good organization.",
                "5.0": "Excellent: Virtually error-free, sophisticated vocabulary, well-organized."
            },
            "grammar_rules": [
                "Subject-verb agreement: Singular subjects take singular verbs",
                "Tense consistency: Maintain consistent verb tense throughout",
                "Article usage: Use 'a' before consonants, 'an' before vowels",
                "Sentence structure: Vary sentence length and structure"
            ],
            "writing_tips": [
                "Introduction: State your main idea clearly",
                "Body: Use topic sentences for each paragraph",
                "Conclusion: Summarize key points",
                "Coherence: Use transition words (however, therefore, moreover)",
                "Vocabulary: Use precise words instead of vague ones",
                "Conciseness: Avoid unnecessary repetition"
            ],
            "common_errors": [
                "Subject-verb agreement errors",
                "Incorrect article usage",
                "Run-on sentences",
                "Sentence fragments",
                "Word choice errors",
                "Tense inconsistency"
            ]
        }
    
    def retrieve_knowledge(self, query):
        """Retrieve relevant knowledge based on query"""
        query_lower = query.lower()
        
        if "rubric" in query_lower or "score" in query_lower:
            return {"type": "rubric", "data": self.knowledge_base["rubric"]}
        
        if "grammar" in query_lower or "error" in query_lower:
            return {"type": "grammar", "data": self.knowledge_base["grammar_rules"]}
        
        if "structure" in query_lower or "writing" in query_lower or "tips" in query_lower:
            return {"type": "writing_tips", "data": self.knowledge_base["writing_tips"]}
        
        if "common" in query_lower or "mistake" in query_lower:
            return {"type": "common_errors", "data": self.knowledge_base["common_errors"]}
        
        return {"type": "general", "data": self.knowledge_base}
    
    def enhance_evaluation(self, essay_text):
        """Enhance essay evaluation with knowledge retrieval"""
        rubric = self.retrieve_knowledge("scoring rubric")
        grammar_tips = self.retrieve_knowledge("grammar rules")
        writing_tips = self.retrieve_knowledge("writing tips")
        
        return {
            "rubric_reference": rubric["data"],
            "grammar_guidance": grammar_tips["data"],
            "writing_tips": writing_tips["data"]
        }
    
    def format_knowledge_for_feedback(self, essay_text):
        """Format retrieved knowledge for feedback generation"""
        enhancement = self.enhance_evaluation(essay_text)
        
        feedback = "\n📚 Scoring Rubric Reference:\n"
        for score, desc in enhancement["rubric_reference"].items():
            feedback += f"  {score}: {desc}\n"
        
        feedback += "\n✅ Grammar Guidelines:\n"
        for i, rule in enumerate(enhancement["grammar_guidance"], 1):
            feedback += f"  {i}. {rule}\n"
        
        feedback += "\n💡 Writing Tips:\n"
        for i, tip in enumerate(enhancement["writing_tips"], 1):
            feedback += f"  {i}. {tip}\n"
        
        return feedback


if __name__ == "__main__":
    rag = RAGEnhancer()
    essay = "Technology has changed our lives. It make communication easier."
    result = rag.enhance_evaluation(essay)
    print(json.dumps(result, indent=2, ensure_ascii=False))
