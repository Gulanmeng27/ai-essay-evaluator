"""
Transformer Model Wrapper for Essay Evaluation
==============================================
Supports:
- Qwen2.5-7B/14B Instruct models
- LoRA fine-tuning
- Efficient inference with bitsandbytes quantization
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, PeftModel
import os


class TransformerEvaluator:
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", use_lora=False, lora_weights=None):
        self.model_name = model_name
        self.use_lora = use_lora
        self.lora_weights = lora_weights
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self._init_tokenizer()
        self._init_model()
    
    def _init_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _init_model(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if self.device == "cuda" else "eager"
        )
        
        if self.use_lora and self.lora_weights:
            self.model = PeftModel.from_pretrained(self.model, self.lora_weights)
    
    def configure_lora(self):
        """Configure LoRA for fine-tuning"""
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def generate(self, prompt, max_tokens=1024, temperature=0.7):
        """Generate response using the model"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        return response[len(prompt):].strip()
    
    def score_essay(self, essay_text):
        """Score an essay using the Transformer model"""
        prompt = f"""
You are an expert English writing evaluator. Score this essay on a scale of 1.0-5.0:

Essay:
{essay_text}

Output ONLY a single numerical score (e.g., 3.5).
"""
        result = self.generate(prompt, max_tokens=50, temperature=0.1)
        try:
            return float(result.strip())
        except ValueError:
            return 3.0
    
    def generate_feedback(self, essay_text, proficiency="mid"):
        """Generate detailed feedback for an essay"""
        prompt = f"""
You are a supportive English writing teacher for {proficiency}-level learners.
Provide detailed feedback on this essay covering:
1. Grammar and sentence structure
2. Vocabulary usage
3. Organization and coherence
4. Content and ideas

Essay:
{essay_text}

Provide constructive feedback with specific examples from the essay.
"""
        return self.generate(prompt, max_tokens=1024, temperature=0.7)
    
    def save_lora_weights(self, path):
        """Save LoRA weights after fine-tuning"""
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(path)


# Example usage
if __name__ == "__main__":
    evaluator = TransformerEvaluator("Qwen/Qwen2.5-7B-Instruct")
    essay = "Technology has changed our lives in many ways. It makes communication easier and faster."
    score = evaluator.score_essay(essay)
    print(f"Score: {score}")
