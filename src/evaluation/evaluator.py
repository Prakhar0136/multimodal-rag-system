# src/evaluation/evaluator.py
import json
import re
from groq import Groq
from typing import Dict, Any
from src.config import Config

class PipelineEvaluator:
    def __init__(self):
        self.groq_client = Groq(api_key=Config.GROQ_API_KEY)
        # Using a fast reasoning model for grading
        self.eval_model = Config.TEXT_MODEL 

    def _extract_json(self, raw_text: str) -> Dict[str, int]:
        """Safely extracts JSON from LLM outputs, bypassing markdown wrappers."""
        try:
            match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return json.loads(raw_text)
        except Exception as e:
            print(f"⚠️ Failed to parse evaluator JSON: {e}")
            return {"faithfulness": 0, "relevance": 0, "context_recall": 0}

    def evaluate_turn(self, query: str, context_used: list, generated_answer: str, expected_facts: list) -> Dict[str, int]:
        """Uses Llama-3 to grade the pipeline's execution on a scale of 0 or 1 for core metrics."""
        
        context_text = "\n".join([c.get("text", "") for c in context_used]) if context_used else "No context retrieved."
        
        evaluation_prompt = f"""
        You are an impartial, strict scoring system evaluating a RAG (Retrieval-Augmented Generation) pipeline.
        
        Evaluate the following execution based on three metrics. Output your scores strictly as a JSON object with keys: "faithfulness", "relevance", "context_recall".
        Score each metric as either 1 (Pass) or 0 (Fail).

        INPUT DATA:
        User Query: {query}
        Retrieved Context: {context_text}
        Generated Answer: {generated_answer}
        Expected Facts to be present: {expected_facts}

        METRICS:
        1. faithfulness: Is the Generated Answer strictly derived from the Retrieved Context? (Score 1 if yes, 0 if it contains outside hallucinations).
        2. relevance: Does the Generated Answer directly address the User Query? (Score 1 if yes, 0 if it dodges the question).
        3. context_recall: Does the Retrieved Context contain the necessary information to address the Expected Facts? (Score 1 if yes, 0 if the context is missing the data).

        OUTPUT FORMAT:
        Return ONLY valid JSON. No explanations.
        """

        try:
            completion = self.groq_client.chat.completions.create(
                model=self.eval_model,
                messages=[{"role": "user", "content": evaluation_prompt}],
                temperature=0.0,
                max_tokens=150
            )
            raw_eval = completion.choices[0].message.content
            return self._extract_json(raw_eval)
        except Exception as e:
            print(f"⚠️ Evaluator LLM failed: {e}")
            return {"faithfulness": 0, "relevance": 0, "context_recall": 0}