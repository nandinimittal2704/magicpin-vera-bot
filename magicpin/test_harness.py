import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
import judge_simulator

class DummyLLM(judge_simulator.LLMProvider):
    def name(self):
        return "Test Dummy"
    
    def complete(self, prompt, system=None):
        return '''{
            "specificity": 9,
            "specificity_reason": "Fact anchored",
            "category_fit": 9,
            "category_fit_reason": "Tone matched",
            "merchant_fit": 9,
            "merchant_fit_reason": "Personalized",
            "decision_quality": 9,
            "decision_quality_reason": "Trigger clear",
            "engagement_compulsion": 9,
            "engagement_reason": "Single binary CTA",
            "hint": "Well formatted"
        }'''

if __name__ == "__main__":
    judge = judge_simulator.JudgeSimulator(DummyLLM())
    judge.run("all")
