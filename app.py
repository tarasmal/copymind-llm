from flask import Flask, request, jsonify
from flask_cors import CORS
from outlines import models, generate
from pydantic import BaseModel, Field
import os

class DecisionAnalysis(BaseModel):
    type: str
    cognitive_insights: list[str] = Field(min_items=1)
    alternatives_not_considered: list[str] = Field(min_items=1)

llm = models.transformers(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device="cpu"
)

json_gen = generate.json(llm, DecisionAnalysis)

app = Flask(__name__)
CORS(app)

@app.route("/generate", methods=["POST"])
def generate_endpoint():
    data = request.get_json(force=True)
    situation = data.get("situation", "").strip()
    decision = data.get("decision", "").strip()
    reasoning = data.get("reasoning", "").strip()

    if not situation or not decision:
        return jsonify({"error": "situation and decision are required"}), 400

    prompt = f"""
Situation: {situation}
Decision: {decision}
{f'Reasoning: {reasoning}' if reasoning else ''}
---
Analyze the decision above and answer STRICTLY in this JSON format:

{{
  "type": "<category of decision, e.g., 'emotional', 'strategic', 'impulsive'>",
  "cognitive_insights": [
    "<cognitive biases that might have influenced the decision, e.g., 'novelty bias', 'risk aversion'>"
  ],
  "alternatives_not_considered": [
    "<realistic alternative options or paths that were not considered, e.g., 'join the corporation', 'start your own business'>"
  ]
}}

Example:
{{
  "type": "emotional",
  "cognitive_insights": ["novelty bias", "risk aversion"],
  "alternatives_not_considered": ["join the corporation", "work as a freelancer"]
}}
Return only the JSON object. No comments or explanations.
"""

    try:
        result: DecisionAnalysis = json_gen(prompt, max_tokens=256)
        return jsonify(result.model_dump())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
