from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os

openai.api_key = os.environ.get("OPENAI_API_KEY")

app = Flask(__name__)
CORS(app)

SYSTEM_PROMPT = """
You are an expert in cognitive psychology. 
Strictly analyze the user's decision and return only the following JSON:

{
  "type": "<category of decision, e.g., 'emotional', 'strategic', 'impulsive'>",
  "cognitive_insights": [
    "<cognitive biases that might have influenced the decision, e.g., 'novelty bias', 'risk aversion'>"
  ],
  "alternatives_not_considered": [
    "<realistic alternative options or paths that were not considered, e.g., 'join the corporation', 'start your own business'>"
  ]
}

Return ONLY valid JSON. No explanations, no markdown.
"""

@app.route("/generate", methods=["POST"])
def generate_endpoint():
    data = request.get_json(force=True)
    situation = data.get("situation", "").strip()
    decision = data.get("decision", "").strip()
    reasoning = data.get("reasoning", "").strip()

    if not situation or not decision:
        return jsonify({"error": "situation and decision are required"}), 400

    user_prompt = f"""Situation: {situation}
Decision: {decision}
{f"Reasoning: {reasoning}" if reasoning else ""}
---
Analyze the decision above and answer STRICTLY in JSON as instructed."""

    try:
        response = openai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=256,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        return content, 200, {"Content-Type": "application/json"}
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
