"""
JEE Augmented Tutor — Flask backend
Serves the frontend and exposes two REST endpoints:
  POST /api/search  → retrieve similar PYQs
  POST /api/tutor   → retrieve PYQs + AI explanation via Mistral
"""
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
try:
    import ollama
except ImportError:
    ollama = None
# from mistralai import Mistral  # API version — comment out when using local LLM
from rag import build_index, build_llm_context, format_results, retrieve
load_dotenv()
app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)  # allow requests from the HTML file during local dev
USER_HISTORY = []
# --------------------------------------------------------------------------- #
# Startup: build / load the vector index once
# --------------------------------------------------------------------------- #
print("Initialising RAG index...")
INDEX = build_index()
print("RAG index ready.")
# mistral_client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))  # API version
# --------------------------------------------------------------------------- #
# Helper
# --------------------------------------------------------------------------- #
def _parse_body() -> dict:
    data = request.get_json(silent=True) or {}
    return data
# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    """Serve the frontend."""
    return send_from_directory("frontend", "index.html")
@app.route("/api/search", methods=["POST"])
def search():
    """
    Body: { query, top_k?, threshold?, subject?, year? }
    Returns: { results: [...], count: int }
    """
    data = _parse_body()
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "query is required"}), 400
    top_k = int(data.get("top_k", 20))
    threshold = float(data.get("threshold", 0.04))
    subject_filter = data.get("subject", "").lower()
    year_filter = str(data.get("year", ""))
    results = retrieve(INDEX, query, top_k=top_k * 2, threshold=threshold)
    formatted = format_results(results)
    # Optional client-side filters
    if subject_filter:
        formatted = [r for r in formatted if r["subject"].lower() == subject_filter]
    if year_filter:
        formatted = [r for r in formatted if str(r["year"]) == year_filter]
    formatted = formatted[:top_k]
    return jsonify({"results": formatted, "count": len(formatted)})
@app.route("/api/tutor", methods=["POST"])
def tutor():
    """
    Body: { query, top_k?, threshold? }
    Returns: { results: [...], explanation: str }
    """
    data = _parse_body()
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "query is required"}), 400
    top_k = int(data.get("top_k", 20))
    threshold = float(data.get("threshold", 0.04))
    results = retrieve(INDEX, query, top_k=top_k, threshold=threshold)
    formatted = format_results(results)
    context = build_llm_context(formatted, max_questions=15)
    prompt = (
        f"You are an expert tutor for JEE/NEET exam preparation.\n\n"
        f"The student asked:\n{query}\n\n"
        f"Here are {min(len(formatted), 15)} relevant past exam questions:\n\n"
        f"{context}\n\n"
        f"Using these questions, answer the student's doubt clearly and completely."
        f"Cover: key concepts, important formulas or techniques, and patterns you "
        f"notice across these questions. Be concise and educational."
    )
    # ── API version (comment out when using local LLM) ──
    # try:
    #     message = mistral_client.chat.complete(
    #         model="mistral-small-latest",
    #         max_tokens=1024,
    #         messages=[{"role": "user", "content": prompt}],
    #     )
    #     explanation = message.choices[0].message.content
    # except Exception as e:
    #     explanation = f"[AI explanation unavailable: {e}]"
    # ── Local LLM version ──
    try:
        from rag import local_llm
        response = local_llm.complete(prompt)
        explanation = str(response)
    except Exception as e:
        explanation = f"[AI explanation unavailable: {e}]"
    return jsonify({"results": formatted, "count": len(formatted), "explanation": explanation})
@app.route("/api/check_answer", methods=["POST"])
def check_answer():
    if ollama is None:
        return jsonify({"result": "error", "error": "ollama library is not installed"}), 500
    data = request.get_json()
    user_ans = data.get("user_answer", "")
    correct_ans = data.get("correct_answer", "")
    answer_desc = data.get("answer_description", "")
    question = data.get("question", "")
    topic = data.get("topic", "unknown")
    prompt = f"""
You are a easy but intelligent answer evaluator for JEE-level questions.
Question:
{question}
Correct Answer:
{correct_ans}
Official Solution:
{answer_desc}
Student Answer:
{user_ans}
Rules:
- If the student answer is conceptually correct or equivalent → reply ONLY: Y
- If incorrect or missing key idea → reply ONLY: N
- Ignore small formatting differences
- Accept if rather than answer student gave answer option
- Accept equivalent forms (e.g., 2π = 6.28, etc.)
- Do NOT explain anything
Final Output (only one letter):"""
    try:
        response = ollama.chat(
            model="phi3",
            messages=[{"role": "user", "content": prompt}]
        )
        result = response["message"]["content"].strip().upper()
        USER_HISTORY.append({
            "topic": topic,
            "correct": result.startswith("Y")
        })
        if result.startswith("Y"):
            return jsonify({"result": "Y"})
        else:
            return jsonify({"result": "N"})
    except Exception as e:
        return jsonify({"result": "error", "error": str(e)})
@app.route("/api/stats", methods=["GET"])
def stats():
    total = len(USER_HISTORY)
    correct = sum(1 for x in USER_HISTORY if x["correct"])
    wrong = total - correct
    topic_stats = {}
    for entry in USER_HISTORY:
        t = entry["topic"] or "unknown"
        if t not in topic_stats:
            topic_stats[t] = {"correct": 0, "wrong": 0}
        if entry["correct"]:
            topic_stats[t]["correct"] += 1
        else:
            topic_stats[t]["wrong"] += 1
    return jsonify({
        "total": total,
        "correct": correct,
        "wrong": wrong,
        "topics": topic_stats
    })
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, port=port)
