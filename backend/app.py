import os
import json
import re
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI


# -------------------------
# Configuration
# -------------------------

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
MODEL = os.environ.get("LLM_MODEL", "llama-3.1-8b-instant")
DATA_FILE = "data.json"

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set")

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)


# -------------------------
# App setup
# -------------------------

app = Flask(__name__, static_folder="../frontend", static_url_path="/")
CORS(app)


# -------------------------
# Database helpers
# -------------------------

def load_db():
    if not os.path.exists(DATA_FILE):
        return {"events": []}
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_db(db):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


def add_event(event_type, text, extra=None):
    db = load_db()
    new_id = db["events"][-1]["id"] + 1 if db["events"] else 1

    entry = {
        "id": new_id,
        "type": event_type,
        "text": text,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "extra": extra or {}
    }

    db["events"].append(entry)
    save_db(db)
    return entry


def parse_json_safe(text):
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"(\{[\s\S]*\})", text)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass
    return None


# -------------------------
# Routes
# -------------------------

@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/history", methods=["GET"])
def get_history():
    return jsonify(load_db())


@app.route("/delete-event", methods=["POST"])
def delete_event():
    payload = request.get_json() or {}
    event_id = payload.get("id")

    if not event_id:
        return jsonify({"error": "id required"}), 400

    db = load_db()
    db["events"] = [e for e in db["events"] if e["id"] != event_id]
    save_db(db)

    return jsonify({"ok": True})


@app.route("/process-text", methods=["POST"])
def process_text():
    payload = request.get_json() or {}
    text = payload.get("text", "").strip()

    if len(text) < 2:
        return jsonify({"error": "Text too short"}), 400

    try:
        extract_prompt = f"""
User text: "{text}"

Return STRICT JSON:
{{
  "transcription_en": "...",
  "symptoms": ["symptom1", "symptom2"],
  "specific_suggestion": "short advice"
}}
"""

        extract_resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You extract structured medical information."},
                {"role": "user", "content": extract_prompt}
            ]
        )

        extraction = parse_json_safe(
            extract_resp.choices[0].message.content
        ) or {
            "transcription_en": text,
            "symptoms": [],
            "specific_suggestion": ""
        }

        db = load_db()
        history_context = [
            {"time": e["timestamp"], "symptom": e["text"]}
            for e in db["events"][-10:]
            if e["type"] == "symptom"
        ]

        triage_prompt = f"""
Patient history:
{json.dumps(history_context)}

Latest complaint:
"{extraction['transcription_en']}"

Return STRICT JSON:
{{
  "specialist": "...",
  "reason": "...",
  "priority": "low | medium | high"
}}
"""

        triage_resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a cautious medical triage assistant."},
                {"role": "user", "content": triage_prompt}
            ]
        )

        triage = parse_json_safe(
            triage_resp.choices[0].message.content
        ) or {
            "specialist": "General Physician",
            "reason": "General evaluation recommended.",
            "priority": "low"
        }

        add_event(
            "symptom",
            extraction["transcription_en"],
            extra={"triage": triage}
        )

        return jsonify({
            "ok": True,
            "extraction": extraction,
            "triage": triage
        })

    except Exception as e:
        print("Text processing error:", e)
        return jsonify({"error": "AI processing failed"}), 500


@app.route("/process-image", methods=["POST"])
def process_image():
    return jsonify({
        "ok": True,
        "triage": {
            "specialist": "General Physician",
            "priority": "medium",
            "reason": "Image analysis is currently unavailable.",
            "visual_observation": "Visual analysis disabled."
        }
    })


@app.route("/generate-soap", methods=["POST"])
def generate_soap():
    try:
        events = load_db()["events"][-30:]

        prompt = f"""
Patient history:
{json.dumps(events)}

Return STRICT JSON with:
patient_summary, critical_alerts[], soap{{subjective, objective, assessment, plan}}
"""

        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You generate clinical SOAP reports."},
                {"role": "user", "content": prompt}
            ]
        )

        parsed = parse_json_safe(resp.choices[0].message.content)
        if not parsed:
            raise ValueError("SOAP parse failed")

        return jsonify({"ok": True, "soap_data": parsed})

    except Exception as e:
        print("SOAP generation error:", e)
        return jsonify({"error": "SOAP generation failed"}), 500


@app.route("/add-history", methods=["POST"])
def add_history():
    payload = request.get_json() or {}
    text = payload.get("text")

    if not text:
        return jsonify({"error": "no text"}), 400

    add_event("history", text)
    return jsonify({"ok": True})


# -------------------------
# Start server
# -------------------------

if __name__ == "__main__":
    if not os.path.exists(DATA_FILE):
        save_db({"events": []})

    print("Backend running on http://127.0.0.1:8000")
    app.run(host="0.0.0.0", port=8000, debug=True)
