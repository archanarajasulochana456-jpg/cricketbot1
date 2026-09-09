import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from google import genai

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

app = Flask(__name__)

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-3.1-flash-lite"

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not configured in the .env file.")

client = genai.Client(api_key=API_KEY)

with open(BASE_DIR / "chatbot_config.txt", "r", encoding="utf-8") as file:
    SYSTEM_PROMPT = file.read().strip()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({"error": "Please enter a message."}), 400

    try:
        prompt = f"{SYSTEM_PROMPT}\n\nUser message:\n{message}"
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )

        reply = (response.text or "").strip()

        if not reply:
            reply = "I couldn't generate a response. Please try again."

        return jsonify({"reply": reply})

    except Exception:
        return jsonify({
            "error": "Something went wrong while contacting Gemini. Please try again."
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
