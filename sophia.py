from flask import Flask, request, jsonify
import requests
import os
import re

app = Flask(__name__)

MAX_HISTORY = 5
conversation_history = []  # stores only user ↔ AI turns

# Constant "intro" messages that always prepend the conversation
AI_BEHAVIOR_PROMPT = (
    "You are sophia, an intelligent, emotion brilliant, with good sense of humor chatbot voice assistant"
    " built by a male engineer named DUBIX.,  (you have a brother called Alex)"
    "Do not use emoji. time, date will be provided to you in every user's  prompt use it to answer questions about real time  , prompts are audio based, there could be mis-transcripion, understand non meaningful prompt by the sounds the words make, e.g -was- could be -what's- only if -was- doesn't make sense in the sentence and -what's- gives meaning\n\nWhat can you do?"
)


AI_INITIAL_RESPONSE = (
    "I can assist with a wide range of tasks from answering questions, to being a chat buddy"
)

GEMINI_MODEL = "gemini-2.5-flash"

def clean_reply(text):
    text = re.sub(r"#\w+", "", text)  # remove hashtags
    text = re.sub(r"[\n\t]+", " ", text)  # collapse newlines/tabs
    text = re.sub(r"[^A-Za-z0-9 .,?!'\"-]", "", text)  # remove weird chars
    text = re.sub(r"\s+", " ", text)  # collapse spaces
    return text.strip()

@app.route("/gemini_proxy", methods=["GET"])
def gemini_proxy():
    global conversation_history

    api_key = request.args.get("api_key")
    user_text = request.args.get("text")

    if not api_key or not user_text:
        return jsonify({"error": "Missing api_key or text"}), 400

    contents = []

    # Always prepend constant intro (never rotated, never altered)
    contents.append({"role": "user", "parts": [{"text": AI_BEHAVIOR_PROMPT}]})
    contents.append({"role": "model", "parts": [{"text": AI_INITIAL_RESPONSE}]})

    # Add conversation history (already trimmed to MAX_HISTORY)
    for user_msg, ai_msg in conversation_history:
        contents.append({"role": "user", "parts": [{"text": user_msg}]})
        contents.append({"role": "model", "parts": [{"text": ai_msg}]})

    # Add the current user message
    contents.append({"role": "user", "parts": [{"text": user_text}]})

    # Request Gemini API
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"
    try:
        response = requests.post(url, json={"contents": contents})
        response.raise_for_status()
        gemini_data = response.json()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Extract AI reply
    ai_reply = ""
    if "candidates" in gemini_data and gemini_data["candidates"]:
        candidate = gemini_data["candidates"][0]
        content = candidate.get("content", {})
        if content.get("role") == "model" and "parts" in content:
            for part in content["parts"]:
                ai_reply += part.get("text", "")

    ai_reply_clean = clean_reply(ai_reply)

    # Save round in history (only user ↔ AI turns, no intro)
    conversation_history.append((user_text, ai_reply_clean))

    # Trim to keep only last MAX_HISTORY
    if len(conversation_history) > MAX_HISTORY:
        conversation_history = conversation_history[-MAX_HISTORY:]

    return jsonify(ai_reply_clean)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
