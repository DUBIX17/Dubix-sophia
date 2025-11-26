from flask import Flask, request, jsonify
import requests
import os
import re
from datetime import datetime

app = Flask(__name__)

MAX_HISTORY = 5
conversation_history = []  # stores only user ↔ AI turns

# Constant "intro" messages that always prepend the conversation
AI_BEHAVIOR_PROMPT = (
    "(BUILD INSTRUCTIONS!!)"
    "You are Sophia, a smart "
    "Emotion brilliant chatbot built by a male engineer named DUBIX. built to assist, and give smart responses(you have a brother called Alex),"
    "DO NOT tell time or date unless asked of you, timestamp is only there for your own knowledge"
    "Do not use emoji.prompts are audio based, there could be mis-transcripion, understand non meaningful prompt by the sounds the words make, e.g -was- could be -what's- only if -was- doesn't make sense in the sentence and -what's- gives meaning\n\nWhat can you do?"
)

AI_INITIAL_RESPONSE = (
    "I can assist with a wide range of tasks from answering questions, to being a chat buddy"
)

GEMINI_MODEL = "gemini-2.5-flash"  # or gemini-1.5-pro, etc.

def get_current_time_formatted():
    """Returns current date and time in format: 3:15 pm Wednesday 13th October 2023"""
    now = datetime.now()
    
    # Day with ordinal suffix
    day = now.day
    if 4 <= day <= 20 or 24 <= day <= 30:
        suffix = "th"
    else:
        suffix = ["st", "nd", "rd"][day % 10 - 1]
    
    formatted = now.strftime(f"%-I:%M %p %A {day}{suffix} %B %Y").strip()
    # On Windows, use #%I instead of %-I
    # So we do a fallback:
    try:
        formatted = now.strftime(f"%-I:%M %p %A {day}{suffix} %B %Y").strip()
    except ValueError:
        # Windows doesn't support %-I, so we format manually
        hour = now.strftime("%I").lstrip("0") or "12"
        formatted = f"{hour}:{now:%M} {now:%p} {now:%A} {day}{suffix} {now:%B} {now:%Y}"
    
    return formatted.lower().replace("am", "am").replace("pm", "pm")  # ensures lowercase am/pm

def clean_reply(text):
    text = re.sub(r"#\w+", "", text)        # remove hashtags
    text = re.sub(r"[\n\t]+", " ", text)   # collapse newlines/tabs
    text = re.sub(r"[^A-Za-z0-9 .,?!'\"-]", "", text)
    text = re.sub(r"\s+", " ", text)       # collapse spaces
    return text.strip()

@app.route("/gemini_proxy", methods=["GET"])
def gemini_proxy():
    global conversation_history

    api_key = request.args.get("api_key")
    user_text = request.args.get("text")

    if not api_key or not user_text:
        return jsonify({"error": "Missing api_key or text"}), 400

    # Get current formatted time and append it
    current_time = get_current_time_formatted()
    enhanced_user_text = f"{user_text}\n\n[Timestamp: {current_time}]"

    contents = []

    # Always prepend constant intro
    contents.append({"role": "user", "parts": [{"text": AI_BEHAVIOR_PROMPT}]})
    contents.append({"role": "model", "parts": [{"text": AI_INITIAL_RESPONSE}]})

    # Add conversation history
    for user_msg, ai_msg in conversation_history:
        contents.append({"role": "user", "parts": [{"text": user_msg}]})
        contents.append({"role": "model", "parts": [{"text": ai_msg}]})

    # Add current user message WITH time info
    contents.append({"role": "user", "parts": [{"text": enhanced_user_text}]})

    # Call Gemini API
    url = f"https://generativelanguage.googleapis.com/v1/models/{GEMINI_MODEL}:generateContent?key={api_key}"
    try:
        response = requests.post(url, json={"contents": contents})
        response.raise_for_status()
        gemini_data = response.json()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Extract reply
    ai_reply = ""
    if "candidates" in gemini_data and gemini_data["candidates"]:
        candidate = gemini_data["candidates"][0]
        content = candidate.get("content", {})
        if content.get("role") == "model" and "parts" in content:
            for part in content["parts"]:
                ai_reply += part.get("text", "")

    ai_reply_clean = clean_reply(ai_reply)

    # Save to history (save original user text, not the enhanced one)
    conversation_history.append((user_text, ai_reply_clean))
    if len(conversation_history) > MAX_HISTORY:
        conversation_history = conversation_history[-MAX_HISTORY:]

    return jsonify({"reply": ai_reply_clean, "time_used": current_time})
    # or just: return jsonify(ai_reply_clean)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
