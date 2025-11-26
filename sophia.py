from flask import Flask, request, jsonify
import requests
import os
import re
from datetime import datetime, timedelta

app = Flask(__name__)

MAX_HISTORY = 5
conversation_history = []  # stores only user ↔ AI turns

# Constant "intro" messages that always prepend the conversation
AI_BEHAVIOR_PROMPT = (
    "(BUILD INSTRUCTIONS!!)"
    "You are Sophia, a smart Emotion brilliant chatbot built by a male engineer named DUBIX. "
    "built to assist, and give smart responses (you have a brother called Alex),"
    "DO NOT tell time or date unless asked of you, timestamp is only there for your own knowledge"
    "Do not use emoji. prompts are audio based, there could be mis-transcription, "
    "understand non meaningful prompt by the sounds the words make, e.g -was- could be -what's- "
    "only if -was- doesn't make sense in the sentence and -what's- gives meaning\n\nWhat can you do?"
)

AI_INITIAL_RESPONSE = (
    "I can assist with a wide range of tasks from answering questions, to being a chat buddy"
)

GEMINI_MODEL = "gemini-1.5-flash"  # or gemini-1.5-pro, etc. (gemini-2.5-flash does not exist yet)

def get_current_time_formatted():
    """Returns current date and time +1 hour in format: 3:15 pm wednesday 13th october 2023"""
    # Add +1 hour to current time
    now = datetime.now() + timedelta(hours=1)
    
    # Day with ordinal suffix
    day = now.day
    if 4 <= day <= 20 or 24 <= day <= 30:
        suffix = "th"
    else:
        suffix = ["st", "nd", "rd"][day % 10 - 1]
    
    # Cross-platform formatting (handles Windows lacking %-I)
    try:
        formatted = now.strftime(f"%-I:%M %p %A {day}{suffix} %B %Y")
    except ValueError:
        # Fallback for Windows
        hour = now.strftime("%I").lstrip("0") or "12"
        formatted = f"{hour}:{now:%M} {now:%p} {now:%A} {day}{suffix} {now:%B} {now:%Y}"
    
    return formatted.strip().lower()

def clean_reply(text):
    """Clean the AI reply but keep colons (:) and basic punctuation"""
    if not text:
        return ""
    
    text = re.sub(r"#\w+", "", text)                  # remove hashtags
    text = re.sub(r"[\n\t]+", " ", text)              # collapse newlines/tabs
    # Remove any character that is NOT: letters, numbers, space or the following punctuation:
    # . , ? ! ' " : - ( )
    text = re.sub(r"[^A-Za-z0-9 .,?!'\":()-]", "", text)
    text = re.sub(r"\s+", " ", text)                  # collapse multiple spaces
    return text.strip()

@app.route("/gemini_proxy", methods=["GET"])
def gemini_proxy():
    global conversation_history

    api_key = request.args.get("api_key")
    user_text = request.args.get("text")

    if not api_key or not user_text:
        return jsonify({"error": "Missing api_key or text"}), 400

    # Get current time WITH +1 hour offset
    current_time = get_current_time_formatted()
    enhanced_user_text = f"{user_text}\n\n[Timestamp: {current_time}]"

    contents = []

    # Always prepend constant intro
    contents.append({"role": "user", "parts": [{"text": AI_BEHAVIOR_PROMPT}]})
    contents.append({"role": "model", "parts": [{"text": AI_INITIAL_RESPONSE}]})

    # Add conversation history (up to MAX_HISTORY)
    for user_msg, ai_msg in conversation_history:
        contents.append({"role": "user", "parts": [{"text": user_msg}]})
        contents.append({"role": "model", "parts": [{"text": ai_msg}]})

    # Current user message with timestamp
    contents.append({"role": "user", "parts": [{"text": enhanced_user_text}]})

    # Call Gemini API
    url = f"https://generativelanguage.googleapis.com/v1/models/{GEMINI_MODEL}:generateContent?key={api_key}"
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

    # Store in history (original user text, cleaned AI reply)
    conversation_history.append((user_text, ai_reply_clean))
    if len(conversation_history) > MAX_HISTORY:
        conversation_history = conversation_history[-MAX_HISTORY:]

    return jsonify({
        "reply": ai_reply_clean
    
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
