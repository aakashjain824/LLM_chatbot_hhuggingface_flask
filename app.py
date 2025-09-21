from flask import Flask, request
from flask_cors import CORS
import json
from chatbot import model, tokenizer, conversation_history, max_length
model.config.use_cache = False
app = Flask(__name__)
CORS(app)

@app.route('/chatbot', methods=['POST'])
def chatbot_api():
    # Get request body as JSON
    data = request.json
    user_input = data.get('message', '')

    if not user_input:
        return json.dumps({'error': 'No input provided'}), 400

    # Add user input to conversation history
    conversation_history.append("User: " + user_input)

    # Keep only the last 4 exchanges to avoid context overflow
    short_history = conversation_history[-4:]
    history_string = "\n".join(short_history)

    # Tokenize with truncation
    inputs = tokenizer(
        history_string,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    )

    # Generate a reply
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,        # limit bot's response length
        max_length=128,           # prevent sequence overflow
        use_cache=False,          # avoid cache index issues
        no_repeat_ngram_size=3    # reduce repetition
    )

    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Add bot reply to history
    conversation_history.append("Bot: " + reply)

    return json.dumps({'reply': reply})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
