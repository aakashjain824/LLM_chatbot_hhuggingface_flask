from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load model and tokenizer
model_name = "facebook/blenderbot-400M-distill"
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_path = r"C:/Users/Zenbook/Documents/models/blenderbot"

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)

conversation_history = []
max_length = 128   # BlenderBot's max input length

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit"]:
        break

    # Add user input to history
    conversation_history.append("User: " + user_input)

    # Keep history as a string
    history_string = "\n".join(conversation_history)

    # Encode input with truncation to avoid exceeding 128 tokens
    inputs = tokenizer(
        history_string,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    )

    outputs = model.generate(**inputs, max_length=128)
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("Bot:", reply)

    # Save bot reply in history
    conversation_history.append("Bot: " + reply)
