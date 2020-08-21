from transformers import AutoModelWithLMHead, AutoTokenizer
from flask import Flask, render_template, request
import argparse

parser = argparse.ArgumentParser(
    description="Process chatbot variables. For help run python bot.py -h"
)
parser.add_argument(
    "-m",
    "--model",
    type=str,
    default="medium",
    help="Size of DialoGPT model"
)

parser.add_argument(
    "-s",
    "--steps",
    type=int,
    default=7,
    help="Number of steps to run the Dialogue System for",
)

args = parser.parse_args()
tokenizer = AutoTokenizer.from_pretrained(f"microsoft/DialoGPT-{args.model}")
model = AutoModelWithLMHead.from_pretrained(f"microsoft/DialoGPT-{args.model}")

app = Flask(__name__)

@app.route('/home')
def index():
    return render_template('index.html')

@app.route('/get')
def get_bot_response():
    user_input = request.args.get('msg')
    new_user_input_ids = tokenizer.encode(
        user_input + tokenizer.eos_token, return_tensors="pt")
    chat_history_ids = model.generate(
        new_user_input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id)
    bot_response = tokenizer.decode(
        chat_history_ids[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)
    bot_response = str(bot_response)
    return bot_response


if __name__ == '__main__':
    app.run(debug=True, port=5002)
