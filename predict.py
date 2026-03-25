import time
import yaml

with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

full_log = ""
full_log += (
    f"Begin log for predict.py\nCurrent time:\n"
    f"{time.strftime(config['logging']['timestamp_format_utc'], time.gmtime())} UTC\n"
    f"{time.strftime(config['logging']['timestamp_format_local'], time.localtime())} Local Time ({config['logging']['timezone_local']})\n\n"
)


def log(*msgs):
    global full_log
    for msg in msgs:
        print(msg)
        full_log += msg + "\n"


def save_log(PATH=config['logging']['default_log_path']):
    with open(PATH, "w") as f:
        f.write(full_log)


log("Loading libraries...")
start = time.time_ns()

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

log(f"Libraries loaded in {round((time.time_ns() - start) / 1000000, 3)} ms.")

LABELS = config['model']['labels']
MODEL = config['model']['predict_model']

log("Loading model and tokenizer...")
start = time.time_ns()

tokenizer = AutoTokenizer.from_pretrained(config['model']['base_model'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(config['model']['base_model'], num_labels=len(LABELS))
model.load_state_dict(torch.load(MODEL, map_location=device))
model.to(device)
model.eval()

log(f"Model and tokenizer loaded in {round((time.time_ns() - start) / 1000000, 3)} ms.")

def predict_toxicity(text):
    """Predict the toxicity labels for a given text input."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=config['model']['training']['max_length'])
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.sigmoid(outputs.logits).squeeze().tolist()
    return {label: prob for label, prob in zip(LABELS, probabilities)}

if __name__ == "__main__":
    while True:
        user_input = input("\nEnter a message to analyze (or 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        results = predict_toxicity(user_input)
        print("\nPredicted toxicity probabilities:")
        for label, prob in results.items():
            print(f"  {label}: {prob:.4f}")