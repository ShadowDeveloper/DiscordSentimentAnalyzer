import time

full_log = ""
full_log += (
    f"Begin log for test.py\nCurrent time:\n"
    f"{time.strftime('%d %B, %Y  %H:%M:%S', time.gmtime())} UTC\n"
    f"{time.strftime('%d %B, %Y  %H:%M:%S', time.localtime())} Local Time (CT)\n\n"
)

def log(*msgs):
    global full_log
    for msg in msgs:
        print(msg)
        full_log += msg + "\n"

def save_log(PATH="model.log"):
    with open(PATH, "w") as f:
        f.write(full_log)

log("Loading libraries...")
start = time.time_ns()

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np

log(f"Libraries loaded in {round((time.time_ns() - start) / 1000000, 3)} ms.")

LABELS = [
    "toxicity", "severe_toxicity", "obscene", "threat",
    "insult", "identity_attack", "sexual_explicit",
]
MODEL = "models/distilroberta_model_16926.pt"
BATCH_SIZE = 64
MAX_LENGTH = 256

log("Loading data and tokenizer...")
start = time.time_ns()

tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

log("Loading dataset...")
ds_test  = load_dataset("google/civil_comments", split="test")

labels_array = np.stack([ds_test[l] for l in LABELS], axis=1)
neg = (labels_array < 0.5).sum(0)
pos = (labels_array >= 0.5).sum(0)
pos_weight = torch.tensor(np.log1p(neg / pos.clip(min=1)), dtype=torch.float32)

for l, w in zip(LABELS, pos_weight.tolist()):
    print(f"{l:20s}: {w:.2f}")

log("Preprocessing dataset...")

def preprocess(batch):
    """Tokenize text and combine the 7 label columns into one 'labels' field."""
    tokens = tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)
    tokens["labels"] = [
        [batch[lbl][i] for lbl in LABELS]
        for i in range(len(batch["text"]))
    ]
    return tokens

ds_test = ds_test.map(
    preprocess, batched=True, batch_size=1000,
    remove_columns=ds_test.column_names, desc="Test",
)

collator = DataCollatorWithPadding(tokenizer=tokenizer)

pin = torch.cuda.is_available()
test_loader = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator, pin_memory=pin)

log(f"Data loaded and tokenized in {round((time.time_ns() - start) / 1000000, 3)} ms.")

log("Loading model...")
start = time.time_ns()

model = AutoModelForSequenceClassification.from_pretrained("distilroberta-base", num_labels=len(LABELS))
model.load_state_dict(torch.load(MODEL, map_location=torch.device("cpu")))

log(f"Model loaded in {round((time.time_ns() - start) / 1000000, 3)} ms.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

log("Evaluating model on test set...")
start = time.time_ns()

criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
model.eval()
total_loss = 0.0
with torch.no_grad():
    for batch in tqdm(test_loader, desc='Testing'):
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        labels = batch["labels"].to(device)
        outputs = model(**inputs)
        loss = criterion(outputs.logits, labels.float())
        total_loss += loss.item() * labels.size(0)

avg_loss = total_loss / len(test_loader.dataset)
log(f"Test set evaluation completed in {round((time.time_ns() - start) / 1000000, 3)} ms.")
log(f"Average test loss: {avg_loss:.4f}")
save_log(MODEL.replace(".pt", "") + "_test.log")
