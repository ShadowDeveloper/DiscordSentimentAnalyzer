import time
import yaml

# Load configuration
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

full_log = ""
full_log += (
    f"Begin log for train.py\nCurrent time:\n"
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
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from torch.utils.data import DataLoader
from ignite.contrib.handlers import PiecewiseLinear
from ignite.handlers import EarlyStopping, ProgressBar, ModelCheckpoint
from ignite.engine import Engine, Events
from ignite.metrics import Loss
import numpy as np

log(f"Libraries loaded in {round((time.time_ns() - start) / 1000000, 3)} ms.")

LABELS = config['model']['labels']
MODEL = config['model']['base_model']
N_EPOCHS = config['model']['training']['n_epochs']
BATCH_SIZE = config['model']['training']['batch_size']
PATIENCE = config['model']['training']['patience']
MAX_LENGTH = config['model']['training']['max_length']

log("Loading data and tokenizer...")
start = time.time_ns()

tokenizer = AutoTokenizer.from_pretrained(MODEL)

log("Loading dataset...")
ds_train = load_dataset(config['dataset']['name'], split=config['dataset']['splits']['train'])
ds_val   = load_dataset(config['dataset']['name'], split=config['dataset']['splits']['validation'])
ds_test  = load_dataset(config['dataset']['name'], split=config['dataset']['splits']['test'])

labels_array = np.stack([ds_train[l] for l in LABELS], axis=1)
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

ds_train = ds_train.map(
    preprocess, batched=True, batch_size=1000,
    remove_columns=ds_train.column_names, desc="Train",
)
ds_val = ds_val.map(
    preprocess, batched=True, batch_size=1000,
    remove_columns=ds_val.column_names, desc="Val",
)
ds_test = ds_test.map(
    preprocess, batched=True, batch_size=1000,
    remove_columns=ds_test.column_names, desc="Test",
)

collator = DataCollatorWithPadding(tokenizer=tokenizer)

pin = torch.cuda.is_available()
train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator, pin_memory=pin)
val_loader = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator, pin_memory=pin)
test_loader = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator, pin_memory=pin)

log(f"Data loaded and tokenized in {round((time.time_ns() - start) / 1000000, 3)} ms.")

log("Loading model...")
start = time.time_ns()

model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(LABELS))

log(f"Model loaded in {round((time.time_ns() - start) / 1000000, 3)} ms.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

num_training_steps = N_EPOCHS * len(train_loader)
lr_scheduler = PiecewiseLinear(
    optimizer, param_name="lr",
    milestones_values=[(0, 5e-5), (num_training_steps, 0.0)],
)

loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

def train_step(engine, batch):
    model.train()
    labels = batch.pop("labels").float().to(device)
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model(**batch)
    loss = loss_fn(outputs.logits, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()

trainer = Engine(train_step)
trainer.add_event_handler(Events.ITERATION_STARTED, lr_scheduler)

pbar = ProgressBar()
pbar.attach(trainer, output_transform=lambda x: {"loss": x})

def evaluate_step(engine, batch):
    model.eval()
    labels = batch.pop("labels").float().to(device)
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    preds = torch.sigmoid(outputs.logits)
    return preds, labels


train_evaluator = Engine(evaluate_step)
val_evaluator   = Engine(evaluate_step)

mse_fn = torch.nn.MSELoss()
Loss(mse_fn).attach(train_evaluator, "mse")
Loss(mse_fn).attach(val_evaluator,   "mse")


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(engine):
    train_evaluator.run(train_loader)
    mse = train_evaluator.state.metrics["mse"]
    log(f"Training Results   - Epoch: {engine.state.epoch}  MSE: {mse:.6f}")


@trainer.on(Events.EPOCH_COMPLETED)
def log_val_results(engine):
    val_evaluator.run(val_loader)
    mse = val_evaluator.state.metrics["mse"]
    log(f"Validation Results - Epoch: {engine.state.epoch}  MSE: {mse:.6f}")


def score_function(engine):
    return -engine.state.metrics["mse"]

handler = EarlyStopping(
    patience=PATIENCE, score_function=score_function, trainer=trainer,
)
val_evaluator.add_event_handler(Events.COMPLETED, handler)

checkpointer = ModelCheckpoint(
    dirname="models", filename_prefix="distilroberta",
    n_saved=2, create_dir=True,
)
trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {"model": model})

log(f"Beginning training on {device} for {N_EPOCHS} epochs "
    f"({num_training_steps} steps)...")
start = time.time_ns()
trainer.run(train_loader, max_epochs=N_EPOCHS)

log(f"Training completed in {round((time.time_ns() - start) / 1e9, 2)} s.")
save_log()
