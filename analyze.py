import time
import yaml

with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

full_log = ""
full_log += (
    f"Begin log for analyze.py\nCurrent time:\n"
    f"{time.strftime(config['logging']['timestamp_format_utc'], time.gmtime())} UTC\n"
    f"{time.strftime(config['logging']['timestamp_format_local'], time.localtime())} Local Time ({config['logging']['timezone_local']})\n\n"
)


def log(*msgs):
    global full_log
    for msg in msgs:
        print(msg)
        full_log += msg + "\n"


def save_log(PATH=config['logging']['default_log_path']):
    with open(PATH, "w", encoding="utf-8") as f:
        f.write(full_log)


log("Loading libraries...")
start = time.time_ns()

import json
import numpy as np
import tqdm
log("-" * 25 + "begin predict.py log" + "-" * 25)
from predict import predict_toxicity
log("-" * 26 + "end predict.py log" + "-" * 26)

log(f"Libraries loaded in {round((time.time_ns() - start) / 1000000, 3)} ms.")

log("Loading messages...")
start = time.time_ns()

with open(config['paths']['collected_messages'], "r", encoding="utf-8") as f:
    messages = json.load(f)

texts = [msg["content"] for msg in messages]

log(f"Messages loaded in {round((time.time_ns() - start) / 1000000, 3)} ms.")

log("Analyzing messages...")
start = time.time_ns()

results = []
for text in tqdm.tqdm(texts):
    result = predict_toxicity(text)
    results.append(result)

log(f"Messages analyzed in {round((time.time_ns() - start) / 1000000, 3)} ms.")

if not results:
    log("No results returned from predict_toxicity; aborting statistics computation.")
    save_log(config['paths']['analysis_log'])
    raise SystemExit(0)

# Build stats keys from first prediction result instead of relying on function defaults
stats = {label: [] for label in results[0].keys()}
for result in results:
    for label, prob in result.items():
        stats[label].append(prob)

log("Computing statistics...")
means = {label: np.mean(probs) for label, probs in stats.items()}
medians = {label: np.median(probs) for label, probs in stats.items()}
mins = {label: np.min(probs) for label, probs in stats.items()}
maxs = {label: np.max(probs) for label, probs in stats.items()}
rng = {label: np.ptp(probs) for label, probs in stats.items()}
std = {label: np.std(probs) for label, probs in stats.items()}
var = {label: np.var(probs) for label, probs in stats.items()}

# Correlation with timestamps
timestamps = [msg["timestamp"] for msg in messages]
min_ts, max_ts = min(timestamps), max(timestamps)
if max_ts == min_ts:
    normalized_ts = [0.0 for _ in timestamps]
else:
    normalized_ts = [(ts - min_ts) / (max_ts - min_ts) for ts in timestamps]

correlations = {}
for label, probs in stats.items():
    # If probs are constant, correlation may be NaN; guard against that
    try:
        corr = np.corrcoef(normalized_ts, probs)[0, 1]
        if np.isnan(corr):
            corr = 0.0
    except Exception:
        corr = 0.0
    correlations[label] = corr

log("Statistics computed:")
for label in means.keys():
    log(
        f"{label:20s} | mean: {means[label]:.4f} | median: {medians[label]:.4f} | "
        f"min: {mins[label]:.4f} | max: {maxs[label]:.4f} | range: {rng[label]:.4f} | "
        f"std: {std[label]:.4f} | var: {var[label]:.4f} | corr with time: {correlations[label]:.4f}"
    )

log(f"Statistics computed in {round((time.time_ns() - start) / 1000000, 3)} ms.")
save_log(config['paths']['analysis_log'])