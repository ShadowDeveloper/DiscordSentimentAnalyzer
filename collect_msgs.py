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

import os
import json
from datetime import datetime, timezone
import re

log(f"Libraries loaded in {round((time.time_ns() - start) / 1000000, 3)} ms.")

log("Collecting messages...")

PATH = config['paths']['messages_input']

messages = []


def parse_timestamp(ts_str):
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%d %H:%M:%S.%f%z"):
        try:
            dt = datetime.strptime(ts_str, fmt)
            return dt.replace(tzinfo=timezone.utc).timestamp() if dt.tzinfo is None else dt.timestamp()
        except ValueError:
            continue
    return None

def clean_message_content(content):
    content = content.strip()
    content = re.sub(config['processing']['patterns']['user_mention'], config['processing']['patterns']['user_mention_replacement'], content)
    content = re.sub(config['processing']['patterns']['url'], config['processing']['patterns']['url_replacement'], content)
    content = re.sub(config['processing']['patterns']['bot_command'], config['processing']['patterns']['bot_command_replacement'], content)
    return content

for root, dirs, files in os.walk(PATH):
    for file in files:
        if file == config['paths']['messages_file']:
            with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                data = json.load(f)

            data.sort(key=lambda m: m["Timestamp"])

            channel_messages = []
            for msg in data:
                if "Contents" in msg and msg["Contents"].strip():
                    ts = parse_timestamp(msg["Timestamp"])
                    if ts is None:
                        continue
                    if channel_messages and ts - channel_messages[-1]["timestamp"] < config['processing']['message_merge_threshold']:
                        channel_messages[-1]["content"] += "\n\n" + clean_message_content(msg["Contents"])
                        channel_messages[-1]["timestamp"] = ts
                    else:
                        channel_messages.append({
                            "content": clean_message_content(msg["Contents"]),
                            "timestamp": ts,
                            "channel": root,
                        })

            messages.extend(channel_messages)

log(f"Collected {len(messages)} messages.")

log("Saving messages...")

with open(config['paths']['collected_messages'], "w", encoding="utf-8") as f:
    json.dump(messages, f, ensure_ascii=False, indent=2)

log(f"Messages saved in collected_messages.json in {round((time.time_ns() - start) / 1000000, 3)} ms.")
save_log(config['paths']['collect_log'])
