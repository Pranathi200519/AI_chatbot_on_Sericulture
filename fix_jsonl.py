import json

clean = []

with open("dataset.jsonl", "r", encoding="utf-8") as f:

    for line in f:
        line = line.strip()
        if not line:
            continue  # skip empty lines

        # if accidentally stored as a list or dict
        if line.startswith("[") or line.startswith("]") or line.endswith(","):
            continue

        try:
            obj = json.loads(line)
            clean.append(obj)
        except:
            print("Skipping invalid line:", line)

with open("clean_dataset_fixed.jsonl", "w", encoding="utf-8") as f:
    for item in clean:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("Fixed file saved as clean_dataset_fixed.jsonl")
