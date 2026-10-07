import os
import json
import pandas as pd

SOURCE = "./new_training_data"
OUTPUT = "./final_training_data"

os.makedirs(OUTPUT, exist_ok=True)

OUT_FILE = os.path.join(OUTPUT, "train.jsonl")


def save_example(f, user, assistant):
    """Save one normalized conversation."""
    if not isinstance(user, str) or not isinstance(assistant, str):
        return False

    user = user.strip()
    assistant = assistant.strip()

    if not user or not assistant:
        return False

    record = {
        "messages": [
            {
                "role": "user",
                "content": user
            },
            {
                "role": "assistant",
                "content": assistant
            }
        ]
    }

    f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return True


count = 0


with open(OUT_FILE, "w", encoding="utf-8") as out:

    # ---------------------------------------------------------
    # 1. NSFW_Chat_Datase
    # input -> output
    # ---------------------------------------------------------

    path = os.path.join(
        SOURCE,
        "NSFW_Chat_Datase",
        "final_dataset.json"
    )

    if os.path.exists(path):
        print("Processing NSFW_Chat_Datase...")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            if save_example(
                out,
                item.get("input"),
                item.get("output")
            ):
                count += 1


    # ---------------------------------------------------------
    # 2. Sexting CSV
    # He -> She
    # ---------------------------------------------------------

    path = os.path.join(
        SOURCE,
        "sexting",
        "chat_data.csv"
    )

    if os.path.exists(path):
        print("Processing sexting...")

        df = pd.read_csv(path)

        for _, row in df.iterrows():
            if save_example(
                out,
                row.get("He"),
                row.get("She")
            ):
                count += 1


    # ---------------------------------------------------------
    # 3. Wifu chat JSONL
    # Already in messages format
    # ---------------------------------------------------------

    path = os.path.join(
        SOURCE,
        "wifu_chat",
        "data",
        "train.jsonl"
    )

    if os.path.exists(path):
        print("Processing wifu_chat...")

        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    item = json.loads(line)

                    if "messages" in item:
                        messages = item["messages"]

                        if isinstance(messages, list) and len(messages) >= 2:
                            out.write(
                                json.dumps(
                                    {"messages": messages},
                                    ensure_ascii=False
                                ) + "\n"
                            )
                            count += 1

                except Exception:
                    continue


    # ---------------------------------------------------------
    # 4. Synthetic-RP
    # ---------------------------------------------------------

    path = os.path.join(
        SOURCE,
        "Synthetic-RP",
        "data",
        "train-00000-of-00001.parquet"
    )

    if os.path.exists(path):
        print("Processing Synthetic-RP...")

        df = pd.read_parquet(path)

        for _, row in df.iterrows():

            messages = row.get("messages")

            if isinstance(messages, list) and len(messages) >= 2:

                out.write(
                    json.dumps(
                        {"messages": messages},
                        ensure_ascii=False
                    ) + "\n"
                )

                count += 1


print()
print("=" * 60)
print("PREPARATION COMPLETE")
print("=" * 60)
print("Training examples:", count)
print("Output:", OUT_FILE)
