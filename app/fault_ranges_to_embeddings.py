from datetime import datetime
import pandas as pd
from database.vector_store import VectorStore
from timescale_vector.client import uuid_from_time

# Initialize your existing VectorStore
vec = VectorStore()

# Load the original full CSV
df = pd.read_csv("../data/MFCG1_RCM136_06132024_008.csv")

# Map fault levels to readable labels
FAULT_LEVELS = {
    2: "⚠️ Warning",
    3: "🔥 Fault",
    4: "🚨 Critical"
}

# Track intervals where fault level is 2, 3, or 4
ranges = []
in_range = False
start_idx = None
current_lvl = None

for idx, row in df.iterrows():
    lvl = int(row["SYS_FaultLvl"])
    
    if lvl in [2, 3, 4]:
        if not in_range:
            in_range = True
            start_idx = idx
            current_lvl = lvl
        elif lvl != current_lvl:
            ranges.append((start_idx, idx - 1, current_lvl))
            start_idx = idx
            current_lvl = lvl
    else:
        if in_range:
            ranges.append((start_idx, idx - 1, current_lvl))
            in_range = False
            start_idx = None
            current_lvl = None

# Edge case: last row ends in a fault state
if in_range:
    ranges.append((start_idx, len(df) - 1, current_lvl))

# Build and store summaries
records = []

for start, end, lvl in ranges:
    level_label = FAULT_LEVELS.get(lvl, f"Unknown ({lvl})")
    start_time = df.loc[start, "Datetime"]
    end_time = df.loc[end, "Datetime"]
    duration = end - start + 1

    summary = (
        f"{level_label} from line {start} to {end} "
        f"({duration} rows affected)\n"
        f"Time Range: {start_time} to {end_time}"
    )

    # Generate embedding
    embedding = vec.get_embedding(summary)

    # Compose record
    record = {
        "id": str(uuid_from_time(datetime.now())),
        "metadata": {
            "start_line": start,
            "end_line": end,
            "fault_lvl": lvl,
            "level_label": level_label,
            "duration_rows": duration,
            "start_time": start_time,
            "end_time": end_time,
            "created_at": datetime.now().isoformat()
        },
        "contents": summary,
        "embedding": embedding
    }

    records.append(record)

# Convert to DataFrame for upserting
records_df = pd.DataFrame(records)

# Insert into PostgreSQL vector DB
print(f"\n🔄 Inserting {len(records_df)} fault range summaries into vector store...")
vec.upsert(records_df)
print("✅ Insert completed.")
