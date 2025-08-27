# app/insert_vectors.py
import os
import math
import uuid
import logging
import pandas as pd
from dotenv import load_dotenv

# Make .env override any stale user/system vars
load_dotenv(override=True)

# Robust import regardless of how you run the script
try:
    from app.database.vector_store import VectorStore
except ImportError:
    from database.vector_store import VectorStore  # fallback

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# === CSV path (override via .env: CSV_PATH=C:\path\to\file.csv) ===
CSV_PATH = os.getenv(
    "CSV_PATH",
    os.path.join(os.path.dirname(__file__), "..", "data", "MFCG1_RCM136_06132024_008.csv"),
)
DATASET_TAG = os.path.basename(CSV_PATH)  # for selective deletes later

def to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def main():
    if not os.path.exists(CSV_PATH):
        print(f"❌ CSV not found: {CSV_PATH}")
        return

    print(f"📄 Loading CSV: {CSV_PATH}")
    df_raw = pd.read_csv(CSV_PATH)

    vec = VectorStore()

    rows = []
    total = len(df_raw)
    print(f"🔍 Generating embeddings for {total} rows…")

    for i, r in df_raw.iterrows():
        # Map from your CSV headers
        fault_id  = to_float(r.get("SYS_FaultID"))
        fault_lvl = to_float(r.get("SYS_FaultLvl"))      # numeric (will also store as string)
        fault_st  = to_float(r.get("SYS_FaultSt"))
        debug_id  = to_float(r.get("Debug_FaultActiveID"))
        ts        = str(r.get("Datetime") or "")

        # Build retrieval text (include numeric level)
        contents = (
            f"⚠️ Active Fault Detected "
            f"Fault ID: {int(fault_id) if fault_id is not None else 'NA'} "
            f"Fault Level: {fault_lvl if fault_lvl is not None else 'NA'} "
            f"Fault State: {fault_st if fault_st is not None else 'NA'} "
            f"Debug_FaultActiveID: {debug_id if debug_id is not None else 'NA'} "
            f"Timestamp: {ts}"
        )

        # One embedding per row
        emb = vec.get_embedding(contents)

        # Write BOTH a string and numeric level into metadata
        meta = {
            "row_index": i + 1,  # 1-based like Excel
            "source": DATASET_TAG,

            "fault_id": int(fault_id) if fault_id is not None and not math.isnan(fault_id) else None,
            "fault_lvl": f"{fault_lvl:.1f}" if fault_lvl is not None and not math.isnan(fault_lvl) else None,  # string
            "fault_lvl_num": fault_lvl,  # numeric for predicates (e.g., > 0)

            "fault_st": fault_st,
            "debug_fault_id": debug_id,
            "timestamp": ts,
        }

        rows.append({
            "id": uuid.uuid4(),           # UUID primary key
            "metadata": meta,             # JSONB
            "contents": contents,         # TEXT
            "embedding": emb,             # VECTOR(n)
        })

        if (i + 1) % 1000 == 0:
            print(f"…processed {i+1}/{total}")

    df = pd.DataFrame(rows, columns=["id", "metadata", "contents", "embedding"])
    print(f"✅ Prepared {len(df)} records.")

    print("🛠 Creating tables…")
    vec.create_tables()  # your VectorStore will fall back to pgvector-only if vectorscale is missing

    try:
        print("🛠 Creating index…")
        vec.create_index()  # ensures HNSW on embedding
    except Exception as e:
        print(f"⚠️ Index creation skipped: {e}")

    print("📥 Inserting records…")
    vec.upsert(df)
    print("🎉 Done.")

if __name__ == "__main__":
    main()
