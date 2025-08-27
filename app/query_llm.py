# app/query_llm.py
import os, sys, argparse, logging
from typing import Optional, List, Tuple, Dict, Any
import pandas as pd
from dotenv import load_dotenv

# load env first
load_dotenv(override=True)

# robust imports (python app\query_llm.py vs -m app.query_llm)
try:
    from app.database.vector_store import VectorStore
except ImportError:
    from database.vector_store import VectorStore

from timescale_vector import client as tclient
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# -------------------------- utils --------------------------
def pick_text_column(df: pd.DataFrame) -> Optional[str]:
    for c in ["contents", "content", "text", "body", "chunk", "snippet", "message"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if df[c].dtype == "object" and df[c].astype(str).str.len().mean() > 40:
            return c
    return None


def safe_to_float(x):
    try:
        return float(x)
    except Exception:
        return None


def find_runs(df: pd.DataFrame, level_col: str = "fault_lvl_num", threshold: float = 0.0) -> pd.DataFrame:
    """
    Detect contiguous runs where level > threshold per fault_id.
    Requires at least 'fault_id' and either 'row_index' or 'timestamp'.
    """
    if df is None or df.empty or "fault_id" not in df.columns or level_col not in df.columns:
        return pd.DataFrame(columns=["fault_id", "row_start", "row_end", "t_start", "t_end", "level", "n_rows"])

    work = df.copy()

    # prefer row_index ordering; else try timestamp
    has_row_idx = "row_index" in work.columns
    if has_row_idx:
        work["row_index"] = pd.to_numeric(work["row_index"], errors="coerce")
    if "timestamp" in work.columns:
        work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")

    # normalize level to float
    work[level_col] = work[level_col].apply(safe_to_float)

    runs_out: List[Dict[str, Any]] = []

    for fid, g in work.groupby("fault_id"):
        g = g.copy()
        if has_row_idx:
            g = g.sort_values("row_index")
        elif "timestamp" in g.columns:
            g = g.sort_values("timestamp")
        else:
            g = g.sort_index()

        cur_start_i = None
        cur_level = None
        rows_accum: List[pd.Series] = []

        prev_above = False
        for _, r in g.iterrows():
            lvl = r[level_col]
            above = (lvl is not None) and (lvl > threshold)
            if above and not prev_above:
                # start a new run
                cur_start_i = r.get("row_index")
                cur_level = lvl
                rows_accum = [r]
            elif above and prev_above:
                rows_accum.append(r)
            elif (not above) and prev_above:
                # close previous run
                if rows_accum:
                    row_start = rows_accum[0].get("row_index")
                    row_end   = rows_accum[-1].get("row_index")
                    t_start   = rows_accum[0].get("timestamp")
                    t_end     = rows_accum[-1].get("timestamp")
                    # choose the most common level inside the run (should be constant)
                    lvl_mode = pd.Series([x[level_col] for _, x in pd.DataFrame(rows_accum).iterrows()]).mode()
                    lvl_val = float(lvl_mode.iloc[0]) if not lvl_mode.empty else float(cur_level or 0.0)
                    runs_out.append({
                        "fault_id": int(fid) if pd.notna(fid) else None,
                        "row_start": int(row_start) if pd.notna(row_start) else None,
                        "row_end": int(row_end) if pd.notna(row_end) else None,
                        "t_start": t_start,
                        "t_end": t_end,
                        "level": lvl_val,
                        "n_rows": len(rows_accum),
                    })
                cur_start_i = None
                cur_level = None
                rows_accum = []

            prev_above = above

        # if ended while still above
        if prev_above and rows_accum:
            row_start = rows_accum[0].get("row_index")
            row_end   = rows_accum[-1].get("row_index")
            t_start   = rows_accum[0].get("timestamp")
            t_end     = rows_accum[-1].get("timestamp")
            lvl_mode = pd.Series([x[level_col] for _, x in pd.DataFrame(rows_accum).iterrows()]).mode()
            lvl_val = float(lvl_mode.iloc[0]) if not lvl_mode.empty else float(cur_level or 0.0)
            runs_out.append({
                "fault_id": int(fid) if pd.notna(fid) else None,
                "row_start": int(row_start) if pd.notna(row_start) else None,
                "row_end": int(row_end) if pd.notna(row_end) else None,
                "t_start": t_start,
                "t_end": t_end,
                "level": lvl_val,
                "n_rows": len(rows_accum),
            })

    if not runs_out:
        return pd.DataFrame(columns=["fault_id", "row_start", "row_end", "t_start", "t_end", "level", "n_rows"])

    out = pd.DataFrame(runs_out)
    # nice ordering
    if "t_start" in out.columns:
        out = out.sort_values(["t_start", "fault_id", "row_start"], kind="stable")
    return out


def summarize_for_console(df: pd.DataFrame, text_col: str, label: str) -> None:
    print(f"\n[{label}] Top matches summary:")
    if "fault_lvl_num" in df.columns:
        counts = df["fault_lvl_num"].value_counts(dropna=False).sort_index()
        for lvl, cnt in counts.items():
            print(f"  - level {lvl}: {cnt}")
    print("  Examples:")
    for _, row in df.head(5).iterrows():
        meta_bits = []
        for k in ("fault_id", "fault_lvl_num", "fault_lvl", "fault_st", "timestamp", "debug_fault_id", "row_index"):
            if k in df.columns and pd.notna(row.get(k)):
                meta_bits.append(f"{k}={row.get(k)}")
        meta = " | ".join(meta_bits) if meta_bits else "(no meta)"
        snippet = (str(row[text_col]) or "")[:240].replace("\n", " ")
        print(f"   - {meta}\n     {snippet}")


def llm_answer(question: str, df: pd.DataFrame, text_col: str, runs_df: pd.DataFrame) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else OpenAI()

    # structured run summary fed first
    if runs_df is not None and not runs_df.empty:
        lines = []
        for _, r in runs_df.iterrows():
            lines.append(
                f"fault_id={r['fault_id']}, level={r['level']}, rows={r['row_start']}-{r['row_end']}, "
                f"t_start={r['t_start']}, t_end={r['t_end']}, n_rows={r['n_rows']}"
            )
        runs_block = "Detected episodes (computed):\n" + "\n".join(lines) + "\n"
    else:
        runs_block = "Detected episodes (computed): none in this retrieval window.\n"

    # compact doc context from top rows (cap to avoid prompt bloat)
    ctx_rows = min(len(df), 40)
    docs = []
    for _, row in df.head(ctx_rows).iterrows():
        hdr_parts = []
        for k in ("fault_id", "fault_lvl_num", "fault_lvl", "fault_st", "timestamp", "debug_fault_id", "row_index"):
            if k in df.columns and pd.notna(row.get(k)):
                hdr_parts.append(f"{k}={row.get(k)}")
        header = ", ".join(hdr_parts) if hdr_parts else "match"
        body = str(row[text_col])[:900]
        docs.append(f"{header}\n{body}")
    context = runs_block + "\n--- Raw matches ---\n" + "\n\n---\n\n".join(docs)

    system_prompt = (
        "You are an expert in Rocketruck generator diagnostics. "
        "Use ONLY the provided context. Do not invent missing semantics. "
        "If the context is scoped to a row window or specific fault IDs, restrict conclusions to that scope. "
        "Summarize fault levels and identify episodes (start/end) using the computed list when present. "
        "If information is insufficient, say so."
    )
    user_prompt = f"Question: {question}\n\nContext:\n{context}"

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user",   "content": user_prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def build_predicates(args) -> Optional[tclient.Predicates]:
    pred: Optional[tclient.Predicates] = None

    def AND(p1: Optional[tclient.Predicates], p2: tclient.Predicates):
        return p2 if p1 is None else (p1 & p2)

    # numeric level filter first (pure numeric; avoids string/int mismatches)
    if args.level_min is not None:
        pred = AND(pred, tclient.Predicates("fault_lvl_num", ">=", float(args.level_min)))
    if args.level_max is not None:
        pred = AND(pred, tclient.Predicates("fault_lvl_num", "<=", float(args.level_max)))

    # row window
    if args.rows:
        try:
            a, b = args.rows.split(":")
            lo, hi = int(a), int(b)
            pred = AND(pred, tclient.Predicates("row_index", ">=", lo))
            pred = AND(pred, tclient.Predicates("row_index", "<=", hi))
        except Exception:
            pass  # ignore if malformed; we’ll just not apply it

    # fault-id filter (single or comma-separated list)
    if args.fault_id:
        ids = []
        for tok in str(args.fault_id).split(","):
            tok = tok.strip()
            if tok:
                try:
                    ids.append(int(tok))
                except Exception:
                    continue
        if ids:
            # OR the fault_id options together, then AND with prior filters
            or_pred = None
            for fid in ids:
                p = tclient.Predicates("fault_id", "==", fid)
                or_pred = p if or_pred is None else (or_pred | p)
            pred = AND(pred, or_pred)

    return pred


# -------------------------- main --------------------------
def parse_args(argv: List[str]):
    ap = argparse.ArgumentParser(description="Ask questions about Rocketruck fault logs via RAG.")
    ap.add_argument("question", type=str, nargs="+", help="Your question (quote if it has spaces).")
    ap.add_argument("--rows", type=str, default=None, help="Row window like 8710:8730")
    ap.add_argument("--fault-id", type=str, default=None, help="Single fault id or comma-separated list")
    ap.add_argument("--level-min", type=float, default=0.0, help="Min numeric level (e.g., 0.0, 2.0, 3.0)")
    ap.add_argument("--level-max", type=float, default=None, help="Max numeric level")
    ap.add_argument("--limit", type=int, default=60, help="Top-K results to retrieve")
    return ap.parse_args(argv)


def main():
    # allow both interactive and CLI usage
    if len(sys.argv) > 1:
        args = parse_args(sys.argv[1:])
        question = " ".join(args.question).strip()
    else:
        # interactive fallback
        question = input("Ask a question about Rocketruck fault logs:\n> ").strip()
        parser = parse_args([])
        args = parser  # defaults

    vec = VectorStore()

    # build predicates according to flags
    preds = build_predicates(args)

    print("\nSearching vector DB (focused: numeric filtering)…")
    results = vec.query(question, limit=args.limit, predicates=preds, return_dataframe=True)

    if results is None or getattr(results, "empty", True):
        print("❌ No relevant results found.")
        return

    # ensure useful sorts for run detection
    if "row_index" in results.columns:
        results = results.sort_values("row_index")
    elif "timestamp" in results.columns:
        results = results.sort_values("timestamp")

    text_col = pick_text_column(results)
    if not text_col:
        print("❌ No valid text column found in vector search results.")
        print("Available columns:", list(results.columns))
        return

    summarize_for_console(results, text_col, label="FOCUSED")

    # compute episodes entirely in Python (no SQL)
    runs_df = find_runs(results, level_col="fault_lvl_num", threshold=0.0)
    if runs_df is not None and not runs_df.empty:
        print("\nDetected episodes (Python):")
        for _, r in runs_df.iterrows():
            print(f"  - fault {r['fault_id']}: level {r['level']} rows {r['row_start']}-{r['row_end']} "
                  f"({r['t_start']} → {r['t_end']}), n={r['n_rows']}")
    else:
        print("\nDetected episodes (Python): none")

    print("\nQuerying OpenAI…")
    answer = llm_answer(question, results, text_col, runs_df)

    print("\n🧠 LLM Response:\n")
    print(answer)


if __name__ == "__main__":
    main()
