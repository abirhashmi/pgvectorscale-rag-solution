import pandas as pd

# Load your dataset
df = pd.read_csv("../data/MFCG1_RCM136_06132024_008.csv")

# --- Fault Transitions ---
transitions = []
prev_value = None

for idx, row in df.iterrows():
    curr = row["Debug_FaultActiveID"]

    if prev_value is not None:
        if prev_value == 0 and curr != 0:
            transitions.append((idx, "FAULT STARTED", row["Datetime"], curr))
        elif prev_value != 0 and curr == 0:
            transitions.append((idx, "FAULT CLEARED", row["Datetime"], prev_value))

    prev_value = curr

print(f"\n🌀 Detected {len(transitions)} fault transitions:\n")
for idx, status, timestamp, fault_id in transitions:
    print(f"{status} at line {idx} – Time: {timestamp} – Fault ID: {int(fault_id)}")

# --- Critical/Fault-Level Detection ---
critical_rows = df[(df["SYS_FaultLvl"] == 3) | (df["SYS_FaultLvl"] == 4)]

print(f"\n🔥 Detected {len(critical_rows)} rows with FaultLvl 3 (Fault) or 4 (Critical):\n")
for idx, row in critical_rows.iterrows():
    print(f"CRITICAL FAULT at line {idx} – FaultLvl: {int(row['SYS_FaultLvl'])}, FaultID: {int(row['SYS_FaultID'])}, Time: {row['Datetime']}")

# --- Export Critical Rows to CSV ---
output_path = "critical_faults_output.csv"
critical_rows.to_csv(output_path, index=False)
print(f"\n✅ Exported all critical/fault-level rows to: {output_path}")
