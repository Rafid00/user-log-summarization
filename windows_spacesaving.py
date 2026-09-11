import time
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
from crick import SpaceSaving


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = "data/windows_logs.csv"

K = 100

OUTPUT_FILE = "results/windows_heavy_hitters.csv"
GRAPH_FILE = "graphs/windows_heavy_hitters.png"


# ============================================================
# 1. Load Windows log data
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("WINDOWS LOG + SPACE-SAVING EXPERIMENT")
print("=" * 70)

print(f"Total log records: {len(df)}")


# ============================================================
# 2. Create normalised event representation
# ============================================================

df["event"] = (
    df["ProviderName"].fillna("UNKNOWN")
    + ":"
    + df["Id"].astype(str)
)

print(f"Unique event types: {df['event'].nunique()}")


# ============================================================
# 3. Numeric encoding for Space-Saving
# ============================================================

event_to_id = {
    event: float(i)
    for i, event in enumerate(df["event"].unique(), start=1)
}

id_to_event = {
    value: key
    for key, value in event_to_id.items()
}


# ============================================================
# 4. Run Space-Saving
# ============================================================

ss = SpaceSaving(K)

start_time = time.perf_counter()

for event in df["event"]:
    ss.update(event_to_id[event])

algorithm_time = time.perf_counter() - start_time

results = ss.topk(K)


# ============================================================
# 5. Exact baseline
# ============================================================

exact_counts = Counter(df["event"])
exact_top = exact_counts.most_common(K)


# ============================================================
# 6. Calculate Recall@K
# ============================================================

approx_events = {
    id_to_event[float(item)]
    for item, frequency, error in results
}

exact_events = {
    event
    for event, frequency in exact_top
}

matching_events = approx_events & exact_events

recall = len(matching_events) / K


# ============================================================
# 7. Build results table
# ============================================================

output_rows = []

for rank, (item, frequency, error) in enumerate(results, start=1):

    event = id_to_event[float(item)]

    exact_frequency = exact_counts[event]

    output_rows.append({
        "Rank": rank,
        "Event": event,
        "Estimated_Frequency": int(frequency),
        "Error": int(error),
        "Exact_Frequency": int(exact_frequency),
    })


results_df = pd.DataFrame(output_rows)


# ============================================================
# 8. Save results
# ============================================================

results_df.to_csv(OUTPUT_FILE, index=False)


# ============================================================
# 9. Print summary
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENT SUMMARY")
print("=" * 70)

print(f"Input records:       {len(df)}")
print(f"Unique events:       {df['event'].nunique()}")
print(f"K:                   {K}")
print(f"Space-Saving time:   {algorithm_time:.6f} seconds")
print(f"Matching events:     {len(matching_events)}/{K}")
print(f"Recall@{K}:           {recall:.4f}")
print(f"Recall percentage:   {recall * 100:.2f}%")

print(f"\nResults saved to: {OUTPUT_FILE}")


# ============================================================
# 10. Generate heavy-hitter graph
# ============================================================

top10 = results_df.head(10).sort_values(
    "Estimated_Frequency"
)

plt.figure(figsize=(10, 6))

plt.barh(
    top10["Event"],
    top10["Estimated_Frequency"]
)

plt.xlabel("Estimated Frequency")
plt.ylabel("Event")
plt.title("Top 10 Windows Heavy Hitters")

plt.tight_layout()

plt.savefig(GRAPH_FILE, dpi=300)

plt.close()

print(f"Graph saved to: {GRAPH_FILE}")
