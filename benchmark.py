import time
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from crick import SpaceSaving


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = "data/windows_logs.csv"
OUTPUT_FILE = "results/runtime_benchmark.csv"
GRAPH_FILE = "graphs/runtime_vs_input_size.png"

K = 100

INPUT_SIZES = [
    5_000,
    10_000,
    20_000,
    30_000,
    50_000,
    100_000,
    200_000,
]

# Repeat each benchmark multiple times to reduce timing noise
REPEATS = 5


# ============================================================
# Load and normalise Windows events
# ============================================================

df = pd.read_csv(INPUT_FILE)

df["event"] = (
    df["ProviderName"].fillna("UNKNOWN")
    + ":"
    + df["Id"].astype(str)
)

events = df["event"].tolist()

print("=" * 70)
print("SPACE-SAVING RUNTIME BENCHMARK")
print("=" * 70)

print(f"Original records: {len(events)}")
print(f"Unique event types: {len(set(events))}")
print(f"K: {K}")
print(f"Repeats per input size: {REPEATS}")


# ============================================================
# Numeric encoding
# ============================================================

event_to_id = {
    event: float(i)
    for i, event in enumerate(sorted(set(events)), start=1)
}

numeric_events = [
    event_to_id[event]
    for event in events
]


# ============================================================
# Function to create a stream of requested size
# ============================================================

def make_stream(size):
    """Create a stream of exactly 'size' events by cycling
    through the available Windows events."""

    result = []

    while len(result) < size:
        remaining = size - len(result)
        result.extend(numeric_events[:remaining])

    return result


# ============================================================
# Benchmark
# ============================================================

benchmark_rows = []

for size in INPUT_SIZES:

    stream = make_stream(size)

    run_times = []

    print(f"\nTesting {size:,} records...")

    for run in range(1, REPEATS + 1):

        ss = SpaceSaving(K)

        start = time.perf_counter()

        for event in stream:
            ss.update(event)

        end = time.perf_counter()

        elapsed = end - start

        run_times.append(elapsed)

        print(f"  Run {run}: {elapsed:.6f} seconds")

    average_time = sum(run_times) / len(run_times)

    minimum_time = min(run_times)
    maximum_time = max(run_times)

    throughput = size / average_time

    benchmark_rows.append({
        "Input_Size": size,
        "K": K,
        "Average_Runtime_Seconds": average_time,
        "Minimum_Runtime_Seconds": minimum_time,
        "Maximum_Runtime_Seconds": maximum_time,
        "Events_Per_Second": throughput,
    })

    print(f"  Average: {average_time:.6f} seconds")


# ============================================================
# Save results
# ============================================================

results_df = pd.DataFrame(benchmark_rows)

Path("results").mkdir(exist_ok=True)
Path("graphs").mkdir(exist_ok=True)

results_df.to_csv(OUTPUT_FILE, index=False)


# ============================================================
# Print final table
# ============================================================

print("\n" + "=" * 70)
print("FINAL RUNTIME RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Average_Runtime_Seconds": "{:.6f}".format,
            "Minimum_Runtime_Seconds": "{:.6f}".format,
            "Maximum_Runtime_Seconds": "{:.6f}".format,
            "Events_Per_Second": "{:.2f}".format,
        }
    )
)

print(f"\nResults saved to: {OUTPUT_FILE}")


# ============================================================
# Generate runtime graph
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    results_df["Input_Size"],
    results_df["Average_Runtime_Seconds"],
    marker="o"
)

plt.xlabel("Number of Input Events")
plt.ylabel("Average Runtime (seconds)")
plt.title("Space-Saving Runtime vs Input Size")

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(GRAPH_FILE, dpi=300)

plt.close()

print(f"Graph saved to: {GRAPH_FILE}")
