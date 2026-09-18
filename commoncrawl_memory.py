import gzip
import json
import gc
import os
import sys
import time
import subprocess
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import psutil
from crick import SpaceSaving


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = "data/cdx-00000.gz"
MAX_RECORDS = 1_000_000

K_VALUES = [100, 500, 1000, 5000]

RESULTS_FILE = "results/commoncrawl_memory.csv"
GRAPH_FILE = "graphs/commoncrawl_k_vs_memory.png"


# ============================================================
# URL normalization
# ============================================================

def normalize_host(url):
    try:
        host = urlparse(url).hostname

        if host is None:
            return None

        host = host.lower().strip()

        if host.startswith("www."):
            host = host[4:]

        return host

    except Exception:
        return None


# ============================================================
# Worker process
# ============================================================

def worker(mode, k):

    gc.collect()

    space_saving = None

    if mode == "spacesaving":
        space_saving = SpaceSaving(k, dtype="object")

    process = psutil.Process(os.getpid())

    # Memory immediately after initialization
    initial_memory = (
        process.memory_info().rss / (1024 * 1024)
    )

    peak_memory = initial_memory

    processed = 0
    skipped = 0

    start = time.perf_counter()

    with gzip.open(
        INPUT_FILE,
        "rt",
        encoding="utf-8",
        errors="replace"
    ) as f:

        for line in f:

            if processed >= MAX_RECORDS:
                break

            line = line.strip()

            if not line:
                continue

            try:

                parts = line.split(" ", 2)

                if len(parts) != 3:
                    skipped += 1
                    continue

                metadata = json.loads(parts[2])

                url = metadata.get("url")

                if not url:
                    skipped += 1
                    continue

                host = normalize_host(url)

                if not host:
                    skipped += 1
                    continue

                if mode == "spacesaving":
                    space_saving.update(host)

                processed += 1

                # Check memory every 10,000 records
                if processed % 10_000 == 0:

                    current_memory = (
                        process.memory_info().rss
                        / (1024 * 1024)
                    )

                    peak_memory = max(
                        peak_memory,
                        current_memory
                    )

            except Exception:
                skipped += 1

    elapsed = time.perf_counter() - start

    final_memory = (
        process.memory_info().rss
        / (1024 * 1024)
    )

    peak_memory = max(
        peak_memory,
        final_memory
    )

    print(
        f"MODE={mode} "
        f"INITIAL={initial_memory:.2f} "
        f"PEAK={peak_memory:.2f} "
        f"TIME={elapsed:.4f} "
        f"PROCESSED={processed}"
    )


# ============================================================
# Run child process and extract results
# ============================================================

def run_worker(mode, k):

    command = [
        sys.executable,
        __file__,
        "--worker",
        mode,
        str(k)
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:

        print(result.stdout)
        print(result.stderr)

        raise RuntimeError(
            f"Worker failed for mode={mode}, K={k}"
        )

    output = result.stdout.strip()

    print(output)

    values = {}

    for part in output.split():

        if "=" not in part:
            continue

        key, value = part.split("=", 1)

        values[key] = value

    return {
        "initial": float(values["INITIAL"]),
        "peak": float(values["PEAK"]),
        "time": float(values["TIME"]),
        "processed": int(values["PROCESSED"])
    }


# ============================================================
# Main experiment
# ============================================================

def main():

    os.makedirs("results", exist_ok=True)
    os.makedirs("graphs", exist_ok=True)

    print("=" * 70)
    print("Common Crawl Space-Saving Memory Experiment")
    print("=" * 70)

    print(f"Records per experiment: {MAX_RECORDS:,}")
    print(f"K values: {K_VALUES}")
    print()

    # --------------------------------------------------------
    # Baseline: no Space-Saving
    # --------------------------------------------------------

    print("-" * 70)
    print("Running baseline pipeline (NO Space-Saving)")
    print("-" * 70)

    baseline = run_worker(
        "baseline",
        0
    )

    baseline_peak = baseline["peak"]

    print()
    print(
        f"Baseline peak memory: "
        f"{baseline_peak:.2f} MB"
    )
    print()

    # --------------------------------------------------------
    # Space-Saving experiments
    # --------------------------------------------------------

    results = []

    for K in K_VALUES:

        print("-" * 70)
        print(f"Running Space-Saving with K = {K}")
        print("-" * 70)

        experiment = run_worker(
            "spacesaving",
            K
        )

        algorithm_peak = experiment["peak"]

        memory_overhead = (
            algorithm_peak - baseline_peak
        )

        results.append({
            "K": K,
            "BaselinePeakMB": baseline_peak,
            "SpaceSavingPeakMB": algorithm_peak,
            "SpaceSavingOverheadMB": memory_overhead,
            "ProcessingTimeSeconds": experiment["time"]
        })

        print(
            f"Space-Saving peak memory: "
            f"{algorithm_peak:.2f} MB"
        )

        print(
            f"Space-Saving memory overhead: "
            f"{memory_overhead:.2f} MB"
        )

        print()

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "K,"
            "BaselinePeakMB,"
            "SpaceSavingPeakMB,"
            "SpaceSavingOverheadMB,"
            "ProcessingTimeSeconds\n"
        )

        for row in results:

            f.write(
                f"{row['K']},"
                f"{row['BaselinePeakMB']:.4f},"
                f"{row['SpaceSavingPeakMB']:.4f},"
                f"{row['SpaceSavingOverheadMB']:.4f},"
                f"{row['ProcessingTimeSeconds']:.6f}\n"
            )

    print("=" * 70)
    print(f"Results saved to: {RESULTS_FILE}")
    print("=" * 70)

    # --------------------------------------------------------
    # Graph
    # --------------------------------------------------------

    k_values = [
        row["K"]
        for row in results
    ]

    memory_values = [
        row["SpaceSavingOverheadMB"]
        for row in results
    ]

    plt.figure(figsize=(9, 5.5))

    plt.plot(
        k_values,
        memory_values,
        marker="o"
    )

    for x, y in zip(
        k_values,
        memory_values
    ):

        plt.annotate(
            f"{y:.2f} MB",
            (x, y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center"
        )

    plt.xlabel("Space-Saving Capacity (K)")
    plt.ylabel("Additional Memory Used (MB)")

    plt.title(
        "Common Crawl: Space-Saving Capacity vs Memory Overhead"
    )

    plt.grid(True, alpha=0.3)

    plt.xticks(k_values)

    plt.tight_layout()

    plt.savefig(
        GRAPH_FILE,
        dpi=300
    )

    plt.close()

    print(f"Graph saved to: {GRAPH_FILE}")

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        f"{'K':>8} "
        f"{'Baseline MB':>15} "
        f"{'SpaceSaving MB':>17} "
        f"{'Overhead MB':>15} "
        f"{'Time (s)':>12}"
    )

    for row in results:

        print(
            f"{row['K']:>8} "
            f"{row['BaselinePeakMB']:>15.2f} "
            f"{row['SpaceSavingPeakMB']:>17.2f} "
            f"{row['SpaceSavingOverheadMB']:>15.2f} "
            f"{row['ProcessingTimeSeconds']:>12.4f}"
        )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) >= 2 and sys.argv[1] == "--worker":

        mode = sys.argv[2]
        k = int(sys.argv[3])

        worker(mode, k)

    else:

        main()
