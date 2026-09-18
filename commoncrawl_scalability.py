import gzip
import json
import os
import time
from urllib.parse import urlparse

import matplotlib.pyplot as plt
from crick import SpaceSaving


# ==========================================
# Configuration
# ==========================================

INPUT_FILE = "data/cdx-00000.gz"

# Keep the algorithm memory capacity fixed
K = 1000

# Amounts of Common Crawl records to process
RECORD_COUNTS = [
    100_000,
    500_000,
    1_000_000,
    5_000_000
]

RESULTS_FILE = "results/commoncrawl_scalability.csv"
GRAPH_FILE = "graphs/commoncrawl_scalability.png"


# ==========================================
# Helper
# ==========================================

def normalize_host(url):
    """
    Extract and normalize the hostname from a URL.
    """

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


# ==========================================
# Run one experiment
# ==========================================

def run_experiment(max_records):
    """
    Process max_records Common Crawl records
    using Space-Saving with fixed K.
    """

    space_saving = SpaceSaving(K, dtype="object")

    processed = 0
    skipped = 0

    start_time = time.perf_counter()

    with gzip.open(
        INPUT_FILE,
        "rt",
        encoding="utf-8",
        errors="replace"
    ) as f:

        for line in f:

            if processed >= max_records:
                break

            line = line.strip()

            if not line:
                continue

            try:

                # CDXJ:
                # SURT_KEY TIMESTAMP JSON
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

                space_saving.update(host)

                processed += 1

            except Exception:
                skipped += 1

    elapsed = time.perf_counter() - start_time

    throughput = processed / elapsed if elapsed > 0 else 0

    return processed, skipped, elapsed, throughput, space_saving


# ==========================================
# Main
# ==========================================

def main():

    os.makedirs("results", exist_ok=True)
    os.makedirs("graphs", exist_ok=True)

    print("=" * 65)
    print("Common Crawl Space-Saving Scalability Experiment")
    print("=" * 65)

    print(f"Input file: {INPUT_FILE}")
    print(f"Space-Saving capacity K: {K}")
    print(f"Record counts: {RECORD_COUNTS}")
    print()

    results = []

    for record_count in RECORD_COUNTS:

        print("-" * 65)
        print(f"Processing {record_count:,} records...")
        print("-" * 65)

        processed, skipped, elapsed, throughput, space_saving = (
            run_experiment(record_count)
        )

        print(f"Records processed : {processed:,}")
        print(f"Records skipped   : {skipped:,}")
        print(f"Processing time   : {elapsed:.4f} seconds")
        print(f"Throughput        : {throughput:,.0f} records/second")

        # Show the resulting top 5 heavy hitters
        print("\nTop 5 hosts:")

        top5 = space_saving.topk(5)

        for rank, (host, estimated_count, error) in enumerate(
            top5,
            start=1
        ):
            print(
                f"{rank}. "
                f"{host} | "
                f"estimated={int(estimated_count)} | "
                f"error={int(error)}"
            )

        print()

        results.append({
            "Records": processed,
            "ProcessingTimeSeconds": elapsed,
            "ThroughputRecordsPerSecond": throughput
        })

    # ==========================================
    # Save results
    # ==========================================

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "Records,ProcessingTimeSeconds,"
            "ThroughputRecordsPerSecond\n"
        )

        for row in results:

            f.write(
                f"{row['Records']},"
                f"{row['ProcessingTimeSeconds']:.6f},"
                f"{row['ThroughputRecordsPerSecond']:.2f}\n"
            )

    print("=" * 65)
    print(f"Results saved to: {RESULTS_FILE}")
    print("=" * 65)

    # ==========================================
    # Runtime graph
    # ==========================================

    records = [
        row["Records"]
        for row in results
    ]

    times = [
        row["ProcessingTimeSeconds"]
        for row in results
    ]

    plt.figure(figsize=(9, 5.5))

    plt.plot(
        records,
        times,
        marker="o"
    )

    # Add values above each point
    for x, y in zip(records, times):

        plt.annotate(
            f"{y:.2f}s",
            (x, y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center"
        )

    plt.xlabel("Number of Common Crawl Records")
    plt.ylabel("Processing Time (seconds)")
    plt.title(
        "Common Crawl: Stream Size vs Processing Time "
        f"(K={K})"
    )

    plt.grid(True, alpha=0.3)

    plt.xticks(
        records,
        [
            "100K",
            "500K",
            "1M",
            "5M"
        ]
    )

    plt.tight_layout()

    plt.savefig(
        GRAPH_FILE,
        dpi=300
    )

    plt.close()

    print(f"Graph saved to: {GRAPH_FILE}")

    # ==========================================
    # Final summary
    # ==========================================

    print()
    print("=" * 65)
    print("FINAL SUMMARY")
    print("=" * 65)

    print(
        f"{'Records':>12} "
        f"{'Time (s)':>15} "
        f"{'Records/sec':>18}"
    )

    for row in results:

        print(
            f"{row['Records']:>12,} "
            f"{row['ProcessingTimeSeconds']:>15.4f} "
            f"{row['ThroughputRecordsPerSecond']:>18,.0f}"
        )


if __name__ == "__main__":
    main()
