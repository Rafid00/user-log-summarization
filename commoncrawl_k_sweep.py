import gzip
import json
import os
import time
from collections import Counter
from urllib.parse import urlparse

import matplotlib.pyplot as plt
from crick import SpaceSaving


# =========================
# Configuration
# =========================

INPUT_FILE = "data/cdx-00000.gz"
MAX_RECORDS = 100_000

# Memory capacities to test
K_VALUES = [100, 500, 1000, 5000]

# We always evaluate how well the algorithm finds
# the exact top 100 heavy hitters.
EVALUATION_K = 100

RESULTS_FILE = "results/commoncrawl_k_sweep.csv"
GRAPH_FILE = "graphs/commoncrawl_k_vs_recall.png"


# =========================
# Helper functions
# =========================

def normalize_host(url):
    """
    Extract and normalize hostname from a URL.
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


def read_hosts(max_records):
    """
    Stream Common Crawl CDXJ records from gzip
    and yield normalized hosts.
    """

    processed = 0
    skipped = 0

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
                # CDXJ format:
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

                yield host

                processed += 1

            except Exception:
                skipped += 1

    print(f"Records read: {processed}")
    print(f"Records skipped: {skipped}")


# =========================
# Main experiment
# =========================

def main():

    os.makedirs("results", exist_ok=True)
    os.makedirs("graphs", exist_ok=True)

    print("=" * 60)
    print("Common Crawl Space-Saving K Sweep")
    print("=" * 60)

    print(f"Input file: {INPUT_FILE}")
    print(f"Records per experiment: {MAX_RECORDS}")
    print(f"K values: {K_VALUES}")
    print(f"Evaluation: Recall@{EVALUATION_K}")
    print()

    # -----------------------------------------
    # Step 1: Build exact ground truth
    # -----------------------------------------

    print("Building exact ground truth...")

    exact_counter = Counter()

    start = time.perf_counter()

    for host in read_hosts(MAX_RECORDS):
        exact_counter[host] += 1

    exact_time = time.perf_counter() - start

    exact_top100 = exact_counter.most_common(EVALUATION_K)
    exact_top100_set = {host for host, _ in exact_top100}

    print()
    print(f"Unique hosts: {len(exact_counter):,}")
    print(f"Exact processing time: {exact_time:.4f} seconds")
    print()

    print("Exact Top 10:")
    for rank, (host, count) in enumerate(exact_top100[:10], start=1):
        print(f"{rank:2d}. {host:<50} {count}")

    print()

    # -----------------------------------------
    # Step 2: Run Space-Saving for each K
    # -----------------------------------------

    results = []

    for K in K_VALUES:

        print("-" * 60)
        print(f"Running Space-Saving with K = {K}")
        print("-" * 60)

        space_saving = SpaceSaving(K, dtype="object")

        start = time.perf_counter()

        for host in read_hosts(MAX_RECORDS):
            space_saving.update(host)

        elapsed = time.perf_counter() - start

        # Get the top 100 from the Space-Saving structure
        space_top100 = space_saving.topk(EVALUATION_K)

        # Convert to normal Python list
        space_top100_items = []

        for item, estimated_count, error in space_top100:
            space_top100_items.append(
                (
                    str(item),
                    int(estimated_count),
                    int(error)
                )
            )

        # Calculate Recall@100
        matching = sum(
            1
            for host, _, _ in space_top100_items
            if host in exact_top100_set
        )

        recall = matching / EVALUATION_K

        print(f"Processing time: {elapsed:.4f} seconds")
        print(f"Matching top-100 hosts: {matching}/{EVALUATION_K}")
        print(f"Recall@100: {recall * 100:.2f}%")
        print()

        print("Space-Saving Top 10:")

        for rank, (host, estimated, error) in enumerate(
            space_top100_items[:10],
            start=1
        ):
            exact_count = exact_counter.get(host, 0)

            print(
                f"{rank:2d}. "
                f"{host:<50} "
                f"estimated={estimated:<6} "
                f"exact={exact_count:<6} "
                f"error={error}"
            )

        print()

        results.append({
            "K": K,
            "Recall@100": recall * 100,
            "Matching": matching,
            "ProcessingTimeSeconds": elapsed
        })

    # -----------------------------------------
    # Step 3: Save CSV
    # -----------------------------------------

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:

        f.write(
            "K,Recall@100,Matching,ProcessingTimeSeconds\n"
        )

        for row in results:

            f.write(
                f"{row['K']},"
                f"{row['Recall@100']:.2f},"
                f"{row['Matching']},"
                f"{row['ProcessingTimeSeconds']:.6f}\n"
            )

    print("=" * 60)
    print(f"Results saved to: {RESULTS_FILE}")
    print("=" * 60)

    # -----------------------------------------
    # Step 4: Create Recall graph
    # -----------------------------------------

    k_values = [row["K"] for row in results]
    recall_values = [row["Recall@100"] for row in results]

    plt.figure(figsize=(8, 5))

    plt.plot(
        k_values,
        recall_values,
        marker="o"
    )

    plt.xlabel("Space-Saving Capacity (K)")
    plt.ylabel("Recall@100 (%)")
    plt.title("Common Crawl: K vs Recall@100")

    plt.grid(True, alpha=0.3)

    # Ensure the x-axis shows the actual K values clearly
    plt.xticks(k_values)

    plt.tight_layout()

    plt.savefig(
        GRAPH_FILE,
        dpi=300
    )

    plt.close()

    print(f"Graph saved to: {GRAPH_FILE}")

    # -----------------------------------------
    # Final summary
    # -----------------------------------------

    print()
    print("=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    print(
        f"{'K':>8} "
        f"{'Recall@100':>15} "
        f"{'Matching':>12} "
        f"{'Time (s)':>12}"
    )

    for row in results:

        print(
            f"{row['K']:>8} "
            f"{row['Recall@100']:>14.2f}% "
            f"{row['Matching']:>10}/{EVALUATION_K} "
            f"{row['ProcessingTimeSeconds']:>12.4f}"
        )


if __name__ == "__main__":
    main()
