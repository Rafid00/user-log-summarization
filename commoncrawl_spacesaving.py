import gzip
import json
from collections import Counter
from urllib.parse import urlparse

from crick import SpaceSaving


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = "data/cdx-00000.gz"

K = 100
MAX_RECORDS = 100_000


# ============================================================
# Host normalisation
# ============================================================

def normalize_host(url):
    try:
        hostname = urlparse(url).hostname

        if hostname is None:
            return None

        hostname = hostname.lower()

        if hostname.startswith("www."):
            hostname = hostname[4:]

        return hostname

    except Exception:
        return None


# ============================================================
# Initialise algorithms
# ============================================================

# Approximate algorithm
space_saving = SpaceSaving(K, dtype="object")

# Exact baseline for evaluation only
exact_counter = Counter()

processed = 0
skipped = 0


# ============================================================
# Read Common Crawl as a stream
# ============================================================

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

            if host is None:
                skipped += 1
                continue

            # ------------------------------------------------
            # Feed the same stream into both methods
            # ------------------------------------------------

            space_saving.update(host)
            exact_counter[host] += 1

            processed += 1

        except Exception:
            skipped += 1


# ============================================================
# Get results
# ============================================================

space_results = space_saving.topk(K)
exact_results = exact_counter.most_common(K)


# ============================================================
# Extract Top-K event names
# ============================================================

space_top_hosts = {
    str(host)
    for host, frequency, error in space_results
}

exact_top_hosts = {
    host
    for host, frequency in exact_results
}


# ============================================================
# Calculate Recall@K
# ============================================================

matching_hosts = space_top_hosts & exact_top_hosts

recall = len(matching_hosts) / K


# ============================================================
# Print experiment summary
# ============================================================

print("=" * 70)
print("COMMON CRAWL + SPACE-SAVING")
print("=" * 70)

print(f"Records processed: {processed:,}")
print(f"Records skipped:   {skipped:,}")
print(f"Unique hosts:      {len(exact_counter):,}")
print(f"K:                 {K}")

print("\n" + "=" * 70)
print(f"TOP {K} HOSTS - SPACE-SAVING")
print("=" * 70)

print(
    f"{'Rank':>4} | "
    f"{'Host':<45} | "
    f"{'Estimated':>10} | "
    f"{'Error':>6} | "
    f"{'Exact':>8}"
)

print("-" * 85)

for rank, (host, frequency, error) in enumerate(space_results, start=1):

    exact_frequency = exact_counter[str(host)]

    print(
        f"{rank:4} | "
        f"{str(host)[:45]:45} | "
        f"{frequency:10} | "
        f"{error:6} | "
        f"{exact_frequency:8}"
    )


# ============================================================
# Accuracy
# ============================================================

print("\n" + "=" * 70)
print("ACCURACY")
print("=" * 70)

print(f"Exact Top-{K} hosts:       {len(exact_top_hosts)}")
print(f"Space-Saving Top-{K}:      {len(space_top_hosts)}")
print(f"Matching hosts:            {len(matching_hosts)}/{K}")
print(f"Recall@{K}:                 {recall:.4f}")
print(f"Recall percentage:         {recall * 100:.2f}%")


# ============================================================
# Exact Top 10 for comparison
# ============================================================

print("\n" + "=" * 70)
print("EXACT TOP 10 HOSTS")
print("=" * 70)

print(
    f"{'Rank':>4} | "
    f"{'Host':<45} | "
    f"{'Exact Count':>11}"
)

print("-" * 70)

for rank, (host, count) in enumerate(exact_results[:10], start=1):

    print(
        f"{rank:4} | "
        f"{host[:45]:45} | "
        f"{count:11}"
    )
