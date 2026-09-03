"""Generic grouped, stratified partitioning — used for REAL-Colon video-level splits.

We must never let frames from one video land in more than one of
{query, reference, cal}: near-duplicate consecutive frames would leak.
"""
from __future__ import annotations

import random
from collections import defaultdict


def grouped_stratified_split(
    groups: list[str],
    strata: list,
    sizes: dict[str, int],
    seed: int = 0,
) -> dict[str, list[str]]:
    """Partition unique `groups` into named parts of the requested sizes, keeping
    the stratum mix of each part as even as possible.

    groups : one id per unit (e.g. video name); duplicates collapsed
    strata : same length as `groups`, the stratum key for each
    sizes  : {part_name: n_groups}; sum must be <= number of unique groups
    """
    uniq = {}
    for g, s in zip(groups, strata):
        uniq.setdefault(g, s)
    items = sorted(uniq.items())
    rng = random.Random(seed)
    rng.shuffle(items)

    by_stratum: dict = defaultdict(list)
    for g, s in items:
        by_stratum[s].append(g)

    total = sum(sizes.values())
    assert total <= len(items), f"want {total} groups, have {len(items)}"

    parts: dict[str, list[str]] = {k: [] for k in sizes}
    order = list(sizes.keys())
    # round-robin draw from each stratum so parts stay balanced
    pool = []
    for s in sorted(by_stratum):
        pool.extend(by_stratum[s])
    quotas = dict(sizes)
    i = 0
    for g in pool:
        # place into the still-hungriest part (largest remaining quota)
        cand = sorted(order, key=lambda k: quotas[k], reverse=True)
        for k in cand:
            if quotas[k] > 0:
                parts[k].append(g)
                quotas[k] -= 1
                break
        i += 1
        if sum(quotas.values()) == 0:
            break
    return {k: sorted(v) for k, v in parts.items()}
