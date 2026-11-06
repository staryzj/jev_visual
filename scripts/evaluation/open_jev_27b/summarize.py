"""Summarize local prediction or measurement JSONL files by shared UID."""

import argparse
import json
from pathlib import Path

from .metrics import distribution, pair, stats, summarize_rows


def read_rows(paths):
    rows = {}
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    if row["uid"] in rows:
                        raise ValueError(f"Duplicate UID: {row['uid']}")
                    rows[row["uid"]] = row
    return rows


def subset(rows, task):
    if task == "scienceqa_image":
        return [row for row in rows if row["dataset"] == "scienceqa" and row["has_image"]]
    return [row for row in rows if row["dataset"] == task]


def summarize(rows, expected, summarize_one):
    tasks = {row["dataset"] for row in expected.values()} | {"scienceqa_image"}
    values = list(rows.values())
    targets = list(expected.values())
    return {
        "overall": summarize_one(values, len(targets)),
        "tasks": {
            task: summarize_one(subset(values, task), len(subset(targets, task)))
            for task in sorted(tasks)
        },
    }


def matched_rows(left, right):
    return [
        (left[uid], right[uid])
        for uid in sorted(left.keys() & right.keys())
        if all(row.get("supported", True) and not row.get("execution_error", False)
               for row in (left[uid], right[uid]))
    ]


def compare_efficiency(pairs):
    if not pairs:
        return None
    left_ms = sum(left["e2e_ms"] for left, _ in pairs)
    right_ms = sum(right["e2e_ms"] for _, right in pairs)
    return {
        "common_UID_N": len(pairs),
        "latency_sum_ratio_comparator_over_route": right_ms / left_ms,
        "route_generated_tokens": distribution([left["generated_tokens"] for left, _ in pairs]),
        "comparator_generated_tokens": distribution([right["generated_tokens"] for _, right in pairs]),
    }


def validate(rows, expected):
    for uid, row in rows.items():
        if uid not in expected:
            raise ValueError(f"Unexpected UID: {uid}")
        for field in ("label", "candidate_N", "dataset", "cluster", "has_image"):
            if row[field] != expected[uid][field]:
                raise ValueError(f"UID {uid}: mismatched {field}")


def report(route, reference, expected, mode):
    if not expected:
        raise ValueError("The expected UID manifest is empty")
    validate(route, expected)
    validate(reference, expected)
    summarize_one = summarize_rows if mode == "efficiency" else stats
    final = all(
        rows.keys() == expected.keys()
        and not any(row.get("execution_error", False) for row in rows.values())
        for rows in (route, reference)
    )
    result = {
        "final": final,
        "route": summarize(route, expected, summarize_one),
        "comparator": summarize(reference, expected, summarize_one),
    }
    if final:
        pairs = matched_rows(route, reference)
        if pairs:
            result["paired_accuracy"] = pair(*zip(*pairs))
        if mode == "efficiency":
            result["paired_efficiency"] = compare_efficiency(pairs)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("accuracy", "efficiency"))
    parser.add_argument("--route", type=Path, nargs="+", required=True)
    parser.add_argument("--reference", type=Path, nargs="+", required=True)
    parser.add_argument("--expected", type=Path, required=True,
                        help="JSONL manifest with uid, dataset, cluster, label, candidate_N and has_image")
    args = parser.parse_args()
    result = report(read_rows(args.route), read_rows(args.reference),
                    read_rows([args.expected]), args.mode)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["final"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
