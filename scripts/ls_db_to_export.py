#!/usr/bin/env python
"""Reconstruct a Label Studio export straight from its SQLite DB, keeping the
SKIP-vs-negative distinction that a JSON-MIN export throws away.

Why this exists: `ls_to_yolo.py` treats a task with no regions as an implicit
NEGATIVE (empty label → teaches the detector to reject rulers/mounts). But in
Label Studio a *skipped* task (annotator pressed Skip → `was_cancelled=1`, "already
cropped / unusable") ALSO has no regions, and a JSON-MIN export cannot tell the two
apart — so skips would silently poison the training set as false negatives. The DB
records `was_cancelled`, so we read it directly:

  * non-cancelled completion, with regions  → a positive page (its boxes)
  * non-cancelled completion, empty result  → a real negative (empty label file)
  * only-cancelled (skipped) task           → EXCLUDED entirely

Emits the JSON list `ls_to_yolo.py` consumes: one object per task with `filename`
(the abs path stamped into `data.filename`) and `label` (the list of LS region
`value` dicts — rectangles ±rotation or polygons, in percentages).

    python scripts/ls_db_to_export.py \
        --db ~/Library/'Application Support'/label-studio/label_studio.sqlite3 \
        --projects 3 4 --out /tmp/frag_v3_export.json
    python scripts/ls_to_yolo.py --export /tmp/frag_v3_export.json \
        --out runs/zones/frag-obb-v3 --obb --val-frac 0.15
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path


def region_values(result_json: str) -> list[dict]:
    """The LS region `value` dicts from a completion's `result` array.

    Keeps only items that carry a box/polygon footprint (rectangle `width` or
    polygon `points`); ignores relations, per-region labels, etc."""
    try:
        result = json.loads(result_json) if result_json else []
    except (json.JSONDecodeError, TypeError):
        return []
    out = []
    for item in result:
        val = item.get("value") if isinstance(item, dict) else None
        if isinstance(val, dict) and ("width" in val or "points" in val):
            out.append(val)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, required=True, help="label_studio.sqlite3 path.")
    ap.add_argument("--projects", type=int, nargs="+", required=True,
                    help="Project ids to pool (e.g. 3 4).")
    ap.add_argument("--out", type=Path, required=True, help="Output export JSON.")
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    placeholders = ",".join("?" for _ in args.projects)
    # Latest NON-CANCELLED completion per task (a later real annotation supersedes an
    # earlier one; a task with only cancelled completions never appears here at all).
    rows = con.execute(f"""
        SELECT t.id,
               json_extract(t.data, '$.filename')   AS filename,
               json_extract(t.data, '$.collection') AS collection,
               c.result
        FROM task t
        JOIN task_completion c ON c.task_id = t.id
        WHERE t.project_id IN ({placeholders})
          AND c.was_cancelled = 0
          AND c.id = (SELECT MAX(c2.id) FROM task_completion c2
                      WHERE c2.task_id = t.id AND c2.was_cancelled = 0)
    """, args.projects).fetchall()
    con.close()

    tasks, per_coll, n_pos, n_neg, n_nofile = [], Counter(), 0, 0, 0
    for _id, filename, collection, result in rows:
        if not filename:
            n_nofile += 1
            continue
        regions = region_values(result)
        tasks.append({"filename": filename, "label": regions})
        per_coll[collection] += 1
        if regions:
            n_pos += 1
        else:
            n_neg += 1

    args.out.write_text(json.dumps(tasks, indent=0))
    print(f"[ls_db_to_export] {len(tasks)} tasks → {args.out}")
    print(f"  {n_pos} with zones, {n_neg} negatives"
          + (f", {n_nofile} dropped (no data.filename)" if n_nofile else ""))
    print("  per collection: " + ", ".join(
        f"{k}={v}" for k, v in sorted(per_coll.items(), key=lambda kv: str(kv[0]))))


if __name__ == "__main__":
    main()
