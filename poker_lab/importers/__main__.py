"""Convert completed PHH hand histories to JSON Lines, once (no folder watching).

    python -m poker_lab.importers DATA_DIR [MORE ...] --out hands.jsonl.gz [--no-rake]

Prints how many hands were imported and why the others were rejected; ``--rejections``
writes one line per rejected hand (file, index, reason, detail) as JSON Lines.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from poker_lab.importers.phh import import_phh
from poker_lab.schema import write_jsonl


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m poker_lab.importers", description=__doc__)
    parser.add_argument("paths", nargs="+", help=".phh/.phhs files or directories")
    parser.add_argument("--out", required=True, help="output .jsonl or .jsonl.gz")
    parser.add_argument("--ids", choices=("path", "field"), default="path")
    parser.add_argument(
        "--no-rake",
        action="store_true",
        help="finishing stacks must match the replay exactly (tournaments, research data)",
    )
    parser.add_argument("--rejections", help="write rejected hands to this .jsonl file")
    args = parser.parse_args(argv)
    report = import_phh(args.paths, ids=args.ids, allow_rake=not args.no_rake)
    write_jsonl(report.hands, args.out)
    if args.rejections:
        with open(args.rejections, "w", encoding="utf-8", newline="\n") as fh:
            for r in report.rejected:
                record = {"file": r.file, "index": r.index, "reason": r.reason.value}
                fh.write(json.dumps({**record, "detail": r.detail}) + "\n")
    print(report.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
