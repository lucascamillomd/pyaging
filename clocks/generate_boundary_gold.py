#!/usr/bin/env python3
"""Regenerate the boundary gold-standard expectations.

Running this is an intentional act, not part of any build. The values it writes
are the assertions in ``tests/predict/test_boundary_gold_standard.py``: if a
regenerated value differs from the pinned one, a clock's predictions have moved
and the diff is the evidence. Review it before pasting anything in; never
regenerate to make a failing test pass.

Predictions come from the local ``clocks/weights/*.pt`` build output, the same
source the test asserts against, so the pinned values describe the clocks in
this working tree rather than whatever is currently published on HuggingFace.

Usage
-----
    uv run python clocks/generate_boundary_gold.py

Writes ``boundary_gold.json``; paste its contents into the test's
``boundary_gold_standard_dict`` and delete the file.
"""

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from predict.test_boundary_gold_standard import WEIGHTS_DIR, predict_at_boundaries  # noqa: E402


def main() -> int:
    names = sorted(path.stem for path in WEIGHTS_DIR.glob("*.pt"))
    if not names:
        print(f"no weights in {WEIGHTS_DIR}; run the clock notebooks first", file=sys.stderr)
        return 1

    results = {}
    non_finite = []
    for clock_name in names:
        results[clock_name] = predict_at_boundaries(clock_name)
        if not all(math.isfinite(value) for value in results[clock_name]):
            non_finite.append(clock_name)
        print(clock_name, results[clock_name])

    # json cannot round-trip nan/inf portably, so they are written as strings and
    # spelled out by hand in the test, next to the reason each one is non-finite.
    serialisable = {
        name: [value if math.isfinite(value) else repr(value) for value in values] for name, values in results.items()
    }
    Path("boundary_gold.json").write_text(json.dumps(serialisable, indent=4, sort_keys=True))
    print(f"\n{len(non_finite)} clock(s) non-finite at a boundary: {non_finite}")
    print("wrote boundary_gold.json — paste into tests/predict/test_boundary_gold_standard.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
