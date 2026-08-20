#!/usr/bin/env python3
"""One-off v0.5.0 migration for saved clock weights.

Renames legacy covariate features to their harmonized package-wide names.
Task 11 extends this script to also write feature units.
"""

import argparse
from pathlib import Path

import torch

LEGACY_FEATURE_NAMES = {"Female": "female", "Age": "age"}


def rename_features(features):
    """Return ``features`` with legacy covariate names replaced."""
    renamed = [LEGACY_FEATURE_NAMES.get(feature, feature) for feature in features]
    for old, new in LEGACY_FEATURE_NAMES.items():
        if old in features and new in features:
            raise ValueError(f"cannot rename {old!r}: {new!r} is already present")
    return renamed


def patch_weight_file(path: Path) -> bool:
    """Apply the v0.5.0 migration to one saved clock; return True if it changed."""
    model = torch.load(path, weights_only=False)
    renamed = rename_features(model.features)
    if renamed == list(model.features):
        return False
    model.features = renamed
    torch.save(model, path)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights_dir", type=Path, help="directory of .pt clock weights")
    arguments = parser.parse_args()

    changed = [path.name for path in sorted(arguments.weights_dir.glob("*.pt")) if patch_weight_file(path)]
    print(f"patched {len(changed)} clocks: {', '.join(changed) if changed else 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
