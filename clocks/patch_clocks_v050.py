#!/usr/bin/env python3
"""One-off v0.5.0 migration for saved clock weights.

Renames legacy covariate features to their harmonized package-wide names.
Task 11 extends this script to also write feature units.

Each file is rewritten individually, so an exception part-way through a
directory leaves the files already processed in their patched state. That is
safe to recover from: the patch is idempotent, so the script can simply be
re-run over the same directory.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import torch

LEGACY_FEATURE_NAMES = {"Female": "female", "Age": "age"}


def rename_features(features):
    """Replace legacy covariate names with their harmonized equivalents.

    Parameters
    ----------
    features : list of str
        Feature names as stored on a clock, in model input order.

    Returns
    -------
    list of str
        ``features`` with every legacy name mapped through
        :data:`LEGACY_FEATURE_NAMES`. Order and length are preserved.

    Raises
    ------
    ValueError
        If renaming would make a harmonized name appear more than once,
        either because both spellings are present (``["Female", "female"]``)
        or because a legacy name is repeated (``["Female", "Female"]``).
        Such a list cannot be renamed without losing a distinct feature.
    """
    renamed = [LEGACY_FEATURE_NAMES.get(feature, feature) for feature in features]
    for new in LEGACY_FEATURE_NAMES.values():
        if renamed.count(new) > 1:
            raise ValueError(f"cannot rename to {new!r}: it is already present {renamed.count(new)} times")
    return renamed


def patch_weight_file(path: Path) -> bool:
    """Apply the v0.5.0 migration to one saved clock.

    The file is replaced atomically: the patched model is written to a
    temporary file in the same directory and then moved over the original, so
    a failed write can never truncate the existing weights.

    Parameters
    ----------
    path : pathlib.Path
        Path to a ``.pt`` clock saved by one of the ``clocks/notebooks``.

    Returns
    -------
    bool
        True if the file was rewritten, False if it already used the
        harmonized names. Re-running over a patched file returns False.

    Raises
    ------
    ValueError
        If the saved object has no usable ``features`` list, or if its
        features cannot be renamed unambiguously.
    """
    model = torch.load(path, weights_only=False)
    features = getattr(model, "features", None)
    if features is None:
        raise ValueError(f"{path}: saved object has no 'features' list to migrate")

    renamed = rename_features(features)
    if renamed == list(features):
        return False

    model.features = renamed
    descriptor, temporary_path = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    os.close(descriptor)
    try:
        torch.save(model, temporary_path)
        os.replace(temporary_path, path)
    except BaseException:
        Path(temporary_path).unlink(missing_ok=True)
        raise
    return True


def main() -> int:
    """Patch every clock in a weights directory and report what changed.

    Returns
    -------
    int
        0 on success, 1 if the directory contains no ``.pt`` files, which
        almost always means the path is wrong rather than that there was
        nothing to do.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights_dir", type=Path, help="directory of .pt clock weights")
    arguments = parser.parse_args()

    paths = sorted(arguments.weights_dir.glob("*.pt"))
    if not paths:
        print(f"no .pt files found in {arguments.weights_dir}", file=sys.stderr)
        return 1

    changed = [path.name for path in paths if patch_weight_file(path)]
    print(f"scanned {len(paths)} clocks, patched {len(changed)}: {', '.join(changed) if changed else 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
