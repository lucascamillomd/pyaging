"""Sync per-clock Hugging Face repositories under the pyaging organization.

Each clock gets its own model repo (``pyaging/<clock_name>``) containing the
weight file, a ``config.json`` with the audited clock metadata (also the file
the Hub counts as a download), and a generated model card. Per-repo download
counters are what power the docs' popularity ranking.

Usage:
    python clocks/hf_repo_sync.py                 # sync every clock
    python clocks/hf_repo_sync.py horvath2013 ... # sync selected clocks
    python clocks/hf_repo_sync.py --tag v0.3.2    # also (re)tag each repo
"""

import argparse
import hashlib
import json
import sys
import time
from contextlib import suppress
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.errors import HfHubHTTPError, RevisionNotFoundError

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = ROOT / "clocks" / "weights"
METADATA_FILE = ROOT / "clocks" / "metadata" / "clock_metadata.json"
OWNER = "pyaging"
DOCS_URL = "https://pyaging.readthedocs.io"

CARD_TEMPLATE = """---
license: mit
library_name: pyaging
tags:
- pyaging
- aging-clock
- biology
{extra_tags}---

# {display_name}

{notes}

| | |
|---|---|
| **Predicts** | {predicts} |
| **Species** | {species} |
| **Tissue** | {tissue} |
| **Data type** | {data_type} |
| **Model type** | {model_type} |
| **Year** | {year} |

## Use with pyaging

```python
import pyaging as pya

pya.pred.predict_age(adata, ["{display_name}"])
```

Browse every clock in the [pyaging Clock Catalogue]({docs_url}).

## Citation

{citation}

{doi}
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remote_weight_sha(api: HfApi, repo_id: str, filename: str) -> str | None:
    with suppress(Exception):
        paths = api.get_paths_info(repo_id, [filename])
        if paths and paths[0].lfs is not None:
            return paths[0].lfs.sha256
    return None


def _display_name(metadata: dict) -> str:
    return metadata.get("display_name") or metadata["clock_name"]


def _build_card(metadata: dict) -> str:
    def join(field):
        value = metadata.get(field)
        if isinstance(value, list):
            return ", ".join(str(item) for item in value)
        return str(value) if value else "-"

    extra_tags = ""
    if metadata.get("data_type"):
        slug = str(metadata["data_type"]).lower().replace(" ", "-")
        extra_tags = f"- {slug}\n"
    return CARD_TEMPLATE.format(
        extra_tags=extra_tags,
        display_name=_display_name(metadata),
        notes=metadata.get("notes") or "",
        predicts=join("predicts"),
        species=join("species"),
        tissue=join("tissue"),
        data_type=join("data_type"),
        model_type=join("model_type"),
        year=metadata.get("year") or "-",
        citation=metadata.get("citation") or "",
        doi=metadata.get("doi") or "",
        docs_url=DOCS_URL,
    )


def sync_clock(api: HfApi, clock_name: str, metadata: dict, tag: str | None, tag_only: bool) -> str:
    weight_path = WEIGHTS_DIR / f"{clock_name}.pt"
    repo_id = f"{OWNER}/{clock_name}"
    outcome = "tagged"
    if not tag_only:
        api.create_repo(repo_id, repo_type="model", exist_ok=True)
        operations = [
            CommitOperationAdd("config.json", json.dumps(metadata, indent=1, sort_keys=True).encode()),
            CommitOperationAdd("README.md", _build_card(metadata).encode()),
        ]
        weight_current = _remote_weight_sha(api, repo_id, weight_path.name) == _sha256(weight_path)
        if not weight_current:
            operations.append(CommitOperationAdd(weight_path.name, str(weight_path)))
        api.create_commit(repo_id, operations=operations, commit_message=f"Sync {clock_name}")
        outcome = "uploaded" if not weight_current else "metadata-only"

    if tag:
        with suppress(RevisionNotFoundError, HfHubHTTPError):
            api.delete_tag(repo_id, tag=tag)
        api.create_tag(repo_id, tag=tag)
    return outcome


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clocks", nargs="*", help="clock names to sync (default: all)")
    parser.add_argument("--tag", help="tag to (re)point at the synced revision, e.g. v0.3.2")
    parser.add_argument("--tag-only", action="store_true", help="only move tags; skip uploads")
    args = parser.parse_args()
    if args.tag_only and not args.tag:
        parser.error("--tag-only requires --tag")

    registry = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    available = sorted(path.stem for path in WEIGHTS_DIR.glob("*.pt"))
    selected = args.clocks or available
    unknown = [name for name in selected if name not in available]
    if unknown:
        parser.error(f"no weight file for: {', '.join(unknown)}")

    api = HfApi()
    failures = []
    for index, clock_name in enumerate(selected, start=1):
        metadata = registry.get(clock_name, {"clock_name": clock_name})
        for attempt in range(3):
            try:
                outcome = sync_clock(api, clock_name, metadata, args.tag, args.tag_only)
                print(f"[{index}/{len(selected)}] {clock_name}: {outcome}", flush=True)
                break
            except Exception as error:  # noqa: BLE001 - report and continue the batch
                if attempt == 2:
                    failures.append(clock_name)
                    print(f"[{index}/{len(selected)}] {clock_name}: FAILED ({error})", flush=True)
                else:
                    time.sleep(5 * (attempt + 1))

    if failures:
        print(f"{len(failures)} clock(s) failed: {', '.join(failures)}", flush=True)
        return 1
    print(f"Synced {len(selected)} clock repos under {OWNER}/", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
