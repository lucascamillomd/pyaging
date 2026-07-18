#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch

RUNTIME_METADATA_FIELDS = {
    "version",
    "preprocess",
    "postprocess",
    "reference_values",
}


def merge_clock_metadata(generated_metadata, curated_metadata):
    """Merge generated runtime data into registry-backed curated metadata."""
    merged_metadata = {}

    for clock_name, curated_entry in curated_metadata.items():
        generated_entry = generated_metadata[clock_name]
        merged_entry = {key: value for key, value in curated_entry.items() if key not in RUNTIME_METADATA_FIELDS}
        merged_entry.update({key: generated_entry[key] for key in RUNTIME_METADATA_FIELDS if key in generated_entry})
        merged_metadata[clock_name] = merged_entry

    return merged_metadata


def load_curated_metadata(registry_path):
    """Load and validate the curated JSON metadata registry."""
    if not registry_path.is_file():
        raise FileNotFoundError(f"Curated metadata registry is required: {registry_path}")

    try:
        with registry_path.open(encoding="utf-8") as registry_file:
            metadata = json.load(registry_file)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid curated metadata JSON: {registry_path}") from error
    if not isinstance(metadata, dict):
        raise ValueError("Curated metadata must be a top-level dictionary")

    for clock_name, entry in metadata.items():
        if not isinstance(clock_name, str) or not clock_name or clock_name != clock_name.lower():
            raise ValueError("Every curated metadata key must be a lowercase non-empty string")
        if not isinstance(entry, dict):
            raise ValueError(f"Curated metadata entry for '{clock_name}' must be a dictionary")
        if entry.get("clock_name") != clock_name:
            raise ValueError(f"Curated metadata entry clock_name for '{clock_name}' must match its key")

    return metadata


def _generated_metadata_entry(clock):
    if not isinstance(clock.metadata, dict):
        raise ValueError("Clock metadata must be a dictionary")

    key = clock.metadata.get("clock_name")
    if not isinstance(key, str) or not key:
        raise ValueError("Clock metadata must contain a non-empty string clock_name")

    file_data = {}
    if clock.reference_values is not None:
        file_data["reference_values"] = True
    if clock.preprocess_name is not None:
        file_data["preprocess"] = clock.preprocess_name
    if clock.postprocess_name is not None:
        file_data["postprocess"] = clock.postprocess_name
    if clock.version is not None:
        file_data["version"] = clock.version

    return key, file_data


def preflight_weight_files(weights_dir):
    """Validate the weight set and inspect every file before any rewrite."""
    if not weights_dir.is_dir():
        raise ValueError(f"A non-empty weights directory is required: {weights_dir}")

    weight_paths = sorted(weights_dir.glob("*.pt"))
    if not weight_paths:
        raise ValueError(f"A non-empty weights directory is required: {weights_dir}")

    clock_names = set()
    for weight_path in weight_paths:
        clock = torch.load(weight_path, weights_only=False)
        clock_name, _ = _generated_metadata_entry(clock)
        clock_names.add(clock_name)
        del clock

    expected_clock_names = {weight_path.stem for weight_path in weight_paths}
    if len(clock_names) != len(weight_paths) or clock_names != expected_clock_names:
        raise ValueError(
            "Generated clock names do not match weight filenames: "
            f"expected {sorted(expected_clock_names)}, "
            f"generated {sorted(clock_names)}"
        )

    return weight_paths, clock_names


def merge_and_update_pt_files(version, weight_paths):
    """
    Merges metadata from .pt files into a single dictionary. Also updates the .pt files with the version number.

    Iterates through all .pt files in the 'weights' directory, extracts metadata, and combines it into a dictionary.

    Parameters:
    version (str): The version number to be added to each file's metadata.
    """
    combined_dict = {}

    for weight_path in weight_paths:
        clock = torch.load(weight_path, weights_only=False)
        clock.version = version
        key, file_data = _generated_metadata_entry(clock)
        torch.save(clock, weight_path)
        del clock

        combined_dict[key] = file_data
        print(f"Added {key} to metadata dictionary.")

    return combined_dict


def regenerate_clock_metadata(
    version,
    weights_dir=Path("weights"),
    registry_path=Path("metadata/clock_metadata.json"),
    metadata_path=Path("metadata/all_clock_metadata.pt"),
):
    """Validate inputs, update all weights, and save the merged aggregate."""
    weights_dir = Path(weights_dir)
    registry_path = Path(registry_path)
    metadata_path = Path(metadata_path)
    curated_dictionary = load_curated_metadata(registry_path)

    weight_paths, validated_clock_names = preflight_weight_files(weights_dir)
    registry_clock_names = set(curated_dictionary)
    if validated_clock_names != registry_clock_names:
        raise ValueError(
            "Registry and weight clock names must match exactly: "
            f"registry {sorted(registry_clock_names)}, "
            f"weights {sorted(validated_clock_names)}"
        )

    generated_dictionary = merge_and_update_pt_files(version, weight_paths)

    generated_clock_names = set(generated_dictionary)
    if len(generated_dictionary) != len(weight_paths) or generated_clock_names != validated_clock_names:
        raise ValueError(
            "Generated clock names changed after preflight: "
            f"expected {sorted(validated_clock_names)}, "
            f"generated {sorted(generated_clock_names)}"
        )

    combined_dictionary = merge_clock_metadata(generated_dictionary, curated_dictionary)
    torch.save(combined_dictionary, metadata_path)
    return combined_dictionary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge PT files metadata.")
    parser.add_argument("version", type=str, help="Version number to be added to the metadata.")
    args = parser.parse_args()

    regenerate_clock_metadata(args.version)
    print("Metadata dictionary saved to 'metadata/all_clock_metadata.pt'.")
