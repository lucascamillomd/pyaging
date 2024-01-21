#!/usr/bin/env python3
import os
import torch
import pyaging as pya
import argparse


def merge_and_update_pt_files(version):
    """
    Merges metadata from .pt files into a single dictionary. Also updates the .pt files with the version number.

    Iterates through all .pt files in the 'weights' directory, extracts metadata, and combines it into a dictionary.

    Parameters:
    version (str): The version number to be added to each file's metadata.
    """
    combined_dict = {}

    for filename in os.listdir("weights"):
        if filename.endswith(".pt"):
            file_path = os.path.join("weights", filename)
            try:
                clock = torch.load(file_path)
                clock.version = version
                torch.save(clock, file_path)

                file_data = clock.metadata
                key = file_data["clock_name"]

                # Add additional information if available
                if clock.reference_values is not None:
                    file_data["reference_values"] = True
                if clock.preprocess_name is not None:
                    file_data["preprocess"] = clock.preprocess_name
                if clock.postprocess_name is not None:
                    file_data["postprocess"] = clock.postprocess_name
                if clock.version is not None:
                    file_data["version"] = clock.version

                combined_dict[key] = file_data
                print(f"Added {key} to metadata dictionary.")
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    return combined_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge PT files metadata.")
    parser.add_argument(
        "version", type=str, help="Version number to be added to the metadata."
    )
    args = parser.parse_args()

    combined_dictionary = merge_and_update_pt_files(args.version)
    torch.save(combined_dictionary, "metadata/all_clock_metadata.pt")
    print("Metadata dictionary saved to 'metadata/all_clock_metadata.pt'.")
