from urllib.request import urlretrieve
import torch
import pandas as pd

# Download all clock metadata
url = f"https://pyaging.s3.amazonaws.com/clocks/metadata0.1.0/all_clock_metadata.pt"
file_path = "_static/all_clock_metadata.pt"
urlretrieve(url, file_path)

# Load all clock metadata df
metadata_dict = torch.load("_static/all_clock_metadata.pt")

# Convert to DataFrame and do some processing
df = pd.DataFrame(metadata_dict).T
df = df.sort_values(["approved_by_author", "clock_name"], ascending=[False, True])
df = df.loc[
    :,
    [
        "data_type",
        "species",
        "year",
        "approved_by_author",
        "doi",
        "notes",
        "preprocess",
        "postprocess",
        "reference_values",
    ],
]
df.columns = [
    "Data type",
    "Species",
    "Year",
    "Approved by author(s)",
    "DOI",
    "Miscellaneous notes",
    "Preprocess",
    "Postprocess",
    "Reference values",
]
df.index.name = "Clock name"

# Save csv
df.to_csv("_static/clock_glossary.csv")
