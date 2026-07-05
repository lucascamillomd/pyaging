from urllib.request import urlretrieve
import torch
import pandas as pd

# Download all clock metadata
url = f"https://pyaging.s3.amazonaws.com/clocks/metadata0.1.0/all_clock_metadata.pt"
file_path = "_static/all_clock_metadata.pt"
urlretrieve(url, file_path)

# Load all clock metadata df
metadata_dict = torch.load("_static/all_clock_metadata.pt", weights_only=False)

# Convert to DataFrame and do some processing
df = pd.DataFrame(metadata_dict).T
df = df.sort_values(["approved_by_author", "clock_name"], ascending=[False, True])

# Column order tuned for "choose your clock": what/how it predicts first, then
# provenance and implementation details. reindex tolerates clocks missing a field.
columns = [
    "data_type",
    "species",
    "predicts",
    "unit",
    "tissue",
    "platform",
    "population",
    "model_type",
    "n_features",
    "year",
    "citations",
    "last_author",
    "journal",
    "doi",
    "notes",
    "preprocess",
    "postprocess",
    "reference_values",
    "approved_by_author",
]
df = df.reindex(columns=columns)
df.columns = [
    "Data type",
    "Species",
    "Predicts",
    "Unit",
    "Tissue",
    "Platform",
    "Population",
    "Model type",
    "N features",
    "Year",
    "Citations",
    "Last author",
    "Journal",
    "DOI",
    "Notes",
    "Preprocess",
    "Postprocess",
    "Reference values",
    "Approved by author(s)",
]
df.index.name = "Clock name"

# Save csv
df.to_csv("_static/clock_glossary.csv")
