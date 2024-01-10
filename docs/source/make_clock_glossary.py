from urllib.request import urlretrieve
import torch
import pandas as pd

# Download all clock metadata
url = f"https://pyaging.s3.amazonaws.com/clocks/metadata/all_clock_metadata.pt"
file_path = "_static/all_clock_metadata.pt"
urlretrieve(url, file_path)

# Load all clock metadata df
metadata_dict = torch.load("_static/all_clock_metadata.pt")

# Convert to DataFrame and do some processing
df = pd.DataFrame(metadata_dict).T
df.index.name = "clock_name"
df = df.sort_values(["implementation_approved_by_author(s)", "year", "clock_name"], ascending=[False, False, True])
df = df.loc[:,["data_type", "species", "year", "implementation_approved_by_author(s)", "doi", "notes", "preprocessing", "postprocessing",]]
df.columns = ["Data type", "Species", "Year", "Approved by author(s)", "DOI", "Miscellaneous notes", "Preprocessing", "Postprocessing",]
df.index.name = "Clock name"

# Save csv
df.to_csv("_static/clock_glossary.csv")