from urllib.request import urlretrieve
import tabulate
import torch
import pandas as pd

def dataframe_to_rst_grid(df):
    """Converts a Pandas DataFrame to a reStructuredText (reST) grid table."""

    def escape_pipe(s):
        """Escapes pipes for reST compatibility."""
        return str(s).replace('|', '\|')

    def pad(s, length):
        """Pads string to a given length."""
        return ' ' + s + ' ' * (length - len(s) - 1)

    # Find the maximum length of the string representation for each column
    col_widths = [max(len(escape_pipe(str(s))) for s in df[col]) + 2 for col in df.columns]

    # Create the header row
    header = '+' + '+'.join('-' * w for w in col_widths) + '+\n'
    header += '|' + '|'.join(pad(escape_pipe(str(col)), col_widths[i]) for i, col in enumerate(df.columns)) + '|\n'
    header += '+' + '+'.join('=' * w for w in col_widths) + '+\n'

    # Create the data rows
    rows = ''
    for _, row in df.iterrows():
        row_str = '|' + '|'.join(pad(escape_pipe(str(val)), col_widths[i]) for i, val in enumerate(row)) + '|\n'
        rows += row_str
        rows += '+' + '+'.join('-' * w for w in col_widths) + '+\n'

    return header + rows

# Download all clock metadata
url = f"https://pyaging.s3.amazonaws.com/clocks/metadata/all_clock_metadata.pt"
file_path = "_static/all_clock_metadata.pt"
urlretrieve(url, file_path)

# Load all clock metadata df
metadata_dict = torch.load("_static/all_clock_metadata.pt")

# Convert to DataFrame and do some processing
df = pd.DataFrame(metadata_dict).T
df = df.reset_index()
df.columns = ["clock_name",] + list(df.columns[1:])
df = df.sort_values("clock_name")
df = df.drop("citation", axis=1)
df = df.drop("implementation_approved_by_author(s)", axis=1)
df = df.loc[:,["clock_name", "data_type", "species", "year", "preprocessing", "postprocessing", "doi", "notes"]]

# Convert to rst format
rst_txt = dataframe_to_rst_grid(df)

# Save it in txt format
file_path = "_static/clock_table.txt"
with open(file_path, 'w') as file:
    file.write(rst_txt)
