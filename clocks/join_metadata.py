import os
import torch

def merge_pt_files(metadata_keys):
    combined_dict = {}

    # Iterate through all files in the current directory
    for filename in os.listdir('weights'):
        # Check if the file is a .pt file
        if filename.endswith('.pt'):
            file_path = os.path.join('weights', filename)
            try:
                # Load the .pt file
                file_data = torch.load(file_path)
                # Check if the loaded data is a dictionary
                if isinstance(file_data, dict):
                    # Use the filename without '.pt' as the key
                    key = filename[:-3]
                    if key == 'all_clock_metadata':
                        continue
                    else:
                        file_data = {k: file_data[k] for k in metadata_keys if k in file_data}
                        if file_data['reference_feature_values']:
                            file_data['reference_feature_values'] = True
                        if file_data['preprocessing']:
                            file_data['preprocessing'] = file_data['preprocessing']['name']
                        if file_data['postprocessing']:
                            file_data['postprocessing'] = file_data['postprocessing']['name']
                        combined_dict[key] = file_data
                        print(f"Added {key} to metadata dictionary.")
                else:
                    print(f"Warning: {filename} is not a dictionary.")
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    return combined_dict

metadata_keys = [
    'clock_name',
    'data_type',
    'model_class',
    'species',
    'year',
    'approved_by_author',
    'citation',
    'doi',
    "notes",
    'preprocessing',
    'postprocessing',
    'reference_feature_values'
]

# Use the function and get the combined dictionary
combined_dictionary = merge_pt_files(metadata_keys)

torch.save(combined_dictionary, 'metadata/all_clock_metadata.pt')