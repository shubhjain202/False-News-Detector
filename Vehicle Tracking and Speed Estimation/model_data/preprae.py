import os

# Path to the directory containing the folders
path_to_folders = 'crops_test'

# List all directories in the path
folders = os.listdir(path_to_folders)

# Sort the folders in alphabetical order
folders.sort()

# Rename the folders
for index, folder in enumerate(folders, start=1):
    # Generate the new name for the folder based on its index in the sorted list
    new_name = f'{index}'  # Padding the index with zeros for a three-digit number
    
    # Construct the old and new paths
    old_path = os.path.join(path_to_folders, folder)
    new_path = os.path.join(path_to_folders, new_name)
    
    # Rename the folder
    os.rename(old_path, new_path)
