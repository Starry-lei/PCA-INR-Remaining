import os

# Define the directory to scan for .npy files
directory = './'

# Path to the text file where you want to save the names
output_file_path = './names_list.txt'

# List to store extracted names
extracted_names = []

# Scan the directory for .npy files
for filename in os.listdir(directory):
    if filename.endswith('.npy'):
        # Extract the part of the filename before ".npy"
        part_name = filename[:-4]
        extracted_names.append(part_name)

# Save the extracted names to a text file
with open(output_file_path, 'w') as file:
    for name in extracted_names:
        file.write(name + '\n')

print(f'Extracted names have been saved to {output_file_path}')

