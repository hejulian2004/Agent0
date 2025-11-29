from datasets import load_dataset
import os

# --- Configuration ---
# Directory containing the old parquet files
data_dir = "data/deepmath_torl/" 
# Directory where the new, fixed files will be saved
output_dir = "data/deepmath_torl_fixed/"

# List of files to convert
files_to_convert = [
    "train.parquet",
    "test.parquet",
    "math500_test.parquet",
    "aime24_test.parquet",
    "aime25_test.parquet"
]
# -------------------

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

for filename in files_to_convert:
    old_file_path = os.path.join(data_dir, filename)
    new_file_path = os.path.join(output_dir, filename)

    if not os.path.exists(old_file_path):
        print(f"Warning: File not found, skipping: {old_file_path}")
        continue
    
    print(f"Converting {old_file_path}...")
    
    # Load the dataset using pyarrow to bypass the schema issue
    dataset = load_dataset("parquet", data_files={"train": old_file_path})

    # Save it back. This writes a new file with the corrected schema.
    dataset["train"].to_parquet(new_file_path)
    
    print(f"Successfully saved corrected file to: {new_file_path}")

print("\nConversion complete! ✅")