import os
import mne

# Directory containing .edf files
data_dir = r'C:\Users\amalk\Desktop\Main project\new\dataset'
output_dir = r'C:\Users\amalk\Desktop\Main project\new\cleaned_data'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# List all .edf files in the directory
edf_files = [f for f in os.listdir(data_dir) if f.endswith('.edf')]

# Function to convert .edf to .fif and preprocess
def convert_and_preprocess(file_path, output_path):
    raw = mne.io.read_raw_edf(file_path, preload=True)
    
    # Convert to .fif format
    fif_filename = os.path.basename(file_path).replace('.edf', '.fif')
    fif_path = os.path.join(output_path, fif_filename)
    raw.save(fif_path, overwrite=True)
    
    # Reload the .fif file
    raw = mne.io.read_raw_fif(fif_path, preload=True)
    
    # Apply band-pass filter
    raw.filter(1., 40.)
    
    # Set up and fit ICA
    ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
    ica.fit(raw)
    
    # Manually specify components to exclude based on prior knowledge or heuristics
    ica.exclude = [0, 1]  # Example component indices to exclude
    
    # Apply ICA to remove artifacts
    reconstructed_raw = raw.copy()
    ica.apply(reconstructed_raw)
    
    # Save the cleaned data back to .fif
    cleaned_fif_filename = os.path.basename(file_path).replace('.edf', '_cleaned.fif')
    cleaned_fif_path = os.path.join(output_path, cleaned_fif_filename)
    reconstructed_raw.save(cleaned_fif_path, overwrite=True)

# Process each .edf file
for edf_file in edf_files:
    file_path = os.path.join(data_dir, edf_file)
    convert_and_preprocess(file_path, output_dir)

print("All files have been processed and saved to:", output_dir)
