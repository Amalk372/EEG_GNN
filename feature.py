import os
import numpy as np
import mne
import torch
from torch_geometric.data import Data

# Directory containing cleaned .fif files
cleaned_data_dir = r'C:\Users\amalk\Desktop\Main project\new\cleaned_data'
output_feature_dir = r'C:\Users\amalk\Desktop\Main project\new\features'

# Ensure output directories exist
os.makedirs(output_feature_dir, exist_ok=True)

# List all .fif files in the directory
fif_files = [f for f in os.listdir(cleaned_data_dir) if f.endswith('.fif')]

# Function to extract features and save to file
def extract_features_and_save(file_path, output_path):
    raw = mne.io.read_raw_fif(file_path, preload=True)
    
    # Compute power spectral density (PSD) using multitaper method for raw data
    psds, freqs = mne.time_frequency.psd_array_multitaper(raw.get_data(), sfreq=raw.info['sfreq'], fmin=1., fmax=40., adaptive=True, normalization='full', verbose=0)
    
    # Average PSD across channels
    psd_mean = np.mean(psds, axis=0)
    
    # Extract features (e.g., mean power in different bands)
    delta_power = psd_mean[(freqs >= 1) & (freqs < 4)].mean()
    theta_power = psd_mean[(freqs >= 4) & (freqs < 8)].mean()
    alpha_power = psd_mean[(freqs >= 8) & (freqs < 12)].mean()
    beta_power = psd_mean[(freqs >= 12) & (freqs < 30)].mean()

    features = np.array([delta_power, theta_power, alpha_power, beta_power])
    
    # Save features to file
    feature_filename = os.path.basename(file_path).replace('.fif', '_features.npy')
    feature_path = os.path.join(output_path, feature_filename)
    np.save(feature_path, features)

# Process each cleaned .fif file
for fif_file in fif_files:
    file_path = os.path.join(cleaned_data_dir, fif_file)
    extract_features_and_save(file_path, output_feature_dir)

print("Feature extraction completed and saved to:", output_feature_dir)
