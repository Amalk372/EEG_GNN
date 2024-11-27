import mne
import matplotlib.pyplot as plt

# Load .edf file
data_path = r'C:\Users\amalk\Desktop\Main project\new\dataset\H S1 EC.edf'

raw = mne.io.read_raw_edf(data_path, preload=True)

# Plot data
raw.plot(n_channels=10, duration=5, start=0)
plt.show()  # This should keep the plot window open
