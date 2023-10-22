import os
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

base_directory = "Data/spectrogram_images/genres_original"
folders = ["blues", "classical", "country", "disco"]

for folder in os.listdir(base_directory):
    folder_path = os.path.join(base_directory, folder)
    num_files = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
    print(f"{folder}: {num_files} files")