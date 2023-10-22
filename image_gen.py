import os
import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt

# Define paths
input_base_dir = "Data/genres_original"
output_base_dir = "Data/spectrogram_images"

# Ensure the output directory exists
if not os.path.exists(output_base_dir):
    os.makedirs(output_base_dir)

# Loop through each genre
for genre in os.listdir(input_base_dir):
    genre_path = os.path.join(input_base_dir, genre)
    if os.path.isdir(genre_path):
        # Create a similar folder structure for output spectrogram images
        output_genre_dir = os.path.join(output_base_dir, genre)
        if not os.path.exists(output_genre_dir):
            os.makedirs(output_genre_dir)
        
        # Loop through each audio file in the genre directory
        for audio_file in os.listdir(genre_path):
            audio_path = os.path.join(genre_path, audio_file)
            y, sr = librosa.load(audio_path)
            
            # Compute the spectrogram
            D = librosa.amplitude_to_db(abs(librosa.stft(y)), ref=np.max)
            
            # Plot the spectrogram
            plt.figure(figsize=(5, 4))
            librosa.display.specshow(D, sr=sr)
            plt.axis('off')  # Remove axes
            plt.tight_layout()
            
            # Save the figure as an image
            output_image_path = os.path.join(output_genre_dir, audio_file.replace('.wav', '.png'))
            plt.savefig(output_image_path)
            
            # Close the figure to free up memory
            plt.close()

print("Spectrogram images generated successfully!")