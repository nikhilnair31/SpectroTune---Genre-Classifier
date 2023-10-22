import os
import librosa
import numpy as np
import soundfile as sf
import librosa.display
import matplotlib.pyplot as plt
from multiprocessing import Pool
import itertools

# Define paths
input_base_dir = "Data/genres_original"
aud_output_base_dir = "Data/genres_noisy"
img_output_base_dir = "Data/spectrogram_images"

def add_noise_to_signal(signal, noise_factor):
    noise = np.random.randn(len(signal))
    augmented_signal = signal + noise_factor * noise
    # Cast back to same data type
    augmented_signal = augmented_signal.astype(type(signal[0]))
    return augmented_signal

# Create a function to generate spectrogram
def generate_spectrogram(params):    
    genre_dir, audio_file = params
    genre = genre_dir.split('/')[-1]
    audio_path = os.path.join(genre_dir, audio_file)
    y, sr = librosa.load(audio_path)
    
    # Compute the spectrogram
    D = librosa.amplitude_to_db(abs(librosa.stft(y)), ref=np.max)
    
    # Plot the spectrogram
    plt.figure(figsize=(4, 3))
    librosa.display.specshow(D, sr=sr)
    plt.axis('off')  # Remove axes
    plt.tight_layout()
    
    # Save the figure as an image
    output_genre_dir = os.path.join(img_output_base_dir, genre)
    if not os.path.exists(output_genre_dir):
        os.makedirs(output_genre_dir)
    output_image_path = os.path.join(output_genre_dir, audio_file.replace('.wav', '.png'))
    plt.savefig(output_image_path)

    # Close the figure to free up memory
    plt.close()

# Create a function to generate spectrogram
def genearte_noisy_wavs(params):
    genre_dir, audio_file = params
    genre = genre_dir.split('/')[-1]
    audio_path = os.path.join(genre_dir, audio_file)

    # Load the clean audio file
    audio_data, sampling_rate = librosa.load(audio_path)
    
    # Add noise to the audio data
    noisy_audio_data = add_noise_to_signal(audio_data, 0.03)

    # Load the noise audio file
    noise_data, _ = librosa.load('Data/noise.wav')

    # Match the length of noise to the clean audio
    if len(noise_data) > len(noisy_audio_data):
        noise_data = noise_data[0:len(noisy_audio_data)]
    elif len(noise_data) < len(noisy_audio_data):
        # If the noise file is short, repeat it
        repeats = len(noisy_audio_data) // len(noise_data) + 1
        noise_data = np.tile(noise_data, repeats)
        noise_data = noise_data[0:len(noisy_audio_data)]
    
    # Mix audio and noise, scaling down to prevent clipping
    real_world_audio_data = audio_data + 0.03 * noise_data

    # Save the figure as an image
    output_genre_dir = os.path.join(aud_output_base_dir, genre)
    if not os.path.exists(output_genre_dir):
        os.makedirs(output_genre_dir)
    output_aud_path = os.path.join(output_genre_dir, audio_file)

    # Save the noisy data to a new .wav file
    sf.write(output_aud_path, real_world_audio_data, sampling_rate)
    
    # Now generate spectrogram
    generate_spectrogram(params)

if __name__ == '__main__':

    # Ensure the output directory exists
    if not os.path.exists(aud_output_base_dir):
        os.makedirs(aud_output_base_dir)
    if not os.path.exists(img_output_base_dir):
        os.makedirs(img_output_base_dir)

    # Prepare the set of parameters for processes
    params = []
    for genre_dir in os.listdir(input_base_dir):
        genre_path = os.path.join(input_base_dir, genre_dir)
        if os.path.isdir(genre_path):
            for audio_file in os.listdir(genre_path):
                params.append([genre_path, audio_file])

    # Generate spectrograms in parallel
    with Pool() as p:
        p.map(genearte_noisy_wavs, params)

    print("Spectrogram images generated successfully!")