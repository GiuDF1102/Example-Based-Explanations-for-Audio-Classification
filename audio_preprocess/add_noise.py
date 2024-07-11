from audio_preprocess import AudioPreprocessor
import soundfile as sf
import numpy as np
import math
import os 
import matplotlib.pyplot as plt

# Base directory of original audio files
BASE_DIR = 'data/genres_original/test'

# Parameters for AudioPreprocessor
SAMPLE_RATE = 22050
DURATION = 30
HOP_LENGTH_FACTOR = 7.45
N_FFT = 1599 * 2
MONO = True
AUDIO_PREPROCESSOR = AudioPreprocessor(SAMPLE_RATE, DURATION, HOP_LENGTH_FACTOR, N_FFT)

# Function to add noise to audio at a given SNR level
def add_noise_to_audio(audio, snr_db):
    rms_signal = math.sqrt(np.mean(audio ** 2))
    rms_noise = rms_signal / (10 ** (snr_db / 20))
    noise = np.random.normal(0, rms_noise, audio.shape[0])
    return audio + noise

# Define SNR levels for testing
snr_levels = [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

for snr in snr_levels:
    for subfolder in os.listdir(BASE_DIR):
        subfolder_path = os.path.join(BASE_DIR, subfolder)
        if os.path.isdir(subfolder_path):
            for audio_file in os.listdir(subfolder_path):
                if audio_file.endswith('.wav'):
                    audio_file_path = os.path.join(subfolder_path, audio_file)
                    
                    # Load and preprocess the audio file
                    audio_section = AUDIO_PREPROCESSOR.load_audio(audio_file_path, select_section=False)
                    audio_section = AUDIO_PREPROCESSOR.normalize_amplitude(audio_section)
                    
                    # Add noise to the audio
                    noisy_audio = add_noise_to_audio(audio_section, snr)
                    
                    # Compute the spectrogram of the noisy audio
                    spectrogram = AUDIO_PREPROCESSOR.compute_log_spectrogram(noisy_audio)
                    
                    # Create the corresponding output directory
                    output_dir = os.path.join('data', f'noisy_1600_224_snr{snr}', subfolder)
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Save the spectrogram image
                    output_filename = f'{os.path.splitext(audio_file)[0]}.png'
                    output_file_path = os.path.join(output_dir, output_filename)
                    plt.imsave(output_file_path, spectrogram, cmap='gray')
