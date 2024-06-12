import librosa
import numpy as np
import random
import zipfile
import math

class AudioPreprocessor:
    def __init__(self, sample_rate=22050, duration=10, hop_length_factor=31.25, n_fft=512, mono = True):
        self.sample_rate = sample_rate
        self.duration = duration
        self.hop_length_factor = hop_length_factor
        self.n_fft = n_fft
        self.mono = mono
        
    def load_audio(self, file_path, select_section = True):
        # Load audio file
        signal = librosa.load(file_path, 
                              sr=self.sample_rate, 
                              duration=self.duration, 
                              mono=self.mono)[0]
        if select_section:
            start_position = random.randint(0, len(signal) - self.sample_rate * self.duration)
            audio_section = signal[start_position:start_position + self.sample_rate * self.duration]
            return audio_section
        return signal
    

    def normalize_amplitude(self, audio_signal):
        # Normalize amplitude
        max_val = np.max(np.abs(audio_signal))
        normalized_signal = audio_signal / max_val
        return normalized_signal
    
    
    def compute_log_spectrogram(self, audio_signal):
        # Compute log spectrogram
        hop_length = int(self.sample_rate / self.hop_length_factor)
        log_spec = librosa.amplitude_to_db(
            np.abs(librosa.stft(audio_signal, n_fft=self.n_fft, hop_length=hop_length)),
            ref=np.max
        )
        return log_spec

    def add_noise_to_audio(self, audio, snr_db):
        rms_signal = math.sqrt(np.mean(audio ** 2))
        rms_noise = rms_signal / (10 ** (snr_db / 20))
        noise = np.random.normal(0, rms_noise, audio.shape[0])
        return audio + noise

    
def extract_zip(zip_file, extract_to):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        