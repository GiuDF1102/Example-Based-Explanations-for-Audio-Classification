import librosa
import numpy as np
import random
import zipfile


class AudioPreprocessor:
    def __init__(self, sample_rate=22050, duration=10, hop_length_factor=31.25, n_fft=512, mono = True):
        self.sample_rate = sample_rate
        self.duration = duration
        self.hop_length_factor = hop_length_factor
        self.n_fft = n_fft
        self.mono = mono
        
    def load_audio(self, file_path):
        # Load audio file
        signal = librosa.load(file_path, 
                              sr=self.sample_rate, 
                              duration=self.duration, 
                              mono=self.mono)[0]
        
        start_position = random.randint(0, len(signal) - self.sample_rate * self.duration)
        audio_section = signal[start_position:start_position + self.sample_rate * self.duration]

        return audio_section

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
    
def extract_zip(zip_file, extract_to):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)