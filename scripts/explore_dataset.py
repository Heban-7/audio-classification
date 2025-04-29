import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from glob import glob
from random import choice
import soundfile as sf
import os, sys

sys.path.append(os.path.abspath('..'))
DATA_PATH = "../data/genres/"
GENRES = sorted(os.listdir(DATA_PATH))

def count_genre_samples():
    genre_counts = {}
    for genre in GENRES:
        files = glob(os.path.join(DATA_PATH, genre, "*.wav"))
        genre_counts[genre] = len(files)
    return genre_counts

def plot_genre_distribution(genre_counts):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(genre_counts.keys()), y=list(genre_counts.values()), palette="viridis")
    plt.title("Number of Samples per Genre in GTZAN Dataset")
    plt.xlabel("Genre")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_random_waveform_per_genre():
    plt.figure(figsize=(20, 10))
    colors = plt.cm.tab10.colors  # 10 distinct colors from the 'tab10' colormap

    for i, genre in enumerate(GENRES):
        wav_path = choice(glob(os.path.join(DATA_PATH, genre, "*.wav")))
        signal, sr = librosa.load(wav_path, sr=22050)
        plt.subplot(2, 5, i + 1)
        librosa.display.waveshow(signal, sr=sr, color=colors[i % len(colors)])
        plt.title(genre)
        plt.tight_layout()
    
    plt.suptitle("Random Waveform from Each Genre", fontsize=20)
    plt.subplots_adjust(top=0.88)
    plt.show()


def plot_random_spectrogram_per_genre():
    plt.figure(figsize=(20, 10))
    for i, genre in enumerate(GENRES):
        wav_path = choice(glob(os.path.join(DATA_PATH, genre, "*.wav")))
        signal, sr = librosa.load(wav_path, sr=22050)
        mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        plt.subplot(2, 5, i + 1)
        librosa.display.specshow(mel_spec_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel', cmap='magma')
        plt.title(genre)
        plt.colorbar(format="%+2.0f dB")
        plt.tight_layout()
    plt.suptitle("Mel-Spectrogram from Random Sample per Genre", fontsize=20)
    plt.subplots_adjust(top=0.88)
    plt.show()



def check_audio_consistency(data_dir):
    genres = os.listdir(data_dir)
    consistent_format = True
    sampling_rates = set()
    durations = []
    bit_depths = set()
    num_files = 0

    for genre in genres:
        genre_path = os.path.join(data_dir, genre)
        if os.path.isdir(genre_path):
            for filename in os.listdir(genre_path):
                file_path = os.path.join(genre_path, filename)
                if filename.endswith('.wav'):
                    num_files += 1
                    try:
                        y, sr = librosa.load(file_path, sr=None)
                        sampling_rates.add(sr)
                        duration = librosa.get_duration(y=y, sr=sr)
                        durations.append(duration)
                        with sf.SoundFile(file_path, 'r') as f:
                            bit_depths.add(f.subtype)
                    except Exception as e:
                        print(f"Error processing .wav file: {file_path} - {e}")
                        consistent_format = False
                elif not filename.startswith('.'):
                    consistent_format = False
                    print(f"Found non-wav file: {file_path}") 

    return consistent_format, sampling_rates, durations, bit_depths, num_files


data_directory = '../data/genres'
consistent, s_rates, durs, b_depths, num_f = check_audio_consistency(data_directory)

def file_format_check():
    print("\n--- File Format Check ---")
    if consistent:
        print("All non-hidden files appear to be in .wav format.")
    else:
        print("All non-hidden files appear to be in .wav format.")

def basic_statistics():
    print("\n--- Basic Statistics ---")
    print(f"Total number of .wav files processed: {num_f}")

    if s_rates:
        if len(s_rates) == 1:
            print(f"Sampling rate: {s_rates.pop()} Hz (consistent across all files)")
        else:
            print(f"Sampling rates found: {s_rates} (inconsistent across files)")
    else:
        print("No sampling rate information found.")

def duration():
    if durs:
        min_duration = min(durs)
        max_duration = max(durs)
        avg_duration = sum(durs) / len(durs)
        print(f"Duration: Min={min_duration:.2f}s, Max={max_duration:.2f}s, Avg={avg_duration:.2f}s")
        if min_duration == max_duration:
            print("Durations are consistent across all files.")
        else:
            print("Durations are inconsistent across files.")
    else:
        print("No duration information found.")


def pad_trim_audio(audio, target_length_seconds, sample_rate):
    """Pads or trims an audio array to a target length in seconds."""
    target_length_samples = int(target_length_seconds * sample_rate)
    current_length = len(audio)

    if current_length < target_length_samples:
        # Pad with zeros at the end
        padding = target_length_samples - current_length
        padded_audio = np.pad(audio, (0, padding), 'constant')
        return padded_audio
    elif current_length > target_length_samples:
        # Trim from the end
        trimmed_audio = audio[:target_length_samples]
        return trimmed_audio
    else:
        return audio  # Already the target length

def preprocess_and_save_audio(data_dir, output_dir, target_length_seconds=30):
    """
    Loads audio files from the data directory, pads/trims them to a target length,
    and saves the processed audio to the output directory.

    Args:
        data_dir (str): Path to the main directory containing genre subdirectories.
        output_dir (str): Path to the directory where processed audio will be saved.
        target_length_seconds (int): The target length of the audio in seconds.
    """
    genres = os.listdir(data_dir)
    original_sr = None  # To store the original sampling rate (assuming it's consistent)

    os.makedirs(output_dir, exist_ok=True)

    for genre in genres:
        genre_path = os.path.join(data_dir, genre)
        output_genre_path = os.path.join(output_dir, genre)
        os.makedirs(output_genre_path, exist_ok=True)

        if os.path.isdir(genre_path):
            for filename in os.listdir(genre_path):
                if filename.endswith('.wav'):
                    file_path = os.path.join(genre_path, filename)
                    output_file_path = os.path.join(output_genre_path, filename)

                    try:
                        audio, sr = librosa.load(file_path, sr=None)

                        if original_sr is None:
                            original_sr = sr
                        elif original_sr != sr:
                            continue

                        processed_audio = pad_trim_audio(audio, target_length_seconds, sr)
                        sf.write(output_file_path, processed_audio, sr)

                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")


def bit_depths():
    if b_depths:
        if len(b_depths) == 1:
            print(f"Bit depth/Subtype: {b_depths.pop()} (consistent across all files)")
        else:
            print(f"Bit depths/Subtypes found: {b_depths} (inconsistent across files)")
    else:
        print("No bit depth information found.")