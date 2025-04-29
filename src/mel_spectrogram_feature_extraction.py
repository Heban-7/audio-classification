import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split

# Define constants
DATA_PATH = '../data/processed_audio'  # Path to GTZAN dataset
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
SR = 22050  # Sampling rate
SEGMENT_LENGTH = 3  # Segment length in seconds
OVERLAP = 1.5  # Overlap between segments in seconds
N_MELS = 128  # Number of Mel bands
HOP_LENGTH = 512  # Hop length for STFT
N_FFT = 2048  # FFT window size

# Map genres to indices
genre_to_idx = {genre: idx for idx, genre in enumerate(GENRES)}

def extract_segments(file_path, segment_length=SEGMENT_LENGTH, overlap=OVERLAP, sr=SR):
    """
    Extract overlapping segments from an audio file.

    Args:
        file_path (str): Path to the audio file.
        segment_length (float): Length of each segment in seconds.
        overlap (float): Overlap between segments in seconds.
        sr (int): Sampling rate.

    Returns:
        list: List of audio segments as numpy arrays.
    """
    y, sr = librosa.load(file_path, sr=sr)
    segment_samples = int(segment_length * sr)
    step = int((segment_length - overlap) * sr)
    segments = []

    for start in range(0, len(y), step):
        end = start + segment_samples
        if end > len(y):
            # Pad the last segment with zeros if shorter
            segment = np.pad(y[start:], (0, end - len(y)), mode='constant')
        else:
            segment = y[start:end]
        segments.append(segment)

    return segments

def compute_mel_spectrogram(y, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=N_FFT):
    """
    Compute Mel spectrogram for an audio segment.

    Args:
        y (np.array): Audio time series.
        sr (int): Sampling rate.
        n_mels (int): Number of Mel bands.
        hop_length (int): Hop length for STFT.
        n_fft (int): FFT window size.

    Returns:
        np.array: Mel spectrogram in dB scale.
    """
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB

def process_files(file_list, label_list):
    """
    Process a list of audio files to extract segments and compute Mel spectrograms.

    Args:
        file_list (list): List of audio file paths.
        label_list (list): List of corresponding genre indices.

    Returns:
        np.array: Array of Mel spectrograms with shape (num_segments, n_mels, num_frames, 1).
        np.array: Array of labels with shape (num_segments,).
    """
    spectrograms = []
    labels = []

    for file_path, label in zip(file_list, label_list):
        segments = extract_segments(file_path)
        for segment in segments:
            S_dB = compute_mel_spectrogram(segment)
            spectrograms.append(S_dB)
            labels.append(label)
        print(f"Processed: {file_path}")

    # Convert to numpy arrays and add channel dimension
    spectrograms = np.array(spectrograms)  # Shape: (num_segments, n_mels, num_frames)
    spectrograms = np.expand_dims(spectrograms, axis=-1)  # Shape: (num_segments, n_mels, num_frames, 1)
    labels = np.array(labels)

    return spectrograms, labels

def save_data(X, y, split_ratios=(0.8, 0.1, 0.1)):
    """
    Split and save the data into train, validation, and test sets.

    Args:
        X (np.array): Features.
        y (np.array): Labels.
        split_ratios (tuple): Ratios for train, val, test splits.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=1 - split_ratios[0], stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=split_ratios[2] / (split_ratios[1] + split_ratios[2]), stratify=y_temp, random_state=42)

    np.save('../data/mel_spectrogram_feature/X_train.npy', X_train)
    np.save('../data/mel_spectrogram_feature/y_train.npy', y_train)
    np.save('../data/mel_spectrogram_feature/X_val.npy', X_val)
    np.save('../data/mel_spectrogram_feature/y_val.npy', y_val)
    np.save('../data/mel_spectrogram_feature/X_test.npy', X_test)
    np.save('../data/mel_spectrogram_feature/y_test.npy', y_test)

    print(f"Data saved: X_train.shape={X_train.shape}, y_train.shape={y_train.shape}")
    print(f"Data saved: X_val.shape={X_val.shape}, y_val.shape={y_val.shape}")
    print(f"Data saved: X_test.shape={X_test.shape}, y_test.shape={y_test.shape}")

def main():
    # Collect all audio files and their genres
    file_list = []
    genre_list = []

    for genre in GENRES:
        genre_path = os.path.join(DATA_PATH, genre)
        files = [os.path.join(genre_path, f) for f in os.listdir(genre_path) if f.endswith('.wav')]
        file_list.extend(files)
        genre_list.extend([genre] * len(files))

    # Convert genre labels to indices
    idx_list = [genre_to_idx[genre] for genre in genre_list]

    # Process files to extract spectrograms
    X, y = process_files(file_list, idx_list)

    # Save the data
    save_data(X, y)

if __name__ == "__main__":
    main()