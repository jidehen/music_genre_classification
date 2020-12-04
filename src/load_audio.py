# Load raw mp3 files, extract features with librosa, store in .npy files
from os import write
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import sys


audio_duration = 30
sampling_rate = 22050
input_size = audio_duration * sampling_rate
fma_dir = "../data/fma_medium/"
npy_folder_path = "../data/npy"
n_mels = 128

def load_audio(path):
    """
    :param path: The path to an audio file
    :return y: The time series of a loaded audio file of size [input_size]
    """
    y, _ = librosa.load(path=path, sr=sampling_rate)

    if (len(y) > input_size):
        y = y[:input_size]

    elif (len(y) < input_size):
        y = np.pad(y, (0, input_size - len(y)), "constant")

    return y


def get_mel_spectrogram(y):
    """
    :param y: Time series extracted from an audio file
    :return spectro: A spectrogram of shape [n_mels, time]
    """
    spectro = librosa.feature.melspectrogram(
        y=y, sr=sampling_rate, n_mels=n_mels)
    spectro = librosa.power_to_db(spectro, ref=np.max)

    return spectro


def show_mel_spectrogram(spectro):
    """
    :param spectro: Spectrogram to display
    """
    fig, ax = plt.subplots()
    img = librosa.display.specshow(
        spectro, x_axis='time', y_axis='mel', sr=sampling_rate, fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    plt.show()


def walk_files():
    """
    Walk the files of a directory, computing its mel spectrograms and storing in an array.
    :return id_to_spectro: Dictionary of track id to array of size [num_inputs, n_mels, time]
    """
    if (len(sys.argv) != 2):
        print("Missing FMA folder number argument")
        exit()

    mfa_folder_path = fma_dir + sys.argv[1]
    if os.path.exists(mfa_folder_path) == False:
        print(mfa_folder_path + " is not a real folder")
        exit()

    print("Traversing " + mfa_folder_path)

    id_to_spectro = {}
    for subdir, _, files, in os.walk(mfa_folder_path):
        print("{} files".format(len(files)))
        for i in range(len(files)):
            if (i % 25 == 0):
                sys.stdout.write("\r{0:.0%}...".format(i/len(files)))
                sys.stdout.flush()
            filename = files[i]
            if filename.endswith(".mp3"):
                path = subdir + os.sep + filename
                id = int(filename.replace(".mp3", ''))
                try:
                    y = load_audio(path)
                    spectro = get_mel_spectrogram(y)
                    id_to_spectro[id] = spectro
                except Exception:
                    continue

    write_to_npy(id_to_spectro)


def write_to_npy(data):
    path = npy_folder_path + os.sep + sys.argv[1] + '.npy'
    with open(path, 'w+b') as f:
        np.save(f, data)
    print("\nWrote data to " + path)


def read_from_npy():
    combined_dict = {}
    for filename in os.listdir(npy_folder_path):
        with open(npy_folder_path + os.sep + filename, 'rb') as f:
            d = np.load(f, allow_pickle=True).item()
            combined_dict.update(d)
    return combined_dict


def main():
    walk_files()


if __name__ == '__main__':
    main()
