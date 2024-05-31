import os
# import sys
import wave
import soundfile
import pickle

import librosa as lbr
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt

labels = None
fft_data = None # data is stored in 3 dimentions: [label][wav_file][freq/mag]
dataFrame = None # pandas data frame with 3 columns: label, freq, mag

# labelsCQT = None
# cqt_data = None
# dataFrameCQT = None

def read_wav(file):
    with wave.open(file, 'r') as f:
        audio_data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
        audio_data_float, sample_rate = lbr.load(file)
        frame_rate = f.getframerate()
    return audio_data, audio_data_float, frame_rate, sample_rate

def fft(audio_data, frame_rate):
    fft_data = np.fft.fft(audio_data)
    freq_bins = np.fft.fftfreq(len(fft_data), d=1.0/frame_rate)
    positive_freq_bins = freq_bins[:len(freq_bins)//2]
    magnitude_spectrum = np.abs(fft_data[:len(fft_data)//2])
    return positive_freq_bins, magnitude_spectrum

def cqt_func(audio_data, frame_rate):
    cqt_data = np.abs(
        lbr.cqt(
            audio_data,
            sr=frame_rate,
            bins_per_octave=48,
            fmin=lbr.note_to_hz("A2"),
            n_bins=288
        )
    )  #  n_bins=480, bins_per_octave=96
    return cqt_data

def plot_fft_data():
    sns.scatterplot(data=dataFrame, x='max_mag_freq', y='max_mag', hue='label', sizes=(20, 200))
    plt.ylabel('Max Magnitude')
    plt.xlabel('Frequency of Max Magnitude (Hz)')
    plt.title('Max Magnitude vs Frequency of Max Magnitude')
    plt.legend(loc='upper right', prop={'size': 8})
    # plt.show()

def plot_cqt_data(cqtd, sample_rate, label, ind):
    fig, ax = plt.subplots()
    img = lbr.display.specshow(
        lbr.amplitude_to_db(cqtd, ref=np.max), 
        y_axis='cqt_note', 
        sr=sample_rate, 
        x_axis='ms', 
        ax=ax, 
        bins_per_octave=48,
        fmin=lbr.note_to_hz("A2"),
    )
    ax.set_title(f'Constant-Q power spectrum + pitch of {label}')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plot_file_name = f"./cqt_graphs/{label}/{label}-v{ind}.jpg"
    if not os.path.exists(plot_file_name):
        plt.savefig(plot_file_name)

if not os.path.exists("./cqt_graphs"):
    os.mkdir("./cqt_graphs")

if not os.path.exists('./pickled_data/data.pkl') or not os.path.exists('./pickled_data/labels.pkl') or not os.path.exists('./pickled_data/dataFrame.pkl'):
    if not os.path.exists('./pickled_data'):
        os.mkdir('./pickled_data')
    labels = [i.name for i in os.scandir('./data') if i.is_dir()]
    labels.sort()
    fft_data = []
    cqt_data = []

    for label in tqdm(labels, desc='Reading wav files'):
        wav_files = [f"./data/{label}/{i.name}" for i in os.scandir(f'./data/{label}') if i.is_file() and i.name.endswith('.wav')]
        _fft_data = []
        _cqt_data = []
        label_fp = f"./cqt_graphs/{label}" # fp - file path
        if not os.path.exists(label_fp):
            os.mkdir(label_fp)
        for ind, wav in enumerate(wav_files):
            wav_data, wdf, wr, sr = read_wav(wav)
            sample_rate_ref = wr
            freq, mag = fft(wav_data, wr)
            cqt_datum = cqt_func(wdf, sr)
            plot_cqt_data(cqt_datum, sr, label, ind + 1)
            _fft_data.append((freq, mag))  # Append as tuple
            _cqt_data.append(cqt_datum)
        fft_data.append(_fft_data)
        cqt_data.append(_cqt_data)
    with open('./pickled_data/data.pkl', 'wb') as f:
        pickle.dump(fft_data, f)
    with open('./pickled_data/labels.pkl', 'wb') as f:
        pickle.dump(labels, f)
    with open("./pickled_data/cqt_data.pkl", "wb") as f:
        pickle.dump(cqt_data, f)

    df_data = []
    for i, label in tqdm(enumerate(labels), desc='adding data to dataFrame'):
        for data in fft_data[i]:
            df_data.append((
                label,
                len(data[0]),
                np.mean(data[1]),
                np.max(data[1]),
                data[0][np.argmax(data[1])]
            ))

    dataFrame = pd.DataFrame(df_data, columns=['label', 'freq', 'mag', 'max_mag', 'max_mag_freq'], index=None)

    dataFrame.to_pickle('./pickled_data/dataFrame.pkl')

else:
    with open('./pickled_data/data.pkl', 'rb') as f:
        fft_data = pickle.load(f)
    with open('./pickled_data/labels.pkl', 'rb') as f:
        labels = pickle.load(f)
    with open('./pickled_data/cqt_data.pkl', 'rb') as f:
        cqt_data = pickle.load(f)
    dataFrame = pd.read_pickle('./pickled_data/dataFrame.pkl')


if __name__ == "__main__":
    print(f"{len(labels)} total labels")
    print(f"Total wav files: {sum([len(i) for i in fft_data])}")
    
    for i, label in enumerate(labels):
        print(f"{i + 1}. {label}: {len(fft_data[i])} wav files")
    
    # graph
    plot_fft_data()
    # plot_cqt_data(cqt_data)
