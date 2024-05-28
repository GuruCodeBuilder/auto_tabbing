import os
import wave
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt

labels = None
fft_data = None # data is stored in 3 dimentions: [label][wav_file][freq/mag]
dataFrame = None # pandas data frame with 3 columns: label, freq, mag

def read_wav(file):
    with wave.open(file, 'r') as f:
        audio_data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
        frame_rate = f.getframerate()
    return audio_data, frame_rate

def fft(audio_data, frame_rate):
    fft_data = np.fft.fft(audio_data)
    freq_bins = np.fft.fftfreq(len(fft_data), d=1.0/frame_rate)
    positive_freq_bins = freq_bins[:len(freq_bins)//2]
    magnitude_spectrum = np.abs(fft_data[:len(fft_data)//2])
    return positive_freq_bins, magnitude_spectrum

def plot_fft_data():
    sns.scatterplot(data=dataFrame, x='max_mag_freq', y='max_mag', hue='label', sizes=(20, 200))
    plt.ylabel('Max Magnitude')
    plt.xlabel('Frequency of Max Magnitude (Hz)')
    plt.title('Max Magnitude vs Frequency of Max Magnitude')
    plt.legend(loc='upper right', prop={'size': 8})
    plt.show()

if not os.path.exists('./pickled_data/data.pkl') or not os.path.exists('./pickled_data/labels.pkl') or not os.path.exists('./pickled_data/dataFrame.pkl'):
    if not os.path.exists('./pickled_data'):
        os.mkdir('./pickled_data')
    labels = [i.name for i in os.scandir('./data') if i.is_dir()]
    labels.sort()
    fft_data = []
    
    for label in tqdm(labels, desc='Reading wav files'):
        wav_files = [f"./data/{label}/{i.name}" for i in os.scandir(f'./data/{label}') if i.is_file() and i.name.endswith('.wav')]
        _fft_data = []
        for wav in wav_files:
            wav_data, wav_rate = read_wav(wav)
            freq, mag = fft(wav_data, wav_rate)
            _fft_data.append((freq, mag))  # Append as tuple
        fft_data.append(_fft_data)
    with open('./pickled_data/data.pkl', 'wb') as f:
        pickle.dump(fft_data, f)
    with open('./pickled_data/labels.pkl', 'wb') as f:
        pickle.dump(labels, f)

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
    dataFrame = pd.read_pickle('./pickled_data/dataFrame.pkl')
    

if __name__ == "__main__":
    print(f"{len(labels)} total labels")
    print(f"Total wav files: {sum([len(i) for i in fft_data])}")
    
    for i, label in enumerate(labels):
        print(f"{i}. {label}: {len(fft_data[i])} wav files")
    
    # graph
    plot_fft_data()