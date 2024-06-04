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
fft_data = None  # data is stored in 3 dimentions: [label][wav_file][freq/mag]
dataFrame = None  # pandas data frame with 3 columns: label, freq, mag


def read_wav(file):
    """Read data from the audio files individually, obtaining information on the dat itself and frame/sample rate"""
    with wave.open(file, "r") as f:
        audio_data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
        audio_data_float, sample_rate = lbr.load(file)
        frame_rate = f.getframerate()
    return audio_data, audio_data_float, frame_rate, sample_rate


def fft(audio_data, frame_rate):
    """
    Simulates the FFT (Fast Fourier Transform) and obtains the positive
    frequencies of the returned transform and its magnitude spectrum
    """
    fft_data = np.fft.fft(audio_data)
    freq_bins = np.fft.fftfreq(len(fft_data), d=1.0 / frame_rate)
    positive_freq_bins = freq_bins[: len(freq_bins) // 2]
    magnitude_spectrum = np.abs(fft_data[: len(fft_data) // 2])
    return positive_freq_bins, magnitude_spectrum


def cqt_func(audio_data, frame_rate):
    """
    Simulates the CQT (Constant Q Transformation) and returns the data,
    which consists of varying frequencies (in Hz) and their time durations
    """
    cqt_data = np.abs(
        lbr.cqt(
            audio_data,
            sr=frame_rate,
            bins_per_octave=48,  # bins_per_octave and n_bins both control resolution of image.
            fmin=lbr.note_to_hz("A2"),  # sets the lower bound of collected notes
            n_bins=288,  # for a whole number of octaves to be displayed, must be a multiple of bins_per_octave
        )
    )  #  n_bins=480, bins_per_octave=96
    return cqt_data


def plot_fft_data():
    """
    Creates a plot made from the data of all the FFTs obtained from the audio
    files that consists of maximum magnitudes vs its respective frequency
    """
    sns.scatterplot(
        data=dataFrame, x="max_mag_freq", y="max_mag", hue="label", sizes=(20, 200)
    )
    plt.ylabel("Max Magnitude")
    plt.xlabel("Frequency of Max Magnitude (Hz)")
    plt.title("Max Magnitude vs Frequency of Max Magnitude")
    plt.legend(loc="upper right", prop={"size": 8})
    # plt.show()


def plot_cqt_data(cqtd, sample_rate, label, ind):
    """
    Creates a CQT display and uploads the image into a ./cqt_graphs/ folder. It displays the
    time vs frequencies (frequencies can be shown in Hz or letter pitch, e.g. 110 Hz or A2).
    Each audio file has its respective CQT plot respectively in the CQT graphs folder. Purely for visulization;
    Visulizations are meant to be viewed in a developer env.
    """
    fig, ax = plt.subplots()
    img = lbr.display.specshow(
        lbr.amplitude_to_db(cqtd, ref=np.max),
        y_axis="cqt_note",  # change this param to switch between showing frequncies in Hz or its pitch
        sr=sample_rate,
        x_axis="ms",
        ax=ax,
        bins_per_octave=48,  # controls the scaling to the resolution of images.
        fmin=lbr.note_to_hz(
            "A2"
        ),  # tries to set the lower bound of frequencies displayed on the y axis
    )
    ax.set_title(f"Constant-Q power spectrum + pitch of {label}")
    fig.colorbar(
        img, ax=ax, format="%+2.0f dB"
    )  # creates a color bar to map the CQT diagrams with relative strengths.
    plot_file_name = f"./cqt_graphs/{label}/{label}-v{ind}.jpg"
    # save files into a CQT diagram folder
    if not os.path.exists(plot_file_name):
        plt.savefig(plot_file_name)


# create the CQT graphs folder, used y the plot_cqt_data function
if not os.path.exists("./cqt_graphs"):
    os.mkdir("./cqt_graphs")

# checks if the pickled data file has been created or not and considers the case ig not created
if (
    not os.path.exists("./pickled_data/data.pkl")
    or not os.path.exists("./pickled_data/labels.pkl")
    or not os.path.exists("./pickled_data/dataFrame.pkl")
):
    if not os.path.exists("./pickled_data"):
        os.mkdir("./pickled_data")  # If the dir does not exist, create it
    labels = [
        i.name for i in os.scandir("./data") if i.is_dir()
    ]  # obtain the note labels from the data folder, 42-43 of them
    labels.sort()
    # Instantiating lists to store the data from the respective transformations
    fft_data = []
    cqt_data = []

    # starts reading the .wav files form the data dir
    for label in tqdm(
        labels, desc="Reading wav files"
    ):  # moves thru the 42 labels while producing a viewable progress bar
        # Obtain a list of strings of the .wav file names and their relative paths under the folder named with the note described by "label"
        wav_files = [
            f"./data/{label}/{i.name}"
            for i in os.scandir(f"./data/{label}")
            if i.is_file() and i.name.endswith(".wav")
        ]
        # Instantiates lists to group together FFT/CQT data from audio files under the same note
        _fft_data = []
        _cqt_data = []
        # creates the file path directory for the respective note described by "label" in the cqt graphs dir
        label_fp = f"./cqt_graphs/{label}"  # fp - file path
        if not os.path.exists(label_fp):
            os.mkdir(label_fp)  # creates the dir if it does not already exist
        # Loop thru wav files as tuples of (index, file)
        for ind, wav in enumerate(wav_files):
            wav_data, wdf, wr, sr = read_wav(
                wav
            )  # wav data, wav data in float, wave rate, sample rate respectively
            # using data obtained from reading wav file, run fft and cqt and obtain their corresponding data
            freq, mag = fft(wav_data, wr)
            cqt_datum = cqt_func(wdf, sr)
            plot_cqt_data(
                cqt_datum, sr, label, ind + 1
            )  # make plots of cqt data for each wav file
            _fft_data.append(
                (freq, mag)
            )  # Append fft freq and mag data as tuple onto list represnting fft data for "label" wav files
            _cqt_data.append(
                cqt_datum
            )  # same logic as above but with cqt data for wav files under the note descfibed by "label"
        # append data for fft and cqt for the label to the overall lists representing all the fft and cqt data repectively
        fft_data.append(_fft_data)
        cqt_data.append(_cqt_data)
    # upload data locally into pickled binary files to be loaded in for later use
    with open("./pickled_data/data.pkl", "wb") as f:
        pickle.dump(fft_data, f)
    with open("./pickled_data/labels.pkl", "wb") as f:
        pickle.dump(labels, f)
    with open("./pickled_data/cqt_data.pkl", "wb") as f:
        pickle.dump(cqt_data, f)

    df_data = []  # list to represent the dataFrame
    # append data into the list from the overall fft list in the format (label, freq, mag, max_mag, max_mag_freq)
    for i, label in tqdm(
        enumerate(labels), desc="adding data to dataFrame"
    ):  # move through all the labels
        for data in fft_data[
            i
        ]:  # move through all the files of each not represented by the label and its index
            df_data.append(
                (
                    label,  # the note/label
                    len(data[0]),  # feq
                    np.mean(data[1]),  # mean magnitude
                    np.max(data[1]),  # max magnitude
                    data[0][np.argmax(data[1])],  # max_mag_freq
                )
            )

    # use pandas to make the data frame from the df_data list, with the columns lining up respective to what was previously shown
    dataFrame = pd.DataFrame(
        df_data, columns=["label", "freq", "mag", "max_mag", "max_mag_freq"], index=None
    )
    # pickle the dataFrame into a pickle file with a pd method
    dataFrame.to_pickle("./pickled_data/dataFrame.pkl")

# if they already exist, then the program has already been ran or the data has been gathered in some different way
else:
    # opens all the pickled files and loads in information with pickle and pd library methods
    with open("./pickled_data/data.pkl", "rb") as f:
        fft_data = pickle.load(f)
    with open("./pickled_data/labels.pkl", "rb") as f:
        labels = pickle.load(f)
    with open("./pickled_data/cqt_data.pkl", "rb") as f:
        cqt_data = pickle.load(f)
    dataFrame = pd.read_pickle("./pickled_data/dataFrame.pkl")

if __name__ == "__main__":
    # prints out infor regarding the number of labels, number of files per note/label, total number of wav files, and more
    print(f"{len(labels)} total labels")
    print(f"Total wav files: {sum([len(i) for i in fft_data])}")
    # the info regarding number of wav files per label is printed here
    for i, label in enumerate(labels):
        print(f"{i + 1}. {label}: {len(fft_data[i])} wav files")

    # graph the summary fft data with the function here
    plot_fft_data()
