import os
import wave

import librosa as lbr
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from cqt_trim import trim_CQT

# constants for quick config
GRAPH = False  # set to True to generate the CQT graphs
TOP_N_FREQ = 5  # number of top frequencies to be returned by the trim_CQT function
TRIMMED_EQUIVALENCE = True
SWITCH = False

SAMPLE_RATE_REF = 12000

labels = None
cqt_data = (
    pd.DataFrame()
)  # data frame to store the cqt data. LABEL, CQT_DATA_TRIMMED_MEAN, CQT_DATA_FULL
cqt_data_complex = (
    pd.DataFrame()
)  # data frame to store cqt data as complex form. LABEL, CQT_DATA_COMPLEX
cqt_data_sum_sorted = (
    pd.DataFrame()
)  # data frame to store cqt data with trimmed data sorted by sum. LABEL, CQT_DATA_TRIMMED_SUM, CQT_DATA_FULL


def read_wav(file):
    """Read data from the audio files individually, obtaining information on the dat itself and frame/sample rate"""
    with wave.open(file, "r") as f:
        audio_data_float, sample_rate = lbr.load(file)
    return audio_data_float, sample_rate


def cqt_func(audio_data, frame_rate):
    """
    Simulates the CQT (Constant Q Transformation) and returns the data,
    which consists of varying frequencies (in Hz) and their time durations
    """
    cqt_data_no_abs = lbr.cqt(  # This represents the frequencies themselves purely in complex form
        audio_data,
        sr=frame_rate,
        bins_per_octave=48,  # bins_per_octave and n_bins both control resolution of image.
        fmin=lbr.note_to_hz("A2"),  # sets the lower bound of collected notes
        n_bins=288,  # for a whole number of octaves to be displayed, must be a multiple of bins_per_octave
    )  #  n_bins=480, bins_per_octave=96
    cqt_data = np.abs(
        cqt_data_no_abs
    )  # take the absolute value of all the frequenceies, taking the magnitudes of the complex numbers
    return cqt_data_no_abs, cqt_data


def plot_cqt_data(cqtd, sample_rate, label, ind):
    """
    Creates a CQT display and uploads the image into a ./cqt_graphs/ folder. It displays the
    time vs frequencies (frequencies can be shown in Hz or letter pitch, e.g. 110 Hz or A2).
    Each audio file has its respective CQT plot respectively in the CQT graphs folder. Purely for visulization;
    Visulizations are meant to be viewed in a developer env.
    """
    if GRAPH:
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


def split_training_valid(
    data: pd.DataFrame,
    training_size: float | None | int = None,
    verbose: bool = False,
):
    """Splits the data into training and validation data.

    Args:
        data (pd.DataFrame): The data to split.
        training_size (float | None | int, optional): If this number is an int, we take that much training data for each label. The rest goes to validation. If this is a float, then allocate that percent of each label for training data and rest for validation. If the value is None, then use the smallest label value as training_size. rest goes to validation. Defaults to None.
        verbose (bool, optional): If True, print out useful stuff. Defaults to False.
    Returns:
        pd.DataFrame, pd.DataFrame: The training and validation data respectively.
    """
    training_data_list = []
    validation_data_list = []

    def _print(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    if training_size is None:
        training_size = data["LABEL"].value_counts().min()  # smallest label value
        _print(
            f"Training size not specified. Using smallest label value: {training_size}"
        )
    elif isinstance(training_size, float):
        _print(f"Training size is a float: {training_size}")
        pass
    else:
        _print(f"Training size is an int: {training_size}")

    # Split the data into training and validation data based on the training_size
    for label in tqdm(data["LABEL"].unique(), desc="Splitting data"):
        label_data = data[data["LABEL"] == label]
        if isinstance(training_size, float):
            training_data_list.append(label_data.sample(frac=training_size))
        else:
            training_data_list.append(label_data[:training_size])
        validation_data_list.append(label_data[training_size:])

    training_data = pd.concat(training_data_list)
    validation_data = pd.concat(validation_data_list)

    return training_data, validation_data


# create the CQT graphs folder, used y the plot_cqt_data function
if not os.path.exists("./cqt_graphs"):
    os.mkdir("./cqt_graphs")

# checks if the pickled data file has been created or not and considers the case it not created
if not os.path.exists("./pickled_data/data.pkl"):
    if not os.path.exists("./pickled_data"):
        os.mkdir("./pickled_data")  # If the dir does not exist, create it
    labels = [
        i.name for i in os.scandir("./data") if i.is_dir()
    ]  # obtain the note labels from the data folder, 42-43 of them
    labels.sort()

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
        # Instantiates lists to group together CQT data from audio files under the same note
        _cqt_data_labels = []
        _cqt_data_trimmed_mean = []
        _cqt_data_trimmed_sum = []
        _cqt_data_full = []
        _cqt_nabs_list = []
        # creates the file path directory for the respective note described by "label" in the cqt graphs dir
        label_fp = f"./cqt_graphs/{label}"  # fp - file path
        if not os.path.exists(label_fp):
            os.mkdir(label_fp)  # creates the dir if it does not already exist
        # Loop thru wav files as tuples of (index, file)
        for ind, wav in enumerate(wav_files):
            wdf, sr = read_wav(
                wav
            )  # wav data, wav data in float, wave rate, sample rate respectively
            # using data obtained from reading wav file, run cqt and obtain their corresponding data
            SAMPLE_RATE_REF = sr
            cqt_nabs, cqt_datum = cqt_func(
                wdf, sr
            )  # cqt_nabs represents the data of cqt_datum w/o absolute value method being applied
            plot_cqt_data(
                cqt_datum, sr, label, ind + 1
            )  # plot the cqt data for the wav file under the note described by "label"

            trimmed_data_mean, trimmed_data_sum, TRIMMED_EQUIVALENCE = trim_CQT(
                cqt_datum, top=TOP_N_FREQ
            )  # trim data

            if TRIMMED_EQUIVALENCE == False or SWITCH == True:
                SWITCH = True
                TRIMMED_EQUIVALENCE = False

            # update columns
            _cqt_data_full.append(cqt_datum)
            _cqt_data_trimmed_mean.append(trimmed_data_mean)
            _cqt_data_trimmed_sum.append(trimmed_data_sum)
            _cqt_data_labels.append(label)
            _cqt_nabs_list.append(cqt_nabs)
        # append data for cqt for the label to the overall lists representing all the cqt data repectively
        new_mean = pd.DataFrame(
            {
                "LABEL": _cqt_data_labels,
                "CQT_DATA_MEAN_TRIMMED": _cqt_data_trimmed_mean,
                "CQT_DATA_FULL": _cqt_data_full,
            }
        )
        new_sum = pd.DataFrame(
            {
                "LABEL": _cqt_data_labels,
                "CQT_DATA_SUM_TRIMMED": _cqt_data_trimmed_sum,
            }
        )
        new_complex = pd.DataFrame(
            {"LABEL": _cqt_data_labels, "CQT_DATA_COMPLEX": _cqt_nabs_list}
        )
        cqt_data = pd.concat([cqt_data, new_mean], ignore_index=True)
        cqt_data_sum_sorted = pd.concat(
            [cqt_data_sum_sorted, new_sum], ignore_index=True
        )
        cqt_data_complex = pd.concat([cqt_data_complex, new_complex], ignore_index=True)
    # upload data locally into pickled binary files to be loaded in for later use
    cqt_data.to_pickle("./pickled_data/data.pkl")
    cqt_data_sum_sorted.to_pickle("./pickled_data/mag_sum_data.pkl")
    # turn above specified data into csv files for readability of data
    cqt_data.to_csv("./cqt_data_frame.csv", sep="\n", index=False)
    cqt_data_complex.to_csv("./cqt_complex_data_frame.csv", sep="\n", index=False)
    cqt_data_sum_sorted.to_csv("./cqt_data_sum_frame.csv", sep="\n", index=False)
else:
    # read already created cqt data
    cqt_data = pd.read_pickle("./pickled_data/data.pkl")
    cqt_data_sum_sorted = pd.read_pickle("./pickled_data/mag_sum_data.pkl")

    for index, _ in enumerate(cqt_data):
        single_equivalency = np.array_equal(
            cqt_data.iloc[index]["CQT_DATA_MEAN_TRIMMED"],
            cqt_data_sum_sorted.iloc[index]["CQT_DATA_SUM_TRIMMED"],
        )
        if not single_equivalency:
            TRIMMED_EQUIVALENCE = False

    labels = cqt_data["LABEL"].unique()

    cqt_data.to_csv("./cqt_data_frame.csv", sep="\n", index=False)
    cqt_data_sum_sorted.to_csv("./cqt_data_sum_frame.csv", sep="\n", index=False)

if __name__ == "__main__":
    # print the cqt head
    print(cqt_data.head())
    print(f"Trimmed data is equivalent: {TRIMMED_EQUIVALENCE}")
