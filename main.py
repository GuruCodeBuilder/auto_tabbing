import os
from tqdm import tqdm

import torch
import numpy as np
import pandas as pd
from data import labels, read_wav, cqt_func, trim_CQT, TOP_N_FREQ
from train import GuitarCNN, train

note_reference = [
    "A",
    "Asharp",
    "B",
    "Bsharp",
    "C",
    "Csharp",
    "D",
    "Dsharp",
    "E",
    "Esharp",
    "F",
    "Fsharp",
    "G",
    "Gsharp",
]


def load_cnn_model(model_path="guitar_cnn_model.pth"):
    """Load the trained CNN model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load training data to get the correct input shape
    try:
        training_data = pd.read_pickle("./pickled_data/training_data.pkl")
        # Get the shape from actual training data
        sample_cqt_data = training_data.iloc[0]["CQT_DATA_MEAN_TRIMMED"]
        dummy_tensor = torch.FloatTensor(sample_cqt_data).unsqueeze(
            0
        )  # Add channel dim
        input_shape = dummy_tensor.shape  # (1, height, width)
    except Exception as e:
        print(f"Warning: Could not load training data, using default shape: {e}")
        # Fallback to default shape
        dummy_cqt_data = np.zeros((84, 87))  # Standard CQT trimmed shape
        dummy_tensor = torch.FloatTensor(dummy_cqt_data).unsqueeze(0)  # Add channel dim
        input_shape = dummy_tensor.shape  # (1, height, width)

    num_classes = len(labels)

    # Initialize model
    model = GuitarCNN(input_shape, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, device


def predict_with_cnn(model, cqt_data, device):
    """Make a prediction using the CNN model"""
    # Convert to tensor and add channel dimension
    input_tensor = (
        torch.FloatTensor(cqt_data).unsqueeze(0).unsqueeze(0)
    )  # Add batch and channel dims
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    return predicted_class


# defining the main function
def main(
    setting: str = "MANUAL", ml_alg_name: str = "dqn_guitar", *args, **kwargs
) -> tuple[list, list, list, list, list, list, list, list] | None:
    """
    The main() function contains the functionality of the program, returning or outputing the predictions
    based off the ml algorithm model, which uses data from the data.py module, trims it with the cqt_trim.py
    module, trains the model with the train.py module, and ultimately creates a model that can be used
    to predict the notes of different audio files

    PARAMS
    ------
    - `setting`: determines what functionality of the main() func will be used
    - `args`: any extra arguments that can be used
    - `kwargs`: key word arguments — likely to be other dicts or lists from the auto_run.py module. Examples of what may be passed into kwargs:
        - `wav_files`, `notes`, etc...

    RETURNS
    -------
    - When on the "MANUAL" mode, nothing is returned. Data is printed out to the terminal as a side effect.
    - When on the "AUTO" mode, lists containing data from the ml model being run on the audio files contined
      in the ./data directory are returned

    USAGE
    -----

    >>> from main import main
    >>> # default call, which requires user input
    >>> main()
    >>> # switched mode to AUTO and passes in required data into kwargs
    >>> main("AUTO", wav_files=wav_files, notes=notes)
    """

    # Load the CNN model
    model_path = (
        ml_alg_name + ".pth" if ml_alg_name != "dqn_guitar" else "guitar_cnn_model.pth"
    )
    if not os.path.exists(model_path):
        train(ml_alg_name.replace("dqn_guitar", "guitar_cnn_model"))
    model, device = load_cnn_model(model_path)

    predicted_list = []
    matches = []
    near = []
    far = []
    note_ref_indexes = []
    predicted_ref_indexes = []
    notes_plain = []
    predictions_plain = []

    print(f"MODE: {setting}")

    # what to do if mode indicates user input (also the default)
    if setting == "MANUAL":
        file = input(
            "Enter the path to the wav file: "
        )  # user input for rel path of audio file
        while file != "":  # empty string idicates user quits the program
            if os.path.exists(file) and file.endswith(".wav") and os.path.isfile(file):
                cqt_nabs, cqt_datum = cqt_func(*read_wav(file))
                trimmed_data_mean, _, _ = trim_CQT(cqt_datum, top=TOP_N_FREQ)
                print(trimmed_data_mean.shape)
                action = predict_with_cnn(model, trimmed_data_mean, device)
                print(f"Predicted label: {labels[action]}")
                predicted_list.append(labels[action])
            file = input("Enter the relative path to the wav file: ")
        return

    # mode is on AUTO
    elif setting == "AUTO":
        wav_files = kwargs[
            "wav_files"
        ]  # extract rel paths for the wav files from the arbitrary kwargs dict
        notes = kwargs["notes"]  # extract the corresponding notes from kwargs
        validation_data: pd.DataFrame | None = kwargs["val_data"]

        if validation_data is not None:
            for index, _ in enumerate(
                tqdm(
                    validation_data["LABEL"],
                    desc="Predicting notes for validation data",
                )
            ):
                val_datum_label = validation_data.iloc[index]["LABEL"]
                val_trimmed_data_mean = validation_data.iloc[index][
                    "CQT_DATA_MEAN_TRIMMED"
                ]
                action = predict_with_cnn(model, val_trimmed_data_mean, device)
                prediction = labels[action]
                predicted_list.append(prediction)
                match_val = 1 if str(prediction) == val_datum_label else 0
                matches.append(match_val)
                note_calculations(
                    val_datum_label,
                    prediction,
                    match_val,
                    near,
                    far,
                    note_ref_indexes,
                    predicted_ref_indexes,
                    notes_plain,
                    predictions_plain,
                )

        else:
            # loop trough the audio files whilst displaying a loading bar
            for index, file in enumerate(
                tqdm(wav_files, desc="Predicting notes for wav files")
            ):
                # run the cqt function on the audio file by inputing its properties taken from the read_wav() func
                _, cqt_datum = cqt_func(*read_wav(file))
                trimmed_data_mean, _, _ = trim_CQT(
                    cqt_datum, top=TOP_N_FREQ
                )  # trim the data before calling the ml_algirhtm for a prediction
                action = predict_with_cnn(
                    model, trimmed_data_mean, device
                )  # obtain the action/prediction for the index of the label
                # obtain the predicted label and append it to the precitions list
                prediction = labels[action]
                predicted_list.append(prediction)
                # determine if it is a match to the correct label of the wav file and append the answer to the list "match"
                match_val = (
                    1 if str(prediction) == notes[index] else 0
                )  # 1 = True, 0 = False
                matches.append(match_val)
                note_calculations(
                    notes[index],
                    prediction,
                    match_val,
                    near,
                    far,
                    note_ref_indexes,
                    predicted_ref_indexes,
                    notes_plain,
                    predictions_plain,
                )

        return (  # return a tuple of all the lists that will be put into the prediction data frame
            predicted_list,
            matches,
            near,
            far,
            note_ref_indexes,
            predicted_ref_indexes,
            notes_plain,
            predictions_plain,
        )


def note_calculations(
    label,
    prediction,
    match_val,
    near,
    far,
    note_ref_indexes,
    predicted_ref_indexes,
    notes_plain,
    predictions_plain,
):
    # obtain the pitch of the original note at the predicted note (which excludes octave)
    note_original = label[:-1]
    note_predicted = prediction[:-1]
    # append these data to the "notes_plain" and "predictions_plain" lists
    notes_plain.append(note_original)
    predictions_plain.append(note_predicted)
    try:
        # get the indexes of note_original and predicted_note in reference to the list "notes"
        note_index = note_reference.index(note_original)
        predicted_index = note_reference.index(note_predicted)
        # append the indexes to the respective lists of indexes for all the audio files
        note_ref_indexes.append(note_index)
        predicted_ref_indexes.append(predicted_index)
        # create the near and far value container vars and set them to default as 0
        near_val = 0  # 0 represents False, 1 represents True
        far_val = 0  # this also allows an easy way to add up totals for near and/or far

        if (
            (  # if the predicted index is 1 unit away form the correct index, set the "near" var to 1
                (note_index - 1) % len(note_reference)
            )
            == predicted_index
            or note_index + 1 == predicted_index
        ):
            near_val = 1
        # The following accounts for the "octave problem"
        # (e.g. when ignoring octaves, Gsharp is essentially one index away from A)
        elif note_index + 1 == len(note_reference) and predicted_index == 0:
            near_val = 1

        # if near and match aren't true
        if near_val != 1 and match_val != 1:
            # if the index of the predicted note is within 3 units of the correct index
            if note_index - 3 <= predicted_index <= note_index + 3:
                far_val = 1  # set "far_val" var to True
            # The following accounts for the octave problem refered to earlier
            elif (
                note_index - 3
                <= predicted_index - len(note_reference)
                <= note_index + 3
            ):
                far_val = 1
        # append the near and far values for this audio file into the list containing the values for all the files
        near.append(near_val)
        far.append(far_val)
    except ValueError:
        # the case for if "note_original" and/or "predicted_note" are not in the notes list
        near.append(-1)
        far.append(-1)


if __name__ == "__main__":
    main()
