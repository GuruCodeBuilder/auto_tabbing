import os
from tqdm import tqdm

from stable_baselines3 import DQN

from data import labels, read_wav, cqt_func, trim_CQT, TOP_N_FREQ
from train import GuitarEnv, train

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


def main(setting: str = "MANUAL", *args, **kwargs):
    env = GuitarEnv()
    if not os.path.exists("dqn_guitar.zip"):
        train("dqn_guitar")
    model = DQN.load("dqn_guitar")

    predicted_list = []
    matches = []
    near = []
    far = []
    note_ref_indexes = []
    predicted_ref_indexes = []
    notes_plain = []
    predictions_plain = []

    print(f"MODE: {setting}")

    if setting == "MANUAL":
        file = input("Enter the path to the wav file: ")
        while file != "":
            if os.path.exists(file) and file.endswith(".wav") and os.path.isfile(file):
                cqt_nabs, cqt_datum = cqt_func(*read_wav(file))
                trimmed_data_mean, trimmed_data_sum, TRIMMED_EQUIVALENCE = trim_CQT(
                    cqt_datum, top=TOP_N_FREQ
                )
                print(trimmed_data_mean.shape)
                action, _ = model.predict(trimmed_data_mean)
                print(f"Predicted label: {labels[action]}")
                predicted_list.append(labels[action])
            file = input("Enter the relative path to the wav file: ")
        return

    else:
        wav_files = kwargs["wav_files"]
        notes = kwargs["notes"]
        for index, file in enumerate(
            tqdm(wav_files, desc="Predicting notes for wav files")
        ):
            cqt_nabs, cqt_datum = cqt_func(*read_wav(file))
            trimmed_data_mean, trimmed_data_sum, TRIMMED_EQUIVALENCE = trim_CQT(
                cqt_datum, top=TOP_N_FREQ
            )
            action, _ = model.predict(trimmed_data_mean)
            prediction = labels[action]
            predicted_list.append(prediction)
            match_val = 1 if str(prediction) == notes[index] else 0
            matches.append(match_val)
            note_original = notes[index][:-1]
            note_predicted = prediction[:-1]
            notes_plain.append(note_original)
            predictions_plain.append(note_predicted)
            try:
                note_index = note_reference.index(note_original)
                predicted_index = note_reference.index(note_predicted)
                note_ref_indexes.append(note_index)
                predicted_ref_indexes.append(predicted_index)
                near_val = 0
                far_val = 0
                # TODO: MAKE THIS `NEAR` STUFF WORK
                if (
                    (note_index - 1) % len(note_reference)
                ) == predicted_index or note_index + 1 == predicted_index:
                    near_val = 1
                elif note_index + 1 == len(note_reference) and predicted_index == 0:
                    near_val = 1
                if near_val != 1 and match_val != 1:
                    if note_index - 3 <= predicted_index <= note_index + 3:
                        far_val = 1
                    elif (
                        note_index - 3
                        <= predicted_index - len(note_reference)
                        <= note_index + 3
                    ):
                        far_val = 1
                near.append(near_val)
                far.append(far_val)
            except ValueError:
                near.append(-1)
        return (
            predicted_list,
            matches,
            near,
            far,
            note_ref_indexes,
            predicted_ref_indexes,
            notes_plain,
            predictions_plain,
        )


if __name__ == "__main__":
    main()
