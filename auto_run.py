from main import main

import pandas as pd
import os

wav_files = []  # list of relative paths to wav files
notes = []  # concurrent list of notes that corresponds to the previous lsit made

# fill the lists with data from the wav files in the ./data folder
for i in os.scandir("./data"):
    if i.is_dir():
        note = i.name
        for i in os.scandir(f"./data/{note}"):
            if i.is_file() and i.name.endswith(".wav"):
                wav_files.append(f"./data/{note}/{i.name}")
                notes.append(f"{note}")

# sort the lists by primarily by name then secondarily by number of note (representing the pitch and octave respectively)
wav_files = sorted(wav_files)
notes = sorted(notes)


def auto_gen():
    """Repeatedly calls the main() function from the main module on validation files to test the ml algorithm's accuracy"""
    # case representing if ml prediction pickled data has not been created or does not exist
    if not os.path.exists("./pickled_data/ml_prediction_df.pkl"):
        # unpack a tuple return from the main() function from the `main` module
        (
            predictions,
            matches,
            near,
            far,
            note_ref_indexes,
            predicted_ref_indexes,
            note_plain,
            predicted_plain,
        ) = main(
            setting="AUTO", wav_files=wav_files, notes=notes
        )  # pass in the setting name, list of relative wav file paths, and corresponding notes

        # make a data frame consisting of all important data
        prediction_frame = pd.DataFrame(
            {
                "ORIGINAL_NOTE": notes,
                "PREDICTED_NOTE": predictions,
                "MATCH": matches,
                "NEAR": near,
                "FAR": far,
                "PLAIN": note_plain,
                "PREDICTED_PLAIN": predicted_plain,
                "PLAIN_REF_INDEXES": note_ref_indexes,
                "PREDICTED_PLAIN_REF_INDEXES": predicted_ref_indexes,
            }
        )  # ORIGINAL NOTE, PREDICTED NOTE, MATCH, NEAR, FAR, PLAIN, PREDICTED_PLAIN, PLAIN_REF_INDEXES, PREDICTED_PLAIN_REF_INDEXES
        # note: a "plain" note/pitch refers to the note excluding octave number.
        # note: ref indexes refers to the list of notes + their respective indexes in main.py
        # TODO: figure out a way to manage data structures and vartiables, like using a seperate module/file containing them

        # pickle the data frame for the predictions made by the current instance of the ml model
        prediction_frame.to_pickle("./pickled_data/ml_prediction_df.pkl")
        # extract the prediction data frame and input it into a csv file labeled "model_output.csv" in the "./csv_files" folder
        prediction_frame.to_csv("./csv_files/model_output.csv", sep="\t", index=False)

    else:
        # the prediction data frame has already been created, so load the data
        prediction_frame = pd.read_pickle(
            "./pickled_data/ml_prediction_df.pkl"
        )  # load the pickled data
        # extract different columns fro mthe data frame
        matches = prediction_frame["MATCH"]
        near = prediction_frame["NEAR"]
        far = prediction_frame["FAR"]

    # compile the data and print them out in a concise manner + summarizing results
    print(f"Total number of matches: {sum(matches)}")
    print(f"Total number of near predicitons: {sum(near)}")
    print(f"Total number of far predicitons (not ideal): {sum(far)}")
    print(
        f"Total number of misses (note exact, near, or far results): {len(notes) - sum(far) - sum(near) - sum(matches)}"
    )
    print(f"Total number of valid results (matches + near): {sum(near) + sum(matches)}")
    print(
        f"Total number of exact or close results (matches + near + far): {sum(near) + sum(matches) + sum(far)}"
    )
    print(f"Success (Match) Rate: {int((sum(matches) / len(notes)) * 10000) * 0.01}%")

    # TODO: compile statistical variables for the data (i.e. mean, standard dev, etc)


if __name__ == "__main__":
    auto_gen()
