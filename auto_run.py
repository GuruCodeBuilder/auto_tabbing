from main import main

import pandas as pd
import os

wav_files = []
notes = []

for i in os.scandir("./data"):
    if i.is_dir():
        note = i.name
        for i in os.scandir(f"./data/{note}"):
            if i.is_file() and i.name.endswith(".wav"):
                wav_files.append(f"./data/{note}/{i.name}")
                notes.append(f"{note}")

wav_files = sorted(wav_files)
notes = sorted(notes)

if not os.path.exists("./pickled_data/ml_prediction_df.pkl"):
    (
        predictions,
        matches,
        near,
        far,
        note_ref_indexes,
        predicted_ref_indexes,
        note_plain,
        predicted_plain,
    ) = main(setting="AUTO", wav_files=wav_files, notes=notes)

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
    )  # ORIGINAL NOTE, PREDICTED NOTE, SHAPE, REL_FILE_PATH, MATCH, NEAR

    prediction_frame.to_pickle("./pickled_data/ml_prediction_df.pkl")
    prediction_frame.to_csv("./csv_files/model_output.csv", sep="\t", index=False)

else:
    prediction_frame = pd.read_pickle("./pickled_data/ml_prediction_df.pkl")
    matches = prediction_frame["MATCH"]
    near = prediction_frame["NEAR"]
    far = prediction_frame["FAR"]

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
