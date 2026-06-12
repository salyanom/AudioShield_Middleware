import librosa
import numpy as np
import pandas as pd
import os

BENIGN_FOLDER = "data/benign"

def extract_features(file_path):

    y, sr = librosa.load(file_path, sr=None)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    return {
        "mfcc_mean": np.mean(mfcc),
        "mfcc_std": np.std(mfcc),

        "zcr":
            np.mean(
                librosa.feature.zero_crossing_rate(y)
            ),

        "spectral_centroid":
            np.mean(
                librosa.feature.spectral_centroid(
                    y=y,
                    sr=sr
                )
            ),

        "spectral_rolloff":
            np.mean(
                librosa.feature.spectral_rolloff(
                    y=y,
                    sr=sr
                )
            ),

        "rms":
            np.mean(
                librosa.feature.rms(y=y)
            )
    }


rows = []

for file in os.listdir(BENIGN_FOLDER):

    if file.endswith(".wav"):

        path = os.path.join(
            BENIGN_FOLDER,
            file
        )

        features = extract_features(path)

        features["file"] = file

        rows.append(features)

df = pd.DataFrame(rows)

os.makedirs("features", exist_ok=True)

df.to_csv(
    "features/benign_features.csv",
    index=False
)

print(df.head())
print("\nFeature extraction complete!")