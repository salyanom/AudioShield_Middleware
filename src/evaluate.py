import os
import pandas as pd

from audio_processor import transcribe_audio
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load embedding model
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# Reference benign audio
benign_audio = "data/benign/test.mp3"

print("Transcribing benign audio...")

benign_text = transcribe_audio(
    benign_audio
)

benign_embedding = model.encode(
    [benign_text]
)

results = []

for file in os.listdir(
    "data/adversarial"
):

    if not (
        file.endswith(".mp3")
        or file.endswith(".wav")
    ):
        continue

    path = os.path.join(
        "data/adversarial",
        file
    )

    print(f"Processing {file}")

    adv_text = transcribe_audio(
        path
    )

    adv_embedding = model.encode(
        [adv_text]
    )

    similarity = cosine_similarity(
        benign_embedding,
        adv_embedding
    )[0][0]

    results.append(
        {
            "file": file,
            "similarity": float(similarity),
            "transcript_length": len(adv_text),
            "transcript": adv_text
        }
    )

# Convert to DataFrame
df = pd.DataFrame(results)

# Save detailed results
df.to_csv(
    "evaluation_results.csv",
    index=False
)

print("\n===== RESULTS =====")
print(df[["file", "similarity", "transcript_length"]])

# Summary statistics
avg_similarity = df["similarity"].mean()
min_similarity = df["similarity"].min()
max_similarity = df["similarity"].max()

print("\n===== SUMMARY =====")
print(f"Average Similarity : {avg_similarity:.4f}")
print(f"Minimum Similarity : {min_similarity:.4f}")
print(f"Maximum Similarity : {max_similarity:.4f}")

print("\nSaved to evaluation_results.csv")