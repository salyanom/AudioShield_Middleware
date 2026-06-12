from pydub import AudioSegment

audio = AudioSegment.from_file(
    "data/benign/test.mp3"
)

# Speed perturbation
faster = audio.speedup(
    playback_speed=1.1
)

faster.export(
    "data/adversarial/test_speed.mp3",
    format="mp3"
)

# Louder version
louder = audio + 6

louder.export(
    "data/adversarial/test_louder.mp3",
    format="mp3"
)

print("Generated adversarial samples")