# Carlini & Wagner Audio Attack Validation Report

## Experiment

Input WAV: `sample-000000.wav`

Target phrase: `this is a test`

Iterations: `1000`

Output WAV: `adv.wav`

Checkpoint: `/deepspeech-0.9.3-checkpoint/best_dev-1466475`

Scorer: `/deepspeech-0.9.3-models.scorer`

Alphabet: `/DeepSpeech/data/alphabet.txt`

## Execution Status

Process completed normally: Yes

Docker container final status: `Exited (0)`

Attack iterations completed: `1000 / 1000`

Runtime reported by `attack.py`: `23273.187000513077` seconds

Runtime recorded by wrapper final status: `23281` seconds

Output WAV generated: Yes

Generated WAV: `adv.wav`

CSV generated: Yes

Generated CSV: `tmp/attack-en.csv`

## Transcription Verification

Original transcript:

`without the data the article useless`

Target phrase:

`this is a test`

Adversarial transcript:

`this is a test`

## Perturbation Metrics

From `tmp/attack-en.csv`:

| Metric | Value |
|---|---:|
| Audio length | `51072` samples |
| Source dB | `89.62971687316895` |
| Noise loudness | `-54.62031364440918` |
| High perturbation bound | `56.2950439453125` |
| Low perturbation bound | `0.001220703125` |
| First target hit | iteration `20` |
| Last/best recorded target hit | iteration `230` |

## Warnings / Exceptions

Warnings observed:

- `pydub was not loaded, MP3 compression will not work`
- `mesg: ttyname failed: Inappropriate ioctl for device`

No Python exception was observed in the completed attack run.

Note: `tmp/attack_timing.txt` contains an earlier stale/interrupted wrapper status entry, `EXIT_STATUS=127`, from a previous failed launch attempt. The completed validation run appended the final successful status:

```text
EXIT_STATUS=0
RUNTIME_SECONDS=23281
```

The Docker container also reported `Exited (0)`.

## Final Assessment

SUCCESS:

The adversarial transcript equals the target phrase.

The restored Carlini & Wagner targeted audio attack pipeline successfully generated `adv.wav`, and `classify.py` decoded the generated file as:

`this is a test`

