# Zhenghuatan adversarial-audio samples

Source: <https://github.com/zhenghuatan/Audio-adversarial-examples>

Paper: Saeid Samizade, Zheng-Hua Tan, Chao Shen, and Xiaohong Guan,
"Adversarial Example Detection by Classification for Deep Speech Recognition,"
ICASSP 2020.

These are the seven WAV samples published directly in the repository:

- `000621_original.wav` and `000621_adv-medium2medium.wav`
- `000639_original.wav` and `000639_adv-medium2medium.wav`
- `yes_original.wav`, `yes2right-black.wav`, and `yes2right-white.wav`

The first two pairs are white-box examples based on Mozilla Common Voice. The
`yes2right` files are targeted black-box and white-box examples based on Google
Speech Commands. The upstream README states that the source and adversarial
datasets are available under CC BY 4.0. The GitHub repository itself does not
contain a machine-readable license file, so retain this attribution and verify
licensing before redistributing a larger downloaded dataset.

These attacks target deep-speech-recognition systems, not Whisper. They are
valid published adversarial examples, but transfer to Whisper must be measured
rather than assumed. Run:

```powershell
python src/evaluate_external_adversarial.py
```
