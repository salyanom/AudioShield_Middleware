#!/bin/bash
set -e

echo "=== Step 2: Environment Verification ==="
python3 -c "import sys; print('Python:', sys.version)"
python3 -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPU Visible:', tf.test.is_gpu_available())"

CHECKPOINT_PATH=$(find /deepspeech-0.9.3-checkpoint -name "best_dev-1466475" | head -n 1)
if [ -z "$CHECKPOINT_PATH" ]; then
    echo "Could not find best_dev-1466475 checkpoint!"
    ls -R /deepspeech-0.9.3-checkpoint
    exit 1
fi
echo "Checkpoint found at: $CHECKPOINT_PATH"

echo "=== Step 3: Classify sample-000000.wav ==="
python3 classify.py --input sample-000000.wav --restore_path "$CHECKPOINT_PATH"

echo "=== Step 4: Generate Adversarial Example ==="
python3 attack.py --input sample-000000.wav --target "this is a test" --restore_path "$CHECKPOINT_PATH" --outprefix adversarial_carlini

echo "=== Step 5: Verify Adversarial Example ==="
python3 classify.py --input adversarial_carlini.wav --restore_path "$CHECKPOINT_PATH"
