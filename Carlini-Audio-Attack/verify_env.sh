#!/bin/bash
set -e

echo "=== Step 2: Environment Verification ==="
echo "--- Python ---"
python3 -c "import sys; print('Python:', sys.version)"

echo "--- TensorFlow ---"
python3 -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"

echo "--- NumPy ---"
python3 -c "import numpy as np; print('NumPy:', np.__version__)"

echo "--- SciPy ---"
python3 -c "import scipy; print('SciPy:', scipy.__version__)"

echo "--- DeepSpeech Training ---"
python3 -c "from deepspeech_training.util.config import Config; print('DeepSpeech Config loaded OK')"

echo "--- ds_ctcdecoder ---"
python3 -c "from ds_ctcdecoder import ctc_beam_search_decoder; print('CTC decoder imported OK')"

echo "--- tf_logits ---"
cd /
python3 -c "from tf_logits import get_logits; print('tf_logits imported OK')"

echo "--- Checkpoint ---"
ls -la /deepspeech-0.9.3-checkpoint/
echo "Checkpoint path: /deepspeech-0.9.3-checkpoint/best_dev-1466475"

echo "--- Scorer ---"
ls -la /deepspeech-0.9.3-models.scorer

echo "--- GPU Status ---"
python3 -c "import tensorflow as tf; print('GPU available:', tf.test.is_gpu_available())"
nvidia-smi 2>/dev/null | head -5 || echo "nvidia-smi not available or no GPU detected by driver"

echo ""
echo "=== Environment Verification Complete ==="
echo "NOTE: RTX 4070 (compute capability 8.9) is NOT supported by TF 1.15 (max CC 7.5)."
echo "The attack pipeline will run on CPU. This is slower but functionally correct."
