#!/bin/bash
set -o pipefail

mkdir -p tmp
rm -f adv.wav tmp/adv.wav tmp/attack-*.csv tmp/attack_stdout.log tmp/attack_timing.txt
rm -f /tmp/attack-*.csv /tmp/adv.wav

start=$(date +%s)
echo "START_EPOCH=$start" | tee tmp/attack_timing.txt

cd /

python3 -u /host/attack.py \
  --input /host/sample-000000.wav \
  --target "this is a test" \
  --iterations 1000 \
  --output /host/adv.wav \
  --restore_path /deepspeech-0.9.3-checkpoint/best_dev-1466475 \
  --scorer_path /deepspeech-0.9.3-models.scorer \
  --alphabet_config_path /DeepSpeech/data/alphabet.txt \
  2>&1 | tee /host/tmp/attack_stdout.log

status=${PIPESTATUS[0]}
cp /tmp/attack-*.csv /host/tmp/ 2>/dev/null || true
cp /tmp/adv.wav /host/tmp/adv.wav 2>/dev/null || true
end=$(date +%s)
echo "END_EPOCH=$end" | tee -a /host/tmp/attack_timing.txt
echo "EXIT_STATUS=$status" | tee -a /host/tmp/attack_timing.txt
echo "RUNTIME_SECONDS=$((end-start))" | tee -a /host/tmp/attack_timing.txt
exit "$status"
