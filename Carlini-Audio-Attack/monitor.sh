#!/bin/bash
PID=8
echo "=== PROCESS STATUS ==="
ps -p $PID -o pid,pcpu,pmem,etime,stat,comm --no-headers

echo ""
echo "=== /proc/$PID/status (key fields) ==="
grep -E '(State|VmRSS|VmSize|Threads|voluntary)' /proc/$PID/status

echo ""
echo "=== KERNEL WAIT CHANNEL ==="
cat /proc/$PID/wchan 2>/dev/null
echo ""

echo ""
echo "=== THREAD STATES ==="
for tid in $(ls /proc/$PID/task/); do
    state=$(cut -d' ' -f3 /proc/$PID/task/$tid/stat 2>/dev/null)
    echo "$state"
done | sort | uniq -c | sort -rn

echo ""
echo "=== OPEN FILES ==="
ls -la /proc/$PID/fd/ 2>/dev/null

echo ""
echo "=== RECENT I/O ==="
cat /proc/$PID/io 2>/dev/null

echo ""
echo "=== CHECK FOR OUTPUT FILE ==="
ls -la /output/adversarial_carlini_test.wav 2>/dev/null || echo "Output file does not exist yet"

echo ""
echo "=== CHECK TMP FILES ==="
ls -la /DeepSpeech/tmp/ 2>/dev/null
ls -la /tmp/ 2>/dev/null | head -10

echo ""
echo "=== PYTHON STACK TRACE (non-destructive) ==="
# Try to get a stack trace via /proc/PID/syscall
cat /proc/$PID/syscall 2>/dev/null
echo ""

echo ""
echo "=== STDOUT BUFFER CHECK ==="
# stdout goes to pipe:[63029], check if Python is using buffered I/O
python3 -c "import sys; print('Python default stdout buffering:', sys.stdout.line_buffering)" 2>/dev/null
