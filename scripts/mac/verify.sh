#!/bin/zsh
# Checks a running ds4-server: models endpoint, a short reply, and decode speed.
P=${DS4_PORT:-8031}; U=http://127.0.0.1:$P/v1
echo "== /v1/models"; curl -fsS -m 10 $U/models || { echo "server not answering on $P"; exit 1; }; echo
echo "== short reply (first request also warms the expert bank, allow a few minutes)"
python3 - "$U" <<'PY'
import json, sys, time, urllib.request
u = sys.argv[1] + "/chat/completions"
def ask(prompt, n):
    b = {"model": "ds4", "temperature": 0, "max_tokens": n, "reasoning_effort": "none",
         "messages": [{"role": "user", "content": prompt}]}
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(u, json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=1800))
    return time.time() - t, r
dt, r = ask("Reply with exactly: ds4 is working", 16)
print(f"{dt:.1f}s:", r["choices"][0]["message"]["content"].strip())
dt, r = ask("Write a Python function that parses ISO-8601 durations, with tests.", 300)
n = r["usage"]["completion_tokens"]
print(f"decode: {n} tokens in {dt:.1f}s = {n/dt:.1f} tok/s (includes prompt time; warm runs are faster)")
PY
