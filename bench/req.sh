#!/bin/zsh
# usage: req.sh PORT MAXTOK [N]  -- same request N times, prints server-reported decode t/s
PORT=$1; MAX=${2:-200}; N=${3:-3}
for i in $(seq 1 $N); do
  t0=$(python3 -c 'import time;print(time.time())')
  out=$(curl -s -m 600 http://127.0.0.1:$PORT/v1/chat/completions -H 'content-type: application/json' -d "{\"model\":\"x\",\"temperature\":0,\"max_tokens\":$MAX,\"messages\":[{\"role\":\"user\",\"content\":\"Review this function for bugs and explain each one in detail:\\n\\nint sum(int *a, int n){int s;for(int i=0;i<=n;i++)s+=a[i];return s;}\"}]}")
  t1=$(python3 -c 'import time;print(time.time())')
  ct=$(echo "$out" | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d["usage"]["completion_tokens"])' 2>/dev/null)
  echo "run $i: completion_tokens=$ct wall=$(python3 -c "print(round($t1-$t0,2))")s"
done
