# Running this ds4 fork on a Mac: setup guide

This guide sets up the `gaineyllc/ds4` fork (branch `rank/1017`) on an Apple Silicon Mac to serve DeepSeek V4.1 Flash 1017. The server takes text and images, uses DSpark speculative decoding, and has a 1M-token context. It exposes an OpenAI-compatible API on `http://127.0.0.1:8031/v1`.

It is written so an AI coding agent can follow it step by step on the target Mac. Each step ends with a check. Do not continue past a failed check; see Troubleshooting. The model files are described in [MODEL.md](MODEL.md).

## 0. What you need
- **Mac:** Apple Silicon with 128 GB unified memory. Tuned on an M5 Max; M3 Max should work, but expect lower speed.
  - Less memory is possible with a smaller `DS4_BANK` and context, but it is untested.
- **Disk:** about 420 GB free on the internal SSD:
  - 375 GB of model files
  - up to 40 GB of prompt cache
  - room for the build
- The model is read from disk while it runs. An external or network drive will be much slower; use the internal SSD.
- **macOS and tools:** a recent macOS and the Xcode Command Line Tools (`xcode-select --install`).
- **Download access:** the download link and access code from Neil. They are not in this repo.

Check:
```sh
sysctl -n hw.memsize | awk '{printf "%.0f GB RAM\n", $1/2^30}'
df -h ~ | tail -1
xcode-select -p
```
Expected: 128 GB RAM, at least 420 GB available, and a developer-tools path.

## 1. Clone and build
Build on the Mac that will run it. The Makefile compiles with `-mcpu=native`, so binaries built on another Mac model may not run here.

```sh
git clone -b rank/1017 https://github.com/gaineyllc/ds4.git ~/ds4
cd ~/ds4
make -j"$(sysctl -n hw.ncpu)" ds4 ds4-server
```
Check: `./ds4-server --help | head -3` prints usage, and `ls metal | head` lists shader files.

The Metal shaders compile at startup from `./metal`. Always run the server from the repo directory; the launcher script does this for you.

## 2. Download the model files
The files are served over HTTPS from Neil's Mac:
- Downloads resume if interrupted.
- The link and code change if that server restarts. Ask Neil for a fresh link if downloads return 404 or fail to connect.

```sh
cd ~/ds4 && mkdir -p gguf && cd gguf
BASE="<link from Neil>"        # e.g. https://something.trycloudflare.com
CODE="<access code from Neil>"
curl -fsS "$BASE/$CODE/"        # lists the files and their sizes
for f in SHA256SUMS DeepSeek-V4.1-Flash-1017-Vision.gguf DeepSeek-V4.1-Flash-1017-DSpark.gguf DeepSeek-V4.1-Flash-1017-UNC-Q2.gguf; do
  until curl -fL -C - --retry 20 --retry-delay 10 -o "$f" "$BASE/$CODE/$f"; do echo "retrying $f"; sleep 15; done
done
shasum -a 256 -c SHA256SUMS
```
Check: `shasum -c` prints `OK` for all three `.gguf` files.

The 366 GB main file takes hours; speed depends on both internet connections. If the loop exits or the Mac sleeps, run it again and `curl -C -` resumes where it stopped. To keep the Mac awake while it runs, prefix the loop with `caffeinate -i`.

## 3. First run (foreground)
```sh
cd ~/ds4
./scripts/mac/ds4-mac-serve.sh &
sleep 90; tail -5 ~/.ds4/ds4-server.log
```
Check: the log ends with `listening on http://127.0.0.1:8031`. Also look for `memory guard: ... -> 2 sessions need N GiB`; that N must be below your RAM. If the guard reports that only 1 of 2 sessions fits, or startup fails, see Troubleshooting.

Then:
```sh
./scripts/mac/verify.sh
```
Check: the reply says `ds4 is working`, and a tokens-per-second figure prints.
- The very first request is slow, because the expert bank fills from disk.
- Run `verify.sh` a second time for a representative speed.
- Reference numbers on an M5 Max: about 15–20 tok/s once warm.

Stop the foreground server with `kill %1` before continuing.

## 4. Start at login and restart on failure (launchd)
```sh
cd ~/ds4
sed "s#__REPO__#$HOME/ds4#g" scripts/mac/com.ds4.server.plist.template > ~/Library/LaunchAgents/com.ds4.server.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.ds4.server.plist
sleep 120; ./scripts/mac/verify.sh
```
- Restart: `launchctl kickstart -k gui/$(id -u)/com.ds4.server`
- Stop: `launchctl bootout gui/$(id -u)/com.ds4.server`
- The server saves open sessions to the disk cache when stopped this way, so the next request resumes without re-reading the whole conversation.

## 5. Connect a client
Any OpenAI-compatible client works:
- Base URL: `http://127.0.0.1:8031/v1`
- API key: any non-empty string. There is no auth; the server only listens on localhost.
- Model name: anything.

Recommended request settings:
- `reasoning_effort: "none"` and `temperature: 0` for agent and tool use. This is fast, deterministic, and gets the most out of DSpark.
- `reasoning_effort: "high"` for hard problems.
  - **Important:** if a request does not set reasoning, ds4 defaults to high reasoning. Many agent frameworks leave the field out when "reasoning off" is selected, so set it explicitly, for example through the client's extra body parameters.
- Set a max output size, for example 32k tokens. Without one, a reasoning loop can run for hours and fill the context.
- Keep the reasoning setting fixed within a conversation. It changes the top of the system prompt, so switching mid-conversation forces the whole history to be re-read from scratch.

Images: send OpenAI-style `image_url` content parts. The vision encoder is loaded by default.

## Settings (`~/.ds4/mac.env`, optional)
| Variable | Default | Meaning |
|---|---|---|
| `DS4_BANK` | `56GB` | Experts kept resident in memory. Larger is faster, until macOS starts swapping. |
| `DS4_CTX` | `1048576` | Context window. Lower it to save memory. |
| `DS4_SLOTS` | `2` | Conversations kept resident. Concurrent requests take 64-token turns. |
| `DS4_KVMB` | `40000` | Prompt-cache disk budget in MB. |
| `DS4_PORT` | `8031` | API port. |

Restart the service after changing them.

## Troubleshooting
- **`memory guard: only 1 of 2 resident sessions fit` or out-of-memory at start:** close large apps, or set `DS4_BANK=48GB` (then `40GB`), or `DS4_SLOTS=1`.
- **`DSpark support model and target checkpoint families differ`:** the wrong DSpark file. Use `DeepSeek-V4.1-Flash-1017-DSpark.gguf` from the download, not an older `DSpark-native` build.
- **`shasum` mismatch:** delete that file and download it again. `curl -C -` cannot repair corrupted bytes.
- **Very slow decode (under 5 tok/s) after warm-up:** confirm the model is on the internal SSD (`df ~/ds4/gguf`), and check `vm_stat` and `sysctl vm.swapusage` for heavy swapping.
- **Metal shader errors at start:** the server was started outside the repo directory, or `metal/` is missing. Use `scripts/mac/ds4-mac-serve.sh`.
- **Logs:** `~/.ds4/ds4-server.log`. `prefill chunk` lines show prompt reading speed; `decoding chunk` lines show generation speed.

## Updating
```sh
cd ~/ds4 && git pull && make -j"$(sysctl -n hw.ncpu)" ds4 ds4-server
launchctl kickstart -k gui/$(id -u)/com.ds4.server
```
