# Using this ds4 server with Hermes Agent

ds4 keeps each conversation's prompt cached, both in memory and on disk. As long as every new request starts with exactly the same text as the last one, ds4 only reads the new part. If anything early in the prompt changes, ds4 must re-read everything after that point. At 300k+ tokens that takes 15–40 minutes.

Out of the box, Hermes breaks this in several ways. `scripts/mac/hermes-ds4-setup.py` fixes all of them.

## What the script changes, and why
1. **Patches one line of Hermes's code** (`tools/mcp_schema_cache.py`).
   - Hermes treats an MCP server's "no expiry" cache setting (0) as "already expired".
   - So `lazy: true` MCP servers quietly connect at startup anyway, and a server that is down drops out of the tool list near the top of the prompt.
   - Every time an MCP server goes up or down, that list changes, and ds4 re-reads everything after it.
   - The script backs up the file first. It registers itself as a Hermes `post_update` command, so the patch is re-applied after every `hermes update`.
2. **Adds the ds4 provider** `ds4-local` (`http://127.0.0.1:8031/v1`, model `deepseek-v4.1-flash`, 1M context). Its `extra_body` sends:
   - **`reasoning_effort: none`**: Hermes leaves the field out when reasoning is "off", and ds4 then defaults to *high* reasoning.
   - **`temperature: 0`**: deterministic output, and it gets the most out of DSpark.
   - **`max_tokens: 32768`**: caps a runaway generation, so Hermes's own repetition guard can step in. Without a cap, one reasoning loop can run for hours.
3. **Sets `lazy: true`** on every enabled MCP server.
4. **Compacts at 95%** of the 1M window (`compression.model_thresholds.deepseek-v4: 0.95`). Hermes's default is 50%.
5. **Moves Hermes's helper tasks off ds4.** These include background review, compaction summaries, titles, the approval and MCP helpers, the goal judge and others. On "auto" they run on the main model, take over the conversation's slot on ds4, and push it out of memory.
   - Background review is the worst: it copies the whole conversation, adds its own turns, and leaves ds4 holding that copy instead of yours.
   - With `--helper-provider/--helper-model`, the helpers use that second model.
   - Without them, background review is turned off and the other helpers stay on the main model. They're short calls, so the cost is small.

The script never changes a helper task you've already pointed somewhere specific. It also backs up every config file it changes as `config.yaml.bak-ds4-<time>`.

## Run it
```sh
cd ~/ds4
# See what would change:
~/.hermes/hermes-agent/venv/bin/python3 scripts/mac/hermes-ds4-setup.py --check
# Apply, with a second model for helpers (recommended). The provider must already exist in your Hermes config:
~/.hermes/hermes-agent/venv/bin/python3 scripts/mac/hermes-ds4-setup.py --helper-provider <provider> --helper-model <model>
# ...or without one:
~/.hermes/hermes-agent/venv/bin/python3 scripts/mac/hermes-ds4-setup.py
```
- Add `--all-profiles` to configure every Hermes profile, or `--profile NAME` for one.
- Running it again changes nothing that's already set.

Check: `--check` now reports `0 setting(s) would change`, and the patch line says `already applied`.

Then start a **new** Hermes session and choose provider `ds4-local`, model `deepseek-v4.1-flash`.

## Rules that keep the cache warm
- **Don't change reasoning, model or system-prompt settings in the middle of a conversation.**
  - ds4 puts the reasoning level at the very top of the prompt. Changing it forces the whole history to be re-read; we've seen 410k tokens take about 25 minutes.
  - To use different settings, start a new session.
- **Expect the first message of a new session to be slow**, possibly minutes, while ds4 reads the system prompt and loads experts from disk. Later turns only read what's new.
- **Two conversations at once is fine:** ds4 holds two sessions and gives them turns. A third waits for a free slot. A session that's pushed out is saved to disk and reloads in under a second.
