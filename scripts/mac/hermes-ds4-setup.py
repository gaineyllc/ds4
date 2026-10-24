#!/usr/bin/env python3
"""Configure Hermes Agent to use a local ds4-server without busting its prompt cache.

Run with Hermes's own Python (it has ruamel.yaml):
  ~/.hermes/hermes-agent/venv/bin/python3 scripts/mac/hermes-ds4-setup.py [options]

Options:
  --helper-provider NAME --helper-model MODEL
        A second model (a provider already defined in your Hermes config) for
        Hermes's helper tasks: background review, compaction summaries, titles,
        and the rest. Strongly recommended. Without it, background review is
        turned off, and the other helpers keep using the main model.
  --profile NAME     Configure ~/.hermes/profiles/NAME instead of ~/.hermes.
  --all-profiles     Configure ~/.hermes and every profile.
  --patch-only       Only (re)apply the Hermes code patch. Registered as a
                     post-update command, so it survives `hermes update`.
  --check            Report status; change nothing.
"""
import argparse, glob, os, shutil, sys, time
from ruamel.yaml import YAML

HOME = os.path.expanduser(os.environ.get("HERMES_HOME", "~/.hermes"))
AGENT = os.path.expanduser(os.environ.get("HERMES_AGENT_DIR", "~/.hermes/hermes-agent"))
SCRIPT = os.path.abspath(__file__)
PROVIDER = "ds4-local"
BASE_URL = "http://127.0.0.1:8031/v1"
MODEL = "deepseek-v4.1-flash"
STAMP = time.strftime("%Y%m%d-%H%M%S")
HELPER_TASKS = ["compression", "background_review", "skills_hub", "approval", "review", "mcp",
                "title_generation", "memory_query_rewrite", "tts_audio_tags", "triage_specifier",
                "kanban_decomposer", "profile_describer", "goal_judge", "curator", "monitor",
                "moa_reference", "moa_aggregator", "compact"]

# --- Hermes code patch -------------------------------------------------------------
# The MCP SDK reports ttl_ms=0 when a server sends no cache TTL, and Hermes's schema
# cache treated 0 as "already expired". So `lazy: true` MCP servers fell back to an
# eager connect, and an MCP server that was down dropped out of the tool catalog in
# the system prompt. Every MCP up/down flip then changed the prompt about 15k tokens in,
# and ds4 had to re-read everything after that point. Only a positive TTL should expire.
PATCH_FILE = os.path.join(AGENT, "tools", "mcp_schema_cache.py")
BUGGY = "expired = (isinstance(ttl_ms, (int, float)) and isinstance(written_at, (int, float))"
FIXED = "expired = (isinstance(ttl_ms, (int, float)) and ttl_ms > 0 and isinstance(written_at, (int, float))"

def patch(check=False):
    try:
        src = open(PATCH_FILE).read()
    except OSError:
        print(f"patch: {PATCH_FILE} not found; this Hermes version may differ. Skipped.")
        return False
    if FIXED in src:
        print("patch: mcp schema cache zero-TTL fix already applied"); return True
    if BUGGY not in src:
        print("patch: expected line not found; Hermes may have fixed or changed this code. Skipped.")
        return False
    if check:
        print("patch: NOT applied"); return False
    shutil.copy2(PATCH_FILE, f"{PATCH_FILE}.bak-{STAMP}")
    open(PATCH_FILE, "w").write(src.replace(BUGGY, FIXED, 1))
    print("patch: applied mcp schema cache zero-TTL fix"); return True

# --- config ------------------------------------------------------------------------
def configure(cfg_path, helper_provider, helper_model, check=False):
    y = YAML(); y.preserve_quotes = True; y.width = 4096
    d = y.load(open(cfg_path)) if os.path.exists(cfg_path) else None
    if d is None:
        print(f"{cfg_path}: missing or empty; run Hermes once first"); return
    changes = []
    def setdefault_map(parent, key):
        if parent.get(key) is None: parent[key] = {}
        return parent[key]
    providers = setdefault_map(d, "providers")
    p = setdefault_map(providers, PROVIDER)
    want = {"name": "DeepSeek V4.1 Flash (local ds4)", "base_url": BASE_URL, "api_key": "ds4-local",
            "context_length": 1048576, "request_timeout_seconds": 5400, "stale_timeout_seconds": 1800}
    for k, v in want.items():
        if p.get(k) != v: p[k] = v; changes.append(f"providers.{PROVIDER}.{k}")
    models = setdefault_map(p, "models")
    if MODEL not in models: models[MODEL] = {"context_length": 1048576}; changes.append(f"providers.{PROVIDER}.models.{MODEL}")
    # Reasoning off and greedy, set explicitly: Hermes omits the field when reasoning is
    # "off" and ds4 then defaults to HIGH. max_tokens caps a runaway generation so
    # Hermes's own repetition guard can act. None of these change the prompt text.
    eb = setdefault_map(p, "extra_body")
    for k, v in {"reasoning_effort": "none", "temperature": 0, "max_tokens": 32768}.items():
        if eb.get(k) != v: eb[k] = v; changes.append(f"providers.{PROVIDER}.extra_body.{k}")
    # Compact only near the 1M window (Hermes's default is 50%).
    comp = setdefault_map(d, "compression")
    mt = setdefault_map(comp, "model_thresholds")
    if mt.get("deepseek-v4") != 0.95: mt["deepseek-v4"] = 0.95; changes.append("compression.model_thresholds.deepseek-v4")
    # MCP servers: register from the schema cache, so an MCP outage never changes the prompt.
    for name, srv in (d.get("mcp_servers") or {}).items():
        if isinstance(srv, dict) and srv.get("enabled", True) and srv.get("lazy") is not True:
            srv["lazy"] = True; changes.append(f"mcp_servers.{name}.lazy")
    # Helper tasks off the main model: on "auto" they run on ds4, take the main session's
    # slot and evict it (background review was the worst: it appends its own turns).
    aux = setdefault_map(d, "auxiliary")
    for t in HELPER_TASKS:
        b = setdefault_map(aux, t)
        if helper_provider:
            if b.get("provider") in (None, "", "auto") and not b.get("base_url"):
                b["provider"] = helper_provider; b["model"] = helper_model; changes.append(f"auxiliary.{t}")
    if helper_provider:
        c = aux["compression"]
        if c.get("context_length") != 1048576:
            # Hermes caps the trigger at the summarizer's window, but sends it at most
            # 160k characters, so declaring 1M is safe and keeps compaction near 1M.
            c["context_length"] = 1048576; changes.append("auxiliary.compression.context_length")
    else:
        br = aux["background_review"]
        if br.get("enabled") is not False:
            br["enabled"] = False; changes.append("auxiliary.background_review.enabled=false")
    # Re-apply the code patch after every `hermes update`.
    pu = setdefault_map(d, "post_update")
    cmds = pu.get("commands")
    if cmds is None: pu["commands"] = []; cmds = pu["commands"]
    cmd = f"{AGENT}/venv/bin/python3 {SCRIPT} --patch-only"
    if not any(isinstance(c, dict) and c.get("command") == cmd for c in cmds):
        cmds.append({"command": cmd, "timeout": 120}); changes.append("post_update.commands")
    if check:
        print(f"{cfg_path}: {len(changes)} setting(s) would change" + (": " + ", ".join(changes) if changes else "")); return
    if changes:
        shutil.copy2(cfg_path, f"{cfg_path}.bak-ds4-{STAMP}")
        with open(cfg_path, "w") as fh: y.dump(d, fh)
    print(f"{cfg_path}: {len(changes)} change(s)" + (": " + ", ".join(changes) if changes else ""))

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--helper-provider"); ap.add_argument("--helper-model")
    ap.add_argument("--profile"); ap.add_argument("--all-profiles", action="store_true")
    ap.add_argument("--patch-only", action="store_true"); ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if bool(a.helper_provider) != bool(a.helper_model):
        ap.error("--helper-provider and --helper-model go together")
    patch(check=a.check)
    if a.patch_only: return
    if a.all_profiles: paths = [os.path.join(HOME, "config.yaml")] + sorted(glob.glob(os.path.join(HOME, "profiles", "*", "config.yaml")))
    elif a.profile: paths = [os.path.join(HOME, "profiles", a.profile, "config.yaml")]
    else: paths = [os.path.join(HOME, "config.yaml")]
    for pth in paths: configure(pth, a.helper_provider, a.helper_model, check=a.check)
    if not a.check:
        print(f"\nDone. In Hermes, pick provider '{PROVIDER}', model '{MODEL}'. Start new sessions after changing settings.")

if __name__ == "__main__":
    main()
