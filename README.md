# sol-hermes

Hermes plugin that cuts token usage. Four mechanisms, all local, no extra API calls.

## What it does

**sol_patch_validate** — edit a file and run its syntax check in one tool call instead of two. A normal `patch` then `terminal: python -m py_compile` costs a full model turn between them. This does both at once. Validators are configurable per file extension.

**observation_pack / observation_recall** — when a tool returns something huge (test log, big file, find output), this stores it on disk and hands back a short handle with a preview. The model reads exact pages of the original later if it needs to. A `post_tool_call` hook does this automatically for `terminal`, `read_file`, and `search_files` results over 4000 chars.

**evidence_compress / evidence_verify** — point it at a long log and it pulls out just the error sections with a few lines of context around each. Then you can verify every quoted section exists verbatim in the original. No paraphrasing, no hallucinated lines.

## Install

```bash
git clone https://github.com/danielkhachaturov/sol-hermes ~/.hermes/plugins/sol-hermes
```

Add `sol-hermes` to `plugins.enabled` in `~/.hermes/config.yaml`:

```yaml
plugins:
  enabled:
    - sol-hermes
```

Restart Hermes. That's it.

## Uninstall

```bash
rm -rf ~/.hermes/plugins/sol-hermes
rm -rf ~/.hermes/sol-hermes   # stored observations, optional
```

Remove from `plugins.enabled` and restart.

## Config

Optional. Drop a `sol-hermes.json` in `~/.hermes/` or in a project's `.hermes/` folder:

```json
{
  "action_fusion": true,
  "observation_pack": true,
  "evidence_reducer": true,
  "observation_threshold": 4000,
  "max_log_lines": 80,
  "validators": {
    ".py": ["python3 -m py_compile {path}"],
    ".go": ["go vet ./..."]
  }
}
```

Project config overrides user config. Missing config means defaults: everything on, threshold 4000 chars, standard validators for py/js/ts/json/yaml.

## Security

- Observations live in `~/.hermes/sol-hermes/obs/`, never leave your machine
- No network calls, no telemetry, no API keys
- `evidence_compress` is pure string processing
- Stdlib only. Audit is one file.

## License

MIT
