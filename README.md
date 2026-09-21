# sol-hermes

Hermes plugin that cuts token usage. Four mechanisms, all local, no extra API calls.

## What it does

**sol_patch_validate** — edit a file and run its syntax check in one tool call instead of two. Validators run without `shell=True` (list-form subprocess, no injection). Empty `old_string` rejected.

**observation_pack / observation_recall** — large tool output stored on disk, model gets a short handle. Session names sanitized to prevent path traversal. Recall validates offset/limit.

**transform_tool_result hook** — automatically replaces oversized `terminal`, `read_file`, `search_files` results with a handle before the model sees them. This is the real token saver.

**evidence_compress / evidence_verify** — pulls error sections from logs with word-boundary regex (no false positives on "errorless"). Verify strips section headers before matching.

## Install

```bash
git clone https://github.com/wm2400/sol-hermes ~/.hermes/plugins/sol-hermes
```

Add `sol-hermes` to `plugins.enabled` in `~/.hermes/config.yaml`:

```yaml
plugins:
  enabled:
    - sol-hermes
```

Restart Hermes.

## Uninstall

```bash
sh ~/.hermes/plugins/sol-hermes/uninstall.sh
```

## Config

Only from `~/.hermes/sol-hermes.json` (never from project directory — security boundary):

```json
{
  "action_fusion": true,
  "observation_pack": true,
  "evidence_reducer": true,
  "observation_threshold": 4000,
  "max_log_lines": 80,
  "max_storage_mb": 500,
  "obs_ttl_hours": 24
}
```

## Security

- No `shell=True` — validators run as list-form subprocess
- Session names sanitized to `[a-zA-Z0-9_-]`
- Config loaded only from `~/.hermes/`, never from CWD
- Observations auto-delete after 24h, storage capped at 500MB
- No network calls, no telemetry

## Development

```bash
python -m unittest test_sol_hermes -v
```

## License

MIT
