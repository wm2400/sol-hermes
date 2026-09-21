# sol-hermes

Token-efficiency plugin for Hermes Agent. Four mechanisms not in Hermes core.

## What it does

**economic_compact_check** — decides when to compress context using a cache write/read ratio (default 12.5). Hermes uses fixed 50%/85% thresholds; this uses economics.

**context_project / context_recall** — projects large tool results into compact handles before the provider sees them. Unlike Hermes's built-in truncation (head+tail at 100KB), this uses configurable thresholds and stores full content for paged recall.

**evidence_compress / evidence_verify** — extracts error sections from logs with word-boundary regex. Falls back to regex if no LLM is available for semantic compression.

**sol_patch_validate** — edit + validate in one call. Detects ambiguous matches (rejects if old_string appears multiple times without replace_all).

## Install

```bash
git clone https://github.com/wm2400/sol-hermes ~/.hermes/plugins/sol-hermes
```

Add to `plugins.enabled` in `~/.hermes/config.yaml`:

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

`~/.hermes/sol-hermes.json`:

```json
{
  "action_fusion": true,
  "observation_pack": true,
  "evidence_reducer": true,
  "online_compact": true,
  "observation_threshold": 4000,
  "max_log_lines": 80,
  "max_storage_mb": 500,
  "obs_ttl_hours": 24,
  "cache_write_read_ratio": 12.5
}
```

## Security

- No `shell=True` — validators run as list-form subprocess
- `handle_id` and `session_id` sanitized to `[a-zA-Z0-9_-]`
- Config loaded only from `~/.hermes/`, never from CWD
- Observations auto-delete after 24h, storage capped at 500MB
- No network calls, no telemetry

## Development

```bash
python -m unittest test_sol_hermes -v
```

## License

MIT
