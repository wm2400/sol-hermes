"""sol-hermes: token-saving tools for Hermes."""

import json
import re
import hashlib
import subprocess
import warnings
import threading
import os
from pathlib import Path
from datetime import datetime, timedelta


def load_config():
    """Read config from sol-hermes.json if it exists. Never load from CWD."""
    config_path = Path.home() / ".hermes" / "sol-hermes.json"
    defaults = {
        "action_fusion": True,
        "observation_pack": True,
        "evidence_reducer": True,
        "observation_threshold": 4000,
        "max_log_lines": 80,
        "max_storage_mb": 500,
        "obs_ttl_hours": 24,
        "validators": {
            ".py": ["python3", "-m", "py_compile"],
            ".js": ["node", "--check"],
            ".ts": ["npx", "tsc", "--noEmit"],
            ".json": ["python3", "-m", "json.tool"],
            ".yaml": ["python3", "-c", "import yaml,sys; yaml.safe_load(open(sys.argv[1]))"],
            ".yml": ["python3", "-c", "import yaml,sys; yaml.safe_load(open(sys.argv[1]))"],
        },
    }
    if not config_path.exists():
        return defaults
    try:
        with open(config_path) as f:
            user = json.load(f)
        unknown = set(user.keys()) - set(defaults.keys())
        if unknown:
            warnings.warn(f"sol-hermes: unknown config keys {unknown}, using defaults")
        return {**defaults, **{k: v for k, v in user.items() if k in defaults}}
    except json.JSONDecodeError as e:
        warnings.warn(f"sol-hermes: bad JSON in {config_path}: {e}, using defaults")
    except Exception as e:
        warnings.warn(f"sol-hermes: failed to load {config_path}: {e}, using defaults")
    return defaults


try:
    CFG = load_config()
except Exception:
    CFG = {
        "action_fusion": True,
        "observation_pack": True,
        "evidence_reducer": True,
        "observation_threshold": 4000,
        "max_log_lines": 80,
        "max_storage_mb": 500,
        "obs_ttl_hours": 24,
        "validators": {},
    }


def estimate_tokens(text):
    """Rough token estimate: 1 token ~ 4 chars for English, ~3 for code."""
    if not text:
        return 0
    return max(1, len(text) // 3)


def count_saved(before, after):
    tb = estimate_tokens(before)
    ta = estimate_tokens(after)
    saved = tb - ta
    return {
        "tokens_before": tb,
        "tokens_after": ta,
        "tokens_saved": saved,
        "saved_percent": round((saved / max(1, tb)) * 100, 1),
    }


# --- action fusion -----------------------------------------------------------

def get_validators(path):
    ext = Path(path).suffix.lower()
    return CFG["validators"].get(ext, [])


def run_cmd(cmd_list, file_path):
    """Run validator without shell=True. Path passed as argv, not interpolated."""
    try:
        full_cmd = [c.replace("{path}", file_path) for c in cmd_list]
        p = subprocess.run(
            full_cmd,
            shell=False,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(Path(file_path).parent),
        )
        return {
            "cmd": " ".join(full_cmd),
            "exit": p.returncode,
            "stdout": p.stdout[:800] if p.stdout else "",
            "stderr": p.stderr[:800] if p.stderr else "",
        }
    except Exception as e:
        return {"cmd": " ".join(cmd_list), "error": str(e)}


def sol_patch_validate(path, old_string, new_string, replace_all=False, validate=True):
    """Edit file and check it in one go."""
    result = {"ok": False, "file": path, "edit_applied": False, "validation": None}

    if not old_string:
        result["error"] = "old_string cannot be empty"
        return json.dumps(result)

    try:
        with open(path, "r") as f:
            content = f.read()
        if old_string not in content:
            result["error"] = "string not found"
            return json.dumps(result)
        if replace_all:
            new_content = content.replace(old_string, new_string)
        else:
            new_content = content.replace(old_string, new_string, 1)
        with open(path, "w") as f:
            f.write(new_content)
        result["edit_applied"] = True
        result["ok"] = True
    except Exception as e:
        result["error"] = str(e)
        return json.dumps(result)

    if validate and CFG["action_fusion"]:
        cmds = get_validators(path)
        result["validation"] = [run_cmd(c, path) for c in cmds]
        result["token_stats"] = count_saved(
            json.dumps({"path": path, "old": old_string, "new": new_string}),
            json.dumps(result),
        )

    return json.dumps(result)


# --- observation pack --------------------------------------------------------

class ObsStore:
    def __init__(self):
        self.dir = Path.home() / ".hermes" / "sol-hermes" / "obs"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cleanup_old()

    def _cleanup_old(self):
        """Remove expired observations and enforce storage cap."""
        try:
            now = datetime.now()
            ttl = timedelta(hours=CFG["obs_ttl_hours"])
            total_size = 0
            files = []

            for f in self.dir.rglob("*.txt"):
                try:
                    stat = f.stat()
                    age = now - datetime.fromtimestamp(stat.st_mtime)
                    if age > ttl:
                        f.unlink()
                        continue
                    files.append((stat.st_mtime, f, stat.st_size))
                    total_size += stat.st_size
                except Exception:
                    pass

            max_bytes = CFG["max_storage_mb"] * 1024 * 1024
            if total_size > max_bytes:
                files.sort()
                for _, f, size in files:
                    if total_size <= max_bytes:
                        break
                    try:
                        f.unlink()
                        total_size -= size
                    except Exception:
                        pass
        except Exception:
            pass

    def _sanitize_session(self, session):
        """Only allow safe session names."""
        if not session:
            return "default"
        return re.sub(r"[^a-zA-Z0-9_-]", "", str(session))[:64] or "default"

    def _sanitize_handle(self, hid):
        """Only allow safe handle names. Prevents path traversal."""
        if not hid:
            return None
        # Only allow alphanumeric, underscore, hyphen
        clean = re.sub(r"[^a-zA-Z0-9_-]", "", str(hid))
        if not clean or clean != hid:
            return None
        return clean[:128]

    def put(self, content, kind="text", session="default"):
        if not isinstance(content, str):
            return {"type": "inline", "content": str(content), "chars": 0}
        if len(content) < CFG["observation_threshold"]:
            return {"type": "inline", "content": content, "chars": len(content)}

        session = self._sanitize_session(session)
        h = hashlib.sha256(content.encode()).hexdigest()[:12]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        hid = f"obs_{ts}_{h}"

        d = self.dir / session
        d.mkdir(exist_ok=True)
        path = d / f"{hid}.txt"

        # Atomic write with lock to prevent race conditions
        with self._lock:
            with open(path, "w") as f:
                f.write(content)

        return {
            "type": "handle",
            "handle_id": hid,
            "chars": len(content),
            "preview": content[:400] + "..." if len(content) > 400 else content,
            "path": str(path),
            "note": f"stored {len(content)} chars, recall with observation_recall",
        }

    def get(self, hid, offset=0, limit=2000, session=None):
        """Recall from a specific session, or search all if session=None."""
        # Sanitize handle_id to prevent path traversal
        hid = self._sanitize_handle(hid)
        if not hid:
            return {"error": "invalid handle_id"}

        if offset < 0:
            return {"error": "offset must be >= 0"}
        if limit < 1 or limit > 50000:
            return {"error": "limit must be 1-50000"}

        dirs = [self.dir / session] if session else list(self.dir.iterdir())
        for d in dirs:
            p = d / f"{hid}.txt"
            if p.exists():
                with open(p) as f:
                    content = f.read()
                chunk = content[offset:offset + limit]
                return {
                    "handle_id": hid,
                    "offset": offset,
                    "limit": limit,
                    "total_chars": len(content),
                    "chunk": chunk,
                    "has_more": offset + limit < len(content),
                    "next_offset": offset + limit if offset + limit < len(content) else None,
                }
        return {"error": "handle not found"}


_store = ObsStore()


def observation_pack(content, content_type="text", session_id="default"):
    result = _store.put(content, content_type, session_id)
    if result["type"] == "handle":
        result["token_stats"] = count_saved(content, json.dumps(result))
    return json.dumps(result)


def observation_recall(handle_id, offset=0, limit=2000, session_id=None):
    return json.dumps(_store.get(handle_id, offset, limit, session_id))


# --- evidence reducer --------------------------------------------------------

_ERR_RE = re.compile(
    r"\b(error|failed|exception|traceback|warning|fatal|panic)\b",
    re.IGNORECASE,
)


def evidence_compress(log_content, context_lines=3):
    """Pull out error sections from a long log."""
    if not isinstance(log_content, str):
        return json.dumps({"error": "log_content must be a string"})
    if not isinstance(context_lines, int) or context_lines < 0:
        context_lines = 3

    lines = log_content.split("\n")
    total = len(lines)

    if total <= CFG["max_log_lines"]:
        return json.dumps({"type": "full", "content": log_content, "lines": total, "compressed": False})

    sections = []
    seen = set()

    for i, line in enumerate(lines):
        if _ERR_RE.search(line):
            start = max(0, i - context_lines)
            end = min(len(lines), i + context_lines + 1)
            key = (start, end)
            if any(s <= key[0] and key[1] <= e for s, e in seen):
                continue
            seen.add(key)
            sections.append({
                "range": [start + 1, end],
                "content": "\n".join(lines[start:end]),
                "matched": line.strip(),
            })

    out = [f"[compressed: {total} lines -> {len(sections)} sections]", ""]
    for i, s in enumerate(sections, 1):
        out.append(f"--- section {i} (lines {s['range'][0]}-{s['range'][1]}) ---")
        out.append(s["content"])
        out.append("")

    compressed = "\n".join(out)
    stats = count_saved(log_content, compressed)
    return json.dumps({
        "type": "compressed",
        "original_lines": total,
        "compressed_lines": len(compressed.split("\n")),
        "sections": len(sections),
        "content": compressed,
        "ratio": f"{total / max(1, len(compressed.split(chr(10)))):.1f}x",
        "token_stats": stats,
    })


def evidence_verify(compressed_content, original_content):
    """Check that every quoted section actually exists in the original."""
    lines = compressed_content.split("\n")
    sections = []
    current = []
    in_section = False

    for line in lines:
        if line.startswith("--- section "):
            if in_section and current:
                sections.append("\n".join(current))
            current = []
            in_section = True
        elif in_section:
            if line.strip() == "":
                if current:
                    sections.append("\n".join(current))
                    current = []
                in_section = False
            else:
                current.append(line)

    if in_section and current:
        sections.append("\n".join(current))

    if not sections:
        return json.dumps({"all_ok": False, "error": "no sections found", "checked": 0, "details": []})

    ok = True
    results = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        found = section in original_content
        results.append({"preview": section[:80] + ("..." if len(section) > 80 else ""), "ok": found})
        if not found:
            ok = False

    return json.dumps({"all_ok": ok, "checked": len(results), "details": results})


# --- hooks -------------------------------------------------------------------

def _transform_result(tool_name, args, result, **kw):
    """Replace large tool results with a handle. Actually saves tokens.

    Signature matches Hermes hook contract: (tool_name, args, result, **kw).
    """
    if not CFG["observation_pack"]:
        return result
    if tool_name not in ("terminal", "read_file", "search_files"):
        return result
    if not isinstance(result, str) or len(result) < CFG["observation_threshold"]:
        return result

    session_id = kw.get("session_id", "default")
    packed = _store.put(result, f"auto_{tool_name}", session_id)
    if packed["type"] == "handle":
        return json.dumps({
            "type": "handle",
            "handle_id": packed["handle_id"],
            "chars": packed["chars"],
            "preview": packed["preview"],
            "note": f"Large output auto-packed. Use observation_recall with handle_id '{packed['handle_id']}' to read it.",
        })
    return result


# --- plugin registration -----------------------------------------------------

def register(ctx):
    """Wire up tools and hooks."""

    ctx.register_tool(
        name="sol_patch_validate",
        toolset="coding",
        schema={
            "name": "sol_patch_validate",
            "description": "Edit a file and run validation in one call. Saves a model turn.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                    "replace_all": {"type": "boolean", "default": False},
                    "validate": {"type": "boolean", "default": True},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
        handler=lambda p, **kw: sol_patch_validate(
            p["path"], p["old_string"], p["new_string"],
            p.get("replace_all", False), p.get("validate", True),
        ),
    )

    ctx.register_tool(
        name="observation_pack",
        toolset="core",
        schema={
            "name": "observation_pack",
            "description": "Store large output on disk, return a handle. Use for long logs or file contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "content_type": {"type": "string", "default": "text"},
                    "session_id": {"type": "string", "default": "default"},
                },
                "required": ["content"],
            },
        },
        handler=lambda p, **kw: observation_pack(
            p["content"], p.get("content_type", "text"), p.get("session_id", "default"),
        ),
    )

    ctx.register_tool(
        name="observation_recall",
        toolset="core",
        schema={
            "name": "observation_recall",
            "description": "Get back content stored with observation_pack.",
            "parameters": {
                "type": "object",
                "properties": {
                    "handle_id": {"type": "string"},
                    "offset": {"type": "integer", "default": 0},
                    "limit": {"type": "integer", "default": 2000},
                    "session_id": {"type": "string", "default": None},
                },
                "required": ["handle_id"],
            },
        },
        handler=lambda p, **kw: observation_recall(
            p["handle_id"], p.get("offset", 0), p.get("limit", 2000), p.get("session_id"),
        ),
    )

    ctx.register_tool(
        name="evidence_compress",
        toolset="core",
        schema={
            "name": "evidence_compress",
            "description": "Compress a long log to just error sections with context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "log_content": {"type": "string"},
                    "context_lines": {"type": "integer", "default": 3},
                },
                "required": ["log_content"],
            },
        },
        handler=lambda p, **kw: evidence_compress(
            p["log_content"], p.get("context_lines", 3),
        ),
    )

    ctx.register_tool(
        name="evidence_verify",
        toolset="core",
        schema={
            "name": "evidence_verify",
            "description": "Check that compressed quotes exist in the original log.",
            "parameters": {
                "type": "object",
                "properties": {
                    "compressed_content": {"type": "string"},
                    "original_content": {"type": "string"},
                },
                "required": ["compressed_content", "original_content"],
            },
        },
        handler=lambda p, **kw: evidence_verify(
            p["compressed_content"], p["original_content"],
        ),
    )

    ctx.register_hook("transform_tool_result", _transform_result)

    print("[sol-hermes] loaded")
