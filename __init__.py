"""sol-hermes: token-saving tools for Hermes."""

import json
import re
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime


def load_config():
    """Read config from sol-hermes.json if it exists."""
    paths = [
        Path.cwd() / ".hermes" / "sol-hermes.json",
        Path.home() / ".hermes" / "sol-hermes.json",
    ]
    defaults = {
        "action_fusion": True,
        "observation_pack": True,
        "evidence_reducer": True,
        "observation_threshold": 4000,
        "max_log_lines": 80,
        "validators": {
            ".py": ["python3 -m py_compile {path}"],
            ".js": ["node --check {path}"],
            ".ts": ["npx tsc --noEmit"],
            ".json": ["python3 -m json.tool {path}"],
            ".yaml": ["python3 -c 'import yaml,sys; yaml.safe_load(open(sys.argv[1]))' {path}"],
            ".yml": ["python3 -c 'import yaml,sys; yaml.safe_load(open(sys.argv[1]))' {path}"],
        },
    }
    for p in paths:
        if p.exists():
            try:
                with open(p) as f:
                    user = json.load(f)
                return {**defaults, **user}
            except Exception:
                pass
    return defaults


CFG = load_config()


# --- action fusion -----------------------------------------------------------

def get_validators(path):
    ext = Path(path).suffix.lower()
    cmds = CFG["validators"].get(ext, [])
    return [c.replace("{path}", path) for c in cmds]


def run_cmd(cmd, cwd):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30, cwd=cwd)
        return {
            "cmd": cmd,
            "exit": p.returncode,
            "stdout": p.stdout[:800] if p.stdout else "",
            "stderr": p.stderr[:800] if p.stderr else "",
        }
    except Exception as e:
        return {"cmd": cmd, "error": str(e)}


def sol_patch_validate(path, old_string, new_string, replace_all=False, validate=True):
    """Edit file and check it in one go."""
    result = {"ok": False, "file": path, "edit_applied": False, "validation": None}

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
        result["validation"] = [run_cmd(c, str(Path(path).parent)) for c in cmds]

    return json.dumps(result)


# --- observation pack --------------------------------------------------------

class ObsStore:
    def __init__(self):
        self.dir = Path.home() / ".hermes" / "sol-hermes" / "obs"
        self.dir.mkdir(parents=True, exist_ok=True)

    def put(self, content, kind="text", session="default"):
        if len(content) < CFG["observation_threshold"]:
            return {"type": "inline", "content": content, "chars": len(content)}

        h = hashlib.sha256(content.encode()).hexdigest()[:12]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        hid = f"obs_{ts}_{h}"

        d = self.dir / session
        d.mkdir(exist_ok=True)
        path = d / f"{hid}.txt"
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

    def get(self, hid, offset=0, limit=2000):
        for d in self.dir.iterdir():
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
    return json.dumps(_store.put(content, content_type, session_id))


def observation_recall(handle_id, offset=0, limit=2000):
    return json.dumps(_store.get(handle_id, offset, limit))


# --- evidence reducer --------------------------------------------------------

_ERR_RE = re.compile(
    r"error|failed|exception|traceback|warning|fatal|panic",
    re.IGNORECASE,
)


def evidence_compress(log_content, context_lines=3):
    """Pull out error sections from a long log."""
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
    return json.dumps({
        "type": "compressed",
        "original_lines": total,
        "compressed_lines": len(compressed.split("\n")),
        "sections": len(sections),
        "content": compressed,
        "ratio": f"{total / max(1, len(compressed.split(chr(10)))):.1f}x",
    })


def evidence_verify(compressed_content, original_content):
    """Check that every quoted section actually exists in the original."""
    parts = re.findall(
        r"--- section \d+ \(lines \d+-\d+\) ---\n(.+?)(?=\n\n|\Z)",
        compressed_content,
        re.DOTALL,
    )
    ok = True
    results = []
    for p in parts:
        p = p.strip()
        found = p in original_content
        results.append({"preview": p[:80] + ("..." if len(p) > 80 else ""), "ok": found})
        if not found:
            ok = False
    return json.dumps({"all_ok": ok, "checked": len(results), "details": results})


# --- hook: auto-pack big outputs ---------------------------------------------

def _auto_pack(tool_name, params, result, **kw):
    if not CFG["observation_pack"]:
        return
    if tool_name not in ("terminal", "read_file", "search_files"):
        return
    if len(result) < CFG["observation_threshold"]:
        return
    packed = _store.put(result, f"auto_{tool_name}", "auto")
    if packed["type"] == "handle":
        print(f"[sol-hermes] packed {len(result)} chars from {tool_name} -> {packed['handle_id']}")


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
                },
                "required": ["handle_id"],
            },
        },
        handler=lambda p, **kw: observation_recall(
            p["handle_id"], p.get("offset", 0), p.get("limit", 2000),
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

    ctx.register_hook("post_tool_call", _auto_pack)

    print("[sol-hermes] loaded")
