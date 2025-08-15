#!/usr/bin/env python3
"""
PMM Chat - Interactive interface for your Persistent Mind Model
Main entry point for chatting with your autonomous AI personality.
"""

import os
import sys
import argparse
import threading

# Add current directory to path for PMM imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from pmm.langchain_memory import PersistentMindMemory
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
from pmm.config import (
    get_default_model,
    get_model_config,
    list_available_models,
    AVAILABLE_MODELS,
    get_ollama_models,
)


# --- Logging helper (module-level, used across functions) ---
_prompt_reprinter = None  # set in interactive setup
_prompt_lock = threading.Lock()

def _log(level: str, msg: str):
    print(f"[pmm][{level}] {msg}", file=sys.stderr)
    # If we're in interactive mode, reprint the user prompt so logs don't appear after '👤 You:'
    global _prompt_reprinter
    if _prompt_reprinter:
        try:
            with _prompt_lock:
                _prompt_reprinter()
        except Exception:
            pass


# --- Self-indexing configuration (module-level, read before main()) ---
CODE_ROOT = os.getenv("PMM_CODE_ROOT", ".")
CODE_MAX_MB = float(os.getenv("PMM_CODE_MAX_MB", "2"))
CODE_EXT = set([
    e.strip().lower()
    for e in os.getenv(
        "PMM_CODE_EXT",
        ".py,.md,.json,.yml,.yaml,.toml,.ts,.tsx,.js,.jsx,.css,.txt",
    ).split(",")
    if e.strip()
])
CODE_INDEX_MODE = os.getenv("PMM_CODE_INDEX", "Auto")  # Auto | Off
MANIFEST_PATH = os.path.join(".pmm_code_manifest.json")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PMM Chat - Interactive AI personality interface"
    )
    parser.add_argument("--model", help="Model name or number from the menu")
    parser.add_argument(
        "--noninteractive",
        action="store_true",
        help="Force non-interactive mode; do not try to read from /dev/tty",
    )
    return parser.parse_args()


def show_model_selection(force_tty=True):
    """Show model selection interface and return selected model."""
    print("=== PMM Model Selection ===")
    print()

    # Show current default model at top
    default_model = get_default_model()
    default_config = get_model_config(default_model)
    default_cost_str = (
        f"${default_config.cost_per_1k_tokens:.4f}/1K"
        if default_config.cost_per_1k_tokens > 0
        else "Free (local)"
    )

    print(f"⭐ CURRENT DEFAULT: {default_model} ({default_config.provider})")
    print(f"   {default_config.description}")
    print(f"   Max tokens: {default_config.max_tokens:,} | Cost: {default_cost_str}")
    print()

    # Show all available models
    print("📋 Available Models:")
    available_models = list_available_models()
    for i, model_name in enumerate(available_models, 1):
        config = AVAILABLE_MODELS[model_name]
        cost_str = (
            f"${config.cost_per_1k_tokens:.4f}/1K"
            if config.cost_per_1k_tokens > 0
            else "Free (local)"
        )
        marker = "⭐" if model_name == default_model else f"{i:2d}."
        status = ""
        if config.provider == "ollama":
            # Quick check if Ollama model is available
            status = (
                " 🟢"
                if model_name in [m["name"] for m in get_ollama_models()]
                else " 🔴"
            )

        print(f"{marker} {model_name} ({config.provider}){status}")
        print(f"    {config.description}")
        print(f"    Max tokens: {config.max_tokens:,} | Cost: {cost_str}")
        print()

    print("💡 Select a model:")
    print("   • Press ENTER to use current default")
    print(
        "   • Type model number (1-{}) or exact model name".format(
            len(available_models)
        )
    )
    print("   • Type 'list' to see this menu again")
    print()

    # Handle piped input more gracefully
    if not sys.stdin.isatty():
        if not force_tty:
            print("🎯 Non-interactive mode detected, using default model")
            return default_model

        # Try to open /dev/tty for interactive selection even with piped stdin
        try:
            with open("/dev/tty", "r+") as tty:
                print(
                    "🎯 Piped input detected, but opening /dev/tty for model selection..."
                )
                while True:
                    tty.write("🎯 Your choice: ")
                    tty.flush()
                    choice = tty.readline().strip()

                    if not choice:
                        return default_model

                    if choice.lower() == "list":
                        tty.write("\n📋 Available Models (see above)\n")
                        continue

                    # Try to parse as number
                    if choice.isdigit():
                        idx = int(choice)
                        if 1 <= idx <= len(available_models):
                            selected_model = available_models[idx - 1]
                            tty.write(f"✅ Selected model {idx}: {selected_model}\n")
                            return selected_model
                        tty.write(
                            f"❌ Please enter a number between 1 and {len(available_models)}\n"
                        )
                        continue

                    # Try exact model name
                    if choice in available_models:
                        tty.write(f"✅ Selected model by name: {choice}\n")
                        return choice

                    tty.write(
                        f"❌ Unknown model '{choice}'. Type 'list' to see available models.\n"
                    )

        except Exception as e:
            print(
                f"🎯 Non-interactive mode & no /dev/tty available ({e}); using default model"
            )
            return default_model

    while True:
        try:
            choice = input("🎯 Your choice: ").strip()

            if not choice:
                return default_model

            if choice.lower() == "list":
                show_model_selection()
                continue

            # Try to parse as number
            try:
                choice_num = int(choice)
                if 1 <= choice_num <= len(available_models):
                    selected_model = available_models[choice_num - 1]
                    return selected_model
                else:
                    print(
                        f"❌ Please enter a number between 1 and {len(available_models)}"
                    )
                    continue
            except ValueError:
                pass

            # Try exact model name
            if choice in available_models:
                return choice

            print(f"❌ Unknown model '{choice}'. Type 'list' to see available models.")

        except KeyboardInterrupt:
            print("\n👋 Exiting model selection...")
            return None


def main():
    """Interactive chat with PMM using working LangChain memory system."""
    load_dotenv()
    args = parse_args()

    print("🧠 PMM Chat - Your Persistent AI Mind")
    print("=====================================\n")

    # Model selection
    if args.model:
        # Allow number or name from CLI
        available_models = list_available_models()
        chosen = None

        # Try as number first
        if args.model.isdigit():
            idx = int(args.model)
            if 1 <= idx <= len(available_models):
                chosen = available_models[idx - 1]
                print(f"✅ CLI selected model {idx}: {chosen}")

        # Try as exact name if number didn't work
        if not chosen and args.model in available_models:
            chosen = args.model
            print(f"✅ CLI selected model by name: {chosen}")

        if not chosen:
            print(f"❌ Invalid model '{args.model}', showing selection menu...")
            model_name = show_model_selection(force_tty=not args.noninteractive)
        else:
            model_name = chosen
    else:
        model_name = show_model_selection(force_tty=not args.noninteractive)

    if not model_name:
        return

    print(f"🔄 {model_name} selected... Loading model... Please wait...")
    print()

    # Initialize PMM with selected model
    model_config = get_model_config(model_name)
    # Only require API key for OpenAI provider
    if model_config.provider != "ollama" and not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set (required for OpenAI provider)")
        return

    pmm_memory = PersistentMindMemory(
        agent_path="persistent_self_model.json",
        personality_config={
            "openness": 0.7,
            "conscientiousness": 0.6,
            "extraversion": 0.8,
            "agreeableness": 0.9,
            "neuroticism": 0.3,
        },
    )

    print(f"🤖 Using model: {model_name} ({model_config.description})")

    # Show personality state
    personality = pmm_memory.get_personality_summary()
    print(f"📚 Loaded PMM with {personality['total_events']} events")
    print(
        f"🎭 Personality: O:{personality['personality_traits']['openness']:.2f} C:{personality['personality_traits']['conscientiousness']:.2f} E:{personality['personality_traits']['extraversion']:.2f} A:{personality['personality_traits']['agreeableness']:.2f} N:{personality['personality_traits']['neuroticism']:.2f}"
    )

    # Initialize LangChain components based on provider
    if model_config.provider == "ollama":
        llm = OllamaLLM(model=model_name, temperature=0.7)
    else:  # openai
        llm = ChatOpenAI(model=model_name, temperature=0.7)

    # (moved) Self-index local codebase happens after helper definitions below

    # --- Retrieval helpers (keyword fallback; embeddings later) ---
    STOP = {"the","and","for","with","your","that","this","from","have","what","when","where","who","how","are","you","him","her","its","ours","mine","just","okay","fine","please","thanks"}

    def _load_identity_facts(k: int = 5):
        rows = []
        try:
            rows = pmm_memory.pmm.sqlite_store.recent_by_etype("identity_info", limit=k)
        except Exception:
            rows = []
        out = []
        for (eid, ts, etype, summary, content, meta) in rows:
            s = summary or (content[:160] if content else "")
            out.append((eid, ts, s))
        return out

    def _semantic_matches_keyword(user_text: str, m: int = 5):
        toks = [t.lower() for t in (user_text or "").split() if len(t) > 3 and t.lower() not in STOP]
        if not toks:
            return []
        try:
            rows = pmm_memory.pmm.sqlite_store.recent_events(limit=300)
        except Exception:
            rows = []
        scored = []
        for (eid, ts, kind, content, meta, prev, hsh) in rows:
            text = (content or "").lower()
            score = sum(1 for t in toks if t in text)
            if score:
                scored.append((score, eid, ts, content))
        scored.sort(reverse=True)
        return [(eid, ts, (content or "")[:160]) for score, eid, ts, content in scored[:m]]

    # Create enhanced system prompt with PMM context and targeted retrieval
    def get_pmm_system_prompt(latest_user_msg: str = ""):
        base_context = pmm_memory.load_memory_variables({}).get("history", "")
        personality = pmm_memory.get_personality_summary()
        traits = personality["personality_traits"]
        agent_name = pmm_memory.pmm.model.core_identity.name

        facts = _load_identity_facts(k=5)
        related = _semantic_matches_keyword(latest_user_msg, m=5)
        facts_block = "\n".join([f"- E{eid} ({ts}): {s}" for (eid, ts, s) in facts]) or "- (none)"
        rel_block = "\n".join([f"- E{eid} ({ts}): {s}" for (eid, ts, s) in related]) or "- (none)"
        # Opportunistic code explanation if the user asks about code
        code_expl = ""
        try:
            ql = (latest_user_msg or "").lower()
            if any(k in ql for k in ("code", "function", "class", "script", "module", "def ", "what does _")):
                code_expl = _explain_code_snippet(pmm_memory, latest_user_msg, top=3)
        except Exception:
            code_expl = ""

        return (
            "You are a Persistent Mind Model (PMM). Use stored, provenance-backed facts. Do not invent identity facts.\n"
            "When asked about code, strictly use [CODE CONTEXT] and [CODE EXPLANATION].\n"
            "- Prefer [CODE CONTEXT] entries that match the query; treat them as the source of truth.\n"
            "- Cite EIDs (e.g., E123) and file paths with line ranges when describing code.\n"
            "- If context is insufficient, say so and suggest running 'code reflect <query>'.\n"
            "- Avoid generic summaries; focus on functions/classes actually present in cited snippets.\n\n"
            f"IDENTITY: {agent_name}\n"
            f"PERSONALITY TRAITS (Big Five): O={traits['openness']:.2f} C={traits['conscientiousness']:.2f} E={traits['extraversion']:.2f} A={traits['agreeableness']:.2f} N={traits['neuroticism']:.2f}\n\n"
            "PMM CONTEXT (use; do not invent):\n"
            "[FACTS]\n" + facts_block + "\n\n"
            + "[RELATED MEMORIES]\n" + rel_block + "\n\n"
            + "[CODE EXPLANATION]\n" + (code_expl or "- (none)") + "\n\n"
            + "CROSS-SESSION MEMORY (truncated):\n" + base_context[:1500]
        )

    # --- Command registry: load from memory, seed defaults, and bind actions ---
    def _parse_command_row(row):
        import json as _json  # local to avoid import order issues
        # row: (id, ts, etype, summary, content, meta)
        eid, ts, _etype, summary, content, meta = row
        name = None
        desc = None
        try:
            m = _json.loads(meta) if isinstance(meta, str) else (meta or {})
        except Exception:
            m = {}
        # Prefer explicit meta fields
        name = m.get("name") or m.get("command") or name
        desc = m.get("description") or m.get("desc") or desc
        # Try content as JSON
        if not name and isinstance(content, str) and content.strip().startswith("{"):
            try:
                cj = _json.loads(content)
                name = cj.get("name") or name
                desc = cj.get("description") or desc
            except Exception:
                pass
        # Fallback: parse summary patterns like "command: status - Show counts"
        if isinstance(summary, str):
            s = summary.strip()
            low = s.lower()
            if low.startswith("command:"):
                rest = s.split(":", 1)[1].strip()
                if " - " in rest:
                    nm, ds = rest.split(" - ", 1)
                    name = name or nm.strip()
                    desc = desc or ds.strip()
                else:
                    name = name or rest
            elif low.startswith("cmd:"):
                rest = s.split(":", 1)[1].strip()
                name = name or rest
        if name:
            return name.strip(), (desc or "(no description)").strip()
        return None, None

    def _load_command_registry():
        try:
            rows = pmm_memory.pmm.sqlite_store.recent_by_etype("command", limit=200)
        except Exception:
            rows = []
        reg = {}
        for row in rows:
            nm, ds = _parse_command_row(row)
            if nm and nm not in reg:
                reg[nm] = ds
        return reg

    def _seed_default_commands():
        defaults = [
            {"name": "help", "description": "List available PMM commands from memory"},
            {"name": "personality", "description": "Show current Big Five trait snapshot"},
            {"name": "memory", "description": "Show cross-session memory summary"},
            {"name": "status", "description": "Show counts by event type and quick stats"},
            {"name": "dump", "description": "Show last 5 raw events (debug)"},
            {"name": "evolution", "description": "Show recent self-reflections and personality drift"},
        ]
        existing = _load_command_registry()
        added = 0
        for cmd in defaults:
            if cmd["name"] not in existing:
                try:
                    pmm_memory.pmm.add_event(
                        summary=f"command: {cmd['name']} - {cmd['description']}",
                        effects=[],
                        etype="command",
                        full_text=None,
                        tags=["system", "command"],
                    )
                    added += 1
                except Exception:
                    pass
        if added:
            _log("info", f"seeded {added} default commands")

    def _print_command_list(reg):
        if not reg:
            print("\n🧩 No commands in memory yet.")
            return
        print("\n🧩 PMM Commands (from memory):")
        for nm in sorted(reg.keys()):
            print(f"   • {nm:<12} - {reg[nm]}")

    def _handle_evolution():
        # Show last few reflection events and any trait deltas if present in text
        print("\n🌱 Recent Evolution:")
        try:
            rows = pmm_memory.pmm.sqlite_store.recent_by_etype("reflection", limit=5)
        except Exception:
            rows = []
        if not rows:
            print("   • (no reflections yet)")
        else:
            for (eid, ts, _et, summary, content, _meta) in rows:
                line = (summary or content or "").strip().splitlines()[0][:100]
                print(f"   • E{eid} ({ts}): {line}")

    # Bind command names to handlers
    def _make_action_map():
        return {
            "help": lambda: _print_command_list(_load_command_registry()),
            "personality": lambda: (
                (lambda p: (
                    print("\n🎭 Current Personality State:"),
                    [print(f"   • {t.title():<15} : {s:>6.2f}") for t, s in p["personality_traits"].items()],
                    print(f"\n📊 Stats: {p['total_events']} events, {p['open_commitments']} commitments"),
                ))(pmm_memory.get_personality_summary())
            ),
            "memory": lambda: (
                (lambda ctx: (
                    print("\n🧠 Cross-Session Memory Context:"),
                    print(ctx[:500] if ctx else "No cross-session memory yet"),
                ))(pmm_memory.load_memory_variables({}).get("history", ""))
            ),
            "status": lambda: (
                (lambda counts, facts, rel, total: (
                    print("\n📊 Status:"),
                    print(f"   • total_events: {total}"),
                    (print("   • counts_by_etype:") or [print(f"     - {et}: {c}") for et, c in counts]) if counts else None,
                    print(f"   • sample_facts: {len(facts)} | sample_related: {len(rel)}"),
                ))(
                    (pmm_memory.pmm.sqlite_store.counts_by_etype() if getattr(pmm_memory.pmm, "sqlite_store", None) else []),
                    _load_identity_facts(5),
                    _semantic_matches_keyword("status", 3),
                    len(getattr(pmm_memory.pmm.model.self_knowledge, "autobiographical_events", []) or []),
                )
            ),
            "dump": lambda: (
                (lambda rows: (
                    print("\n🗃️  Last 5 events:"),
                    (print("   (none)") if not rows else [
                        (lambda rid, et, summ, has_emb: print(f"   • {rid:>5} | {(et or '(null)'):<16} | {((summ or '').replace('\n',' ')[:60]):<60} | embed={'✓' if has_emb else '✗'}"))(*r)
                        for r in rows
                    ])
                ))(
                    (pmm_memory.pmm.sqlite_store.conn.execute(
                        "SELECT id,etype,summary,embedding IS NOT NULL FROM events ORDER BY id DESC LIMIT 5"
                    ).fetchall() if getattr(pmm_memory.pmm, "sqlite_store", None) else [])
                )
            ),
            "evolution": _handle_evolution,
        }

    # Seed defaults once, then load registry and actions
    _seed_default_commands()
    _command_registry = _load_command_registry()
    _actions = _make_action_map()

    # --- PMM provenance facts helper (minimal, local-only) ---
    import json as _json

    def _pmm_facts_block(pmm_memory, k: int = 5) -> str:
        """
        Return a small provenance block with explicit identity facts.
        Works with current 7-column schema by reading meta['type'].
        """
        try:
            store = getattr(pmm_memory.pmm, "sqlite_store", None)
            if not store:
                return ""
            rows = store.recent_events(limit=200)  # (id, ts, kind, content, meta, prev, hash)
            rows = list(reversed(rows))            # chronological
            facts = []
            for (eid, ts, kind, content, meta, _prev, _hsh) in rows:
                try:
                    m = _json.loads(meta) if isinstance(meta, str) else (meta or {})
                except Exception:
                    m = {}
                etype = m.get("type") or m.get("etype")  # meta-based typing
                if etype == "identity_info":
                    s = (content or "")[:160]
                    facts.append(f"- E{eid} ({ts}): {s}")
            if not facts:
                return ""
            return "[FACTS]\n" + "\n".join(facts[-k:]) + "\n"
        except Exception:
            return ""

    # --- PMM networking + indexing helpers (stdlib only) ---
    import os as _os, re as _re, sys as _sys, io, time as _time, json as __json, math as _math, hashlib as _hashlib, subprocess as _subp, urllib.parse as _urlp, urllib.request as _urlreq
    from html.parser import HTMLParser as _HTMLParser

    ALLOW = set(filter(None, _os.getenv("PMM_NET_ALLOW", "github.com,raw.githubusercontent.com,readthedocs.io,docs.python.org").split(",")))
    MAX_MB = float(_os.getenv("PMM_NET_MAX_MB", "5"))
    TIMEOUT = int(_os.getenv("PMM_NET_TIMEOUT_S", "15"))

    def _sha256_bytes(b: bytes) -> str:
        return _hashlib.sha256(b).hexdigest()

    class _TextExtractor(_HTMLParser):
        def __init__(self):
            super().__init__(); self._buf=[]; self._skip=False
        def handle_starttag(self, tag, attrs): self._skip = tag in ("script","style")
        def handle_endtag(self, tag): self._skip = False
        def handle_data(self, data):
            if not self._skip: self._buf.append(data)
        def text(self):
            import re as __re
            return __re.sub(r"[ \t]+\n", "\n", __re.sub(r"[ \t]+", " ", "".join(self._buf))).strip()

    def _domain_ok(url:str)->bool:
        try:
            host = _urlp.urlparse(url).hostname
            return host in ALLOW
        except Exception:
            return False

    def _http_get_text(url:str) -> tuple[str,str]:
        """Return (text, sha256) from URL; enforce domain allowlist, size & timeout; HTML->text."""
        if not _domain_ok(url):
            raise ValueError(f"domain not allowed: {url}")
        req = _urlreq.Request(url, headers={"User-Agent":"PMM/1.0"})
        with _urlreq.urlopen(req, timeout=TIMEOUT) as r:
            ct = r.headers.get("Content-Type","\n").lower()
            if ("text" not in ct) and ("json" not in ct) and ("markdown" not in ct) and ("html" not in ct):
                raise ValueError(f"unsupported content-type: {ct}")
            max_bytes = int(MAX_MB*1024*1024)
            data = r.read(max_bytes+1)
            if len(data) > max_bytes:
                raise ValueError("document too large")
        if "html" in ct:
            p = _TextExtractor(); p.feed(data.decode(errors="replace")); txt = p.text()
        else:
            txt = data.decode(errors="replace")
        return txt, _sha256_bytes(data)

    def _chunk_lines(text:str, path:str, max_lines:int=200):
        """Chunk text with safer boundaries to avoid mid-function splits.
        Tries to end chunks at blank lines or just before a def/class.
        """
        lines = text.splitlines()
        i = 0
        n = len(lines)
        while i < n:
            s = i + 1
            e = min(i + max_lines, n)
            # back off to a safer boundary if possible
            if e < n:
                j = e
                # prefer a blank line boundary
                while j > i and lines[j-1].strip() and not lines[j-1].lstrip().startswith(("def ", "class ")):
                    j -= 1
                if j <= i + 5:  # too close to start; keep original e
                    j = e
                e = j
            snippet = "\n".join(lines[s-1:e])
            if not snippet.strip():
                # fall back to original window to avoid empty chunks
                e = min(i + max_lines, n)
                snippet = "\n".join(lines[s-1:e])
            yield s, e, snippet
            i = e

    _CODE_EXT = {".py",".md",".json",".yml",".yaml",".toml",".ts",".tsx",".js",".jsx",".css",".txt"}
    def _iter_code_files(root:str):
        for dirpath, dirnames, filenames in _os.walk(root):
            if any(x in dirpath for x in (".git","node_modules","venv",".venv","__pycache__")):
                continue
            for fn in filenames:
                if _os.path.splitext(fn)[1].lower() in _CODE_EXT:
                    path = _os.path.join(dirpath, fn)
                    try:
                        if _os.path.getsize(path) <= MAX_MB*1024*1024:
                            yield path
                    except Exception:
                        continue

    # --- Enhanced PMM self-index with mtime optimization (local codebase) ---

    def _iter_local_code_files(root: str):
        for dirpath, dirnames, filenames in _os.walk(root):
            # ignore typical junk
            if any(
                x in dirpath
                for x in (
                    ".git",
                    "node_modules",
                    "venv",
                    ".venv",
                    "__pycache__",
                    ".mypy_cache",
                    ".pytest_cache",
                )
            ):
                continue
            for fn in filenames:
                ext = _os.path.splitext(fn)[1].lower()
                if ext in CODE_EXT:
                    path = _os.path.join(dirpath, fn)
                    try:
                        if _os.path.getsize(path) <= CODE_MAX_MB * 1024 * 1024:
                            yield path
                    except FileNotFoundError:
                        continue

    def _load_manifest():
        """Load manifest with backward compatibility for old format."""
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                data = __json.load(f)

            # Handle old format: {relpath: sha256_string}
            # New format: {relpath: {"sha256": "...", "mtime": 123.45, "size": 1024}}
            if data and isinstance(list(data.values())[0], str):
                _log("info", "upgrading manifest format (old->new)")
                return {}  # Force full re-index on format upgrade

            return data
        except Exception:
            return {}

    def _save_manifest(m):
        try:
            with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
                __json.dump(m, f, indent=2)
        except Exception as e:
            _log("warn", f"failed to save manifest: {e}")

    def _get_file_info(path: str) -> dict:
        """Get file metadata for change detection."""
        try:
            stat = _os.stat(path)
            return {"mtime": stat.st_mtime, "size": stat.st_size}
        except Exception:
            return {"mtime": 0, "size": 0}

    def _file_likely_unchanged(path: str, prev_entry: dict) -> bool:
        """Fast heuristic: if mtime and size are same, file probably unchanged."""
        current = _get_file_info(path)
        return (
            current["mtime"] == prev_entry.get("mtime", 0)
            and current["size"] == prev_entry.get("size", 0)
        )

    def _index_own_codebase(pmm_memory, root: str):
        """Scan local code, index only new/changed files as code_chunk events."""
        root = _os.path.abspath(root)
        prev = _load_manifest()  # {relpath: {"sha256": "...", "mtime": 123.45, "size": 1024}}
        curr = {}
        added_chunks = 0
        changed_files = 0
        skipped_files = 0
        hash_checks = 0  # files that needed hash verification

        for path in _iter_local_code_files(root):
            rel = os.path.relpath(path, root)
            file_info = _get_file_info(path)
            prev_entry = prev.get(rel, {})

            # Fast path: if mtime/size unchanged, assume file is unchanged
            if _file_likely_unchanged(path, prev_entry):
                # Copy forward the previous manifest entry
                curr[rel] = prev_entry
                skipped_files += 1
                continue

            # Slow path: read file and compute hash to be sure
            hash_checks += 1
            try:
                data = open(path, "rb").read()
            except Exception as e:
                _log("warn", f"skip {rel}: {e}")
                continue

            sha = _sha256_bytes(data)

            # Update manifest entry with new info
            curr[rel] = {"sha256": sha, "mtime": file_info["mtime"], "size": file_info["size"]}

            # Check if content actually changed (mtime can change without content changes)
            if prev_entry.get("sha256") == sha:
                # Content unchanged despite mtime difference (e.g., touch, git checkout)
                continue

            # Content changed - index it
            changed_files += 1
            text = data.decode(errors="replace")
            for s, e, snip in _chunk_lines(text, rel, max_lines=200):
                pmm_memory.pmm.add_event(
                    summary=f"CODE: {rel} [{s}-{e}]\n{snip[:1000]}",
                    effects=[],
                    etype="code_chunk",
                    full_text=None,
                    tags=["code", "chunk", rel],
                )
                added_chunks += 1

        # Clean up manifest: remove entries for files that no longer exist
        all_current_files = set(_os.path.relpath(p, root) for p in _iter_local_code_files(root))
        removed_files = set(prev.keys()) - all_current_files
        if removed_files:
            _log("info", f"removed {len(removed_files)} files from manifest")

        _save_manifest(curr)
        _log("info", f"self-index: {changed_files} changed files -> {added_chunks} chunks")
        _log("info", f"performance: {skipped_files} skipped, {hash_checks} hash checks (root={root})")

    def _cleanup_old_code_chunks(pmm_memory, keep_days=7):
        """Optional: remove old code_chunk events to prevent DB bloat."""
        try:
            # Placeholder: requires sqlite_store support to delete by etype+time
            # pmm_memory.pmm.sqlite_store.delete_events_before(cutoff_ts, etype="code_chunk")
            pass
        except Exception as e:
            _log("warn", f"cleanup failed: {e}")

    def _validate_manifest():
        """Debug helper: check if manifest entries match actual files."""
        manifest = _load_manifest()
        issues = []
        for rel_path, entry in manifest.items():
            if not _os.path.exists(_os.path.join(CODE_ROOT, rel_path)):
                issues.append(f"missing: {rel_path}")
            elif not isinstance(entry, dict) or "sha256" not in entry:
                issues.append(f"bad format: {rel_path}")
        if issues:
            _log("warn", f"manifest issues: {issues}")
        else:
            _log("info", "manifest validated OK")

    # Now that helper functions are defined, perform self-index
    try:
        if (CODE_INDEX_MODE or "").lower() != "off":
            _index_own_codebase(pmm_memory, CODE_ROOT)
            # Optional periodic cleanup example:
            # _cleanup_old_code_chunks(pmm_memory, keep_days=7)
    except Exception as e:
        _log("warn", f"self-index skipped: {e}")

    # --- Background: periodic code self-reflection ---------------------------------
    def _start_code_reflection_thread(pmm_memory, interval_seconds: int = 180):
        """Start a daemon thread that periodically reflects on top code topics.
        Creates small 'reflection' events with summaries to enrich memory.
        """
        # Guard against multiple starts
        if getattr(pmm_memory, "_code_reflect_thread_started", False):
            return

        def _runner():
            _log("info", "code reflection thread started")
            topics = [
                "_index_own_codebase",
                "get_pmm_system_prompt",
                "sqlite_store",
                "self_model_manager",
                "reflection",
                "commitment",
            ]
            while True:
                try:
                    for q in topics:
                        _log("info", f"reflecting on topic: {q}")
                        expl = _explain_code_snippet(pmm_memory, q, top=1)
                        if expl and "no relevant" not in expl.lower():
                            preview = expl.strip()[:1000]
                            try:
                                pmm_memory.pmm.add_event(
                                    summary=f"REFLECT(code): {q}\n{preview}",
                                    effects=[],
                                    etype="reflection",
                                    full_text=None,
                                    tags=["code", "reflect", q],
                                )
                                _log("info", f"added reflection event for {q}")
                            except Exception:
                                pass
                    _time.sleep(interval_seconds)
                except Exception:
                    # Never die; wait a bit and continue
                    _time.sleep(interval_seconds)

        t = threading.Thread(target=_runner, name="pmm_code_reflect", daemon=True)
        t.start()
        setattr(pmm_memory, "_code_reflect_thread_started", True)

    # --- Background: periodic self-reflection (identity, commitments, traits) -------
    def _start_self_reflection_thread(pmm_memory, interval_seconds: int = 300):
        """Start a daemon thread that periodically introspects identity, commitments, and Big Five.
        Produces compact 'reflection' events to enrich autobiographical memory.
        """
        # Guard against multiple starts
        if getattr(pmm_memory, "_self_reflect_thread_started", False):
            return

        def _runner():
            _log("info", "self reflection thread started")
            # Keep last Big Five snapshot in-memory for delta computation
            last_snapshot = getattr(pmm_memory, "_last_big5_snapshot", None)
            while True:
                try:
                    # Identity basics
                    try:
                        name = (pmm_memory.pmm.model.core_identity.name or "").strip()
                    except Exception:
                        name = ""

                    # Recent identity events (if any)
                    id_notes = []
                    try:
                        rows = pmm_memory.pmm.sqlite_store.recent_by_etype("identity_change", limit=5)
                        for (_eid, ts, et, summary, content, meta) in rows:
                            if summary:
                                id_notes.append(f"- {ts}: {summary}")
                    except Exception:
                        pass

                    # Commitments snapshot (up to 5)
                    commits_preview = []
                    try:
                        opens = pmm_memory.pmm.get_open_commitments()
                        for c in (opens or [])[:5]:
                            txt = c.get("text") or c.get("title") or "(no text)"
                            status = c.get("status", "open")
                            due = c.get("due")
                            tail = f" (due {due})" if due else ""
                            commits_preview.append(f"- [{status}] {txt}{tail}")
                    except Exception:
                        pass

                    # Big Five snapshot and deltas
                    deltas = []
                    try:
                        big5 = pmm_memory.pmm.get_big5() or {}
                    except Exception:
                        big5 = {}
                    if big5:
                        if last_snapshot is None:
                            last_snapshot = dict(big5)
                        else:
                            for k in ("openness","conscientiousness","extraversion","agreeableness","neuroticism"):
                                try:
                                    prev = float(last_snapshot.get(k, 0.0))
                                    cur = float(big5.get(k, prev))
                                except Exception:
                                    prev, cur = 0.0, 0.0
                                delta = cur - prev
                                if abs(delta) >= 0.01:
                                    arrow = "↑" if delta > 0 else "↓"
                                    deltas.append(f"{k}: {prev:.3f} {arrow} {cur:.3f} ({delta:+.3f})")
                            last_snapshot = dict(big5)
                        # persist snapshot on the memory object for future cycles
                        try:
                            setattr(pmm_memory, "_last_big5_snapshot", dict(last_snapshot))
                        except Exception:
                            pass

                    # Compose reflection summary
                    lines = ["REFLECT(self): periodic self-introspection"]
                    if name:
                        lines.append(f"identity: name={name}")
                    if id_notes:
                        lines.append("recent identity events:")
                        lines.extend(id_notes)
                    if commits_preview:
                        lines.append("open commitments (up to 5):")
                        lines.extend(commits_preview)
                    if big5:
                        lines.append("big5 snapshot:")
                        lines.append(
                            "  "
                            + ", ".join(
                                f"{k}={float(big5.get(k, 0.0)):.3f}"
                                for k in ("openness","conscientiousness","extraversion","agreeableness","neuroticism")
                            )
                        )
                    if deltas:
                        lines.append("trait changes (|Δ|>=0.01):")
                        for d in deltas:
                            lines.append("  " + d)

                    summary = "\n".join(lines)
                    try:
                        pmm_memory.pmm.add_event(
                            summary=summary,
                            effects=[],
                            etype="reflection",
                            full_text=None,
                            tags=["self", "reflection"],
                        )
                        _log("info", "added self-reflection event")
                    except Exception:
                        pass

                    # Optional short self-report narrative
                    if deltas:
                        try:
                            narr = "I feel " + "; ".join(
                                d.split(":",1)[0] + (" more" if "+" in d else " less")
                                for d in deltas
                            )
                            pmm_memory.pmm.add_event(
                                summary=f"Self-report: {narr}",
                                effects=[],
                                etype="reflection",
                                full_text=None,
                                tags=["self", "reflection", "personality"],
                            )
                        except Exception:
                            pass

                    _time.sleep(interval_seconds)
                except Exception:
                    # Never die; wait a bit and continue
                    _time.sleep(interval_seconds)

        t = threading.Thread(target=_runner, name="pmm_self_reflect", daemon=True)
        t.start()
        setattr(pmm_memory, "_self_reflect_thread_started", True)

    # kick off background reflection (non-blocking, safe no-op if already started)
    try:
        _start_code_reflection_thread(pmm_memory, interval_seconds=180)
    except Exception as _e:
        _log("warn", f"code reflect thread not started: {_e}")
    # kick off self-reflection thread (non-blocking)
    try:
        _start_self_reflection_thread(pmm_memory, interval_seconds=300)
    except Exception as _e:
        _log("warn", f"self reflect thread not started: {_e}")

    # --- Ranked code/doc context blocks (keyword only, local) ---
    def _rank_events(pmm_memory, query: str, kinds=("code_chunk", "web_doc"), limit_scan=1000, top=3):
        try:
            rows = pmm_memory.pmm.sqlite_store.recent_events(limit=limit_scan)
        except Exception:
            rows = []
        scored = []
        import re as __re
        toks = set(w for w in __re.findall(r"\w+", (query or "").lower()) if len(w) > 2)
        if not toks:
            return []
        for (eid, ts, kind, content, meta, prev, hsh) in rows:
            try:
                m = __json.loads(meta) if isinstance(meta, str) else (meta or {})
            except Exception:
                m = {}
            t = (m.get("type") or m.get("etype") or "").lower()
            if kinds and t not in kinds:
                continue
            tags = m.get("tags") or []
            # derive src from explicit path/url or tags
            src = m.get("path") or m.get("url") or ""
            if not src and isinstance(tags, list):
                for tg in tags:
                    if isinstance(tg, str) and tg not in ("code", "chunk", "web", "doc") and ("/" in tg or "." in tg):
                        src = tg
                        break
            hay_parts = [content or "", src]
            if isinstance(tags, list):
                hay_parts.extend([str(tg) for tg in tags])
            hay = " ".join(hay_parts).lower()
            score = sum(1 for w in toks if w in hay)
            if score:
                scored.append((score, src or "(unknown)", eid, ts, (content or "")))
        scored.sort(reverse=True)
        return scored[:top]

    # Explain code by parsing top-matched code_chunk events
    def _explain_code_snippet(pmm_memory, query: str, top: int = 3) -> str:
        import ast as _ast
        import re as _re
        items = _rank_events(pmm_memory, query, kinds=("code_chunk",), top=top)
        if not items:
            return "- (no relevant code snippets found)\n"
        out = ["[CODE EXPLANATION]"]
        for _, src, eid, ts, content in items:
            # Skip non-Python sources early (e.g., .json, .md) to avoid AST parse noise
            try:
                if not (src or "").lower().endswith(".py"):
                    out.append(f"- E{eid} ({src}): skipped (non-Python file)")
                    continue
            except Exception:
                pass
            try:
                # limit to reduce chance of partial AST failure
                lines = (content or "").splitlines()
                limited = "\n".join(lines[:200])
                tree = _ast.parse(limited)
                func_names = [n.name for n in _ast.walk(tree) if isinstance(n, _ast.FunctionDef)]
                class_names = [n.name for n in _ast.walk(tree) if isinstance(n, _ast.ClassDef)]
                # docstring sniff for first function/class
                doc = None
                for n in _ast.walk(tree):
                    if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)):
                        ds = _ast.get_docstring(n)
                        if ds:
                            doc = ds.strip()[:120]
                            break
                details = []
                if func_names:
                    details.append(f"functions: {', '.join(func_names)}")
                if class_names:
                    details.append(f"classes: {', '.join(class_names)}")
                if doc:
                    details.append(f"doc: {doc}...")
                if not details:
                    details.append("no top-level functions/classes detected")
                out.append(f"- E{eid} ({src}): " + "; ".join(details))
            except SyntaxError:
                # regex fallback to salvage names from partial chunks
                funcs = [m.group(1) for m in _re.finditer(r"^\s*def\s+(\w+)\s*\(", content or "", _re.M)]
                classes = [m.group(1) for m in _re.finditer(r"^\s*class\s+(\w+)\s*\(", content or "", _re.M)]
                if funcs or classes:
                    bits = []
                    if funcs:
                        bits.append(f"functions: {', '.join(funcs)}")
                    if classes:
                        bits.append(f"classes: {', '.join(classes)}")
                    out.append(f"- E{eid} ({src}): partial parse; " + "; ".join(bits))
                else:
                    out.append(f"- E{eid} ({src}): unable to parse (syntax error)")
            except Exception as e:
                out.append(f"- E{eid} ({src}): analysis error: {e}")
        return "\n".join(out) + "\n"

    def _code_context_block(pmm_memory, q: str, top=5, budget=1000):
        items = _rank_events(pmm_memory, q, kinds=("code_chunk",), top=top)
        if not items:
            return ""
        out = ["[CODE CONTEXT]"]
        used = 0
        for _, src, eid, ts, content in items:
            import re as _re
            first = (content.splitlines()[0] if content else "")
            # Try to recover src and [s-e] range from header
            m_hdr = _re.match(r"^CODE:\s+([^\[]+)\s*\[(\d+)-(\d+)\]", first)
            if (not src) and m_hdr:
                src = m_hdr.group(1).strip()
            line_info = f"[{m_hdr.group(2)}-{m_hdr.group(3)}]" if m_hdr else ""
            preview = first[:140]
            entry = f"- E{eid} ({src} {line_info}). {preview}"
            if used + len(entry) > budget:
                break
            out.append(entry)
            used += len(entry)
        return "\n".join(out) + "\n"

    def _doc_context_block(pmm_memory, q: str, top=3, budget=500):
        items = _rank_events(pmm_memory, q, kinds=("web_doc",), top=top)
        if not items:
            return ""
        out = ["[DOC CONTEXT]"]
        used = 0
        for _, src, eid, ts, content in items:
            line1 = (content.splitlines()[0] if content else "")[:140]
            entry = f"- E{eid} ({src}): {line1}"
            if used + len(entry) > budget:
                break
            out.append(entry)
            used += len(entry)
        return "\n".join(out) + "\n"

    print(f"\n🤖 PMM is ready! Using {model_name} ({model_config.provider})")
    print(
        "💡 Commands: 'quit' to exit, 'personality' for traits, 'memory' for context, 'models' to switch, 'status' for counts, 'dump' for last events, 'selftest'"
    )
    print("Start chatting...")

    # Initialize conversation history with PMM system prompt
    conversation_history = [{"role": "system", "content": get_pmm_system_prompt()}]

    def invoke_model(messages):
        """Invoke model with proper format based on provider type."""
        current_config = get_model_config(model_name)  # Get current model config
        if current_config.provider == "ollama":
            # Ollama expects a single string, so format the conversation
            formatted_prompt = ""
            for msg in messages:
                if msg["role"] == "system":
                    formatted_prompt += f"System: {msg['content']}\n\n"
                elif msg["role"] == "user":
                    formatted_prompt += f"User: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    formatted_prompt += f"Assistant: {msg['content']}\n"
            formatted_prompt += "Assistant: "
            return llm.invoke(formatted_prompt)
        else:
            # OpenAI chat models expect message list
            return llm.invoke(messages)

    # Setup for potentially mixed input modes
    stdin_is_pipe = not sys.stdin.isatty()
    tty_file = None
    if stdin_is_pipe and not args.noninteractive:
        try:
            tty_file = open("/dev/tty", "r")
            print(
                "🎯 Piped input detected. After consuming piped messages, will switch to keyboard input."
            )
        except Exception:
            print("🎯 Piped input detected. Running in non-interactive mode.")
            tty_file = None

    def get_user_input():
        """Get user input from appropriate source."""
        if tty_file:
            # Set prompt reprinter for background logs
            def _rp():
                sys.stdout.write("\n👤 You: ")
                sys.stdout.flush()
            globals()["_prompt_reprinter"] = _rp
            print("\n👤 You: ", end="", flush=True)
            return tty_file.readline().strip()
        # stdin prompt path
        def _rp():
            sys.stdout.write("\n👤 You: ")
            sys.stdout.flush()
        globals()["_prompt_reprinter"] = _rp
        return input("\n👤 You: ").strip()

    while True:
        try:
            # Get user input
            user_input = get_user_input()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ["quit", "exit", "bye"]:
                print("👋 Goodbye! Your conversation is saved with persistent memory.")
                break
            # Natural language help
            if user_input.strip().lower() in ("help", "commands") or (
                any(kw in user_input.strip().lower() for kw in ("what can you do", "what are your commands", "how do i use you"))
            ):
                _print_command_list(_load_command_registry())
                continue
            # Dispatch dynamic commands if present
            if user_input.strip().lower() in _load_command_registry().keys():
                cmd = user_input.strip().lower()
                action = _actions.get(cmd)
                if action:
                    try:
                        action()
                    except Exception as e:
                        print(f"❌ Command '{cmd}' failed: {e}")
                else:
                    print(f"ℹ️ Command '{cmd}' is known but has no bound action.")
                continue
            elif user_input.lower() == "personality":
                personality = pmm_memory.get_personality_summary()
                print("\n🎭 Current Personality State:")
                for trait, score in personality["personality_traits"].items():
                    print(f"   • {trait.title():<15} : {score:>6.2f}")
                print(
                    f"\n📊 Stats: {personality['total_events']} events, {personality['open_commitments']} commitments"
                )
                continue
            elif user_input.lower() == "memory":
                pmm_context = pmm_memory.load_memory_variables({}).get("history", "")
                print("\n🧠 Cross-Session Memory Context:")
                print(
                    pmm_context[:500] if pmm_context else "No cross-session memory yet"
                )
                continue
            elif user_input.lower() == "status":
                try:
                    total = len(pmm_memory.pmm.model.self_knowledge.autobiographical_events)
                except Exception:
                    total = -1
                counts = []
                try:
                    counts = pmm_memory.pmm.sqlite_store.counts_by_etype()
                except Exception:
                    counts = []
                facts = _load_identity_facts(5)
                rel = _semantic_matches_keyword("test", 3)
                print("\n📊 Status:")
                print(f"   • total_events: {total}")
                if counts:
                    print("   • counts_by_etype:")
                    for et, c in counts:
                        print(f"     - {et}: {c}")
                print(f"   • sample_facts: {len(facts)} | sample_related: {len(rel)}")
                continue
            elif user_input.lower() == "dump":
                try:
                    rows = pmm_memory.pmm.sqlite_store.conn.execute(
                        "SELECT id,etype,summary,embedding IS NOT NULL FROM events ORDER BY id DESC LIMIT 5"
                    ).fetchall()
                except Exception as _e:
                    rows = []
                print("\n🗃️  Last 5 events:")
                if not rows:
                    print("   (none)")
                else:
                    for rid, et, summ, has_emb in rows:
                        flag = "✓" if has_emb else "✗"
                        prev = (summ or "").replace("\n", " ")[:60]
                        print(f"   • {rid:>5} | {et or '(null)':<16} | {prev:<60} | embed={flag}")
                continue
            elif user_input.lower() == "selftest":
                try:
                    # minimal self-test of identity continuity without embeddings
                    test_name = "Alice"
                    pmm_memory.save_context({"input": "Hello."}, {"response": "Hi!"})
                    pmm_memory.save_context({"input": f"My name is {test_name}."}, {"response": "Nice to meet you"})
                    sys_prompt = get_pmm_system_prompt("What's my name?")
                    ok = (f"{test_name}" in sys_prompt)
                    print(f"\n🧪 Self-test: {'PASS' if ok else 'FAIL'}")
                except Exception as _e:
                    print(f"\n🧪 Self-test: FAIL ({_e})")
                continue
            elif user_input.lower().startswith("repo sync "):
                target = user_input.split(" ", 2)[2].strip()
                base_dir = os.path.abspath("pmm_sources")
                os.makedirs(base_dir, exist_ok=True)
                local_root = None
                if target.startswith("http"):
                    import re as _re
                    name = _re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(_urlp.urlparse(target).path)) or "repo"
                    local_root = os.path.join(base_dir, name)
                    try:
                        if not os.path.exists(os.path.join(local_root, ".git")):
                            _subp.run(["git", "clone", "--depth", "1", target, local_root], check=True, timeout=60)
                        else:
                            _subp.run(["git", "-C", local_root, "pull", "--ff-only"], check=True, timeout=60)
                        print(f"repo ready: {local_root}")
                    except Exception as e:
                        print(f"repo sync failed: {e}")
                        continue
                else:
                    if os.path.isdir(target):
                        local_root = os.path.abspath(target)
                    else:
                        print("path not found")
                        continue

                added = 0
                for path in _iter_code_files(local_root):
                    rel = os.path.relpath(path, local_root)
                    try:
                        data = open(path, "rb").read()
                        text = data.decode(errors="replace")
                        for s, e, snip in _chunk_lines(text, rel, max_lines=200):
                            pmm_memory.pmm.add_event(
                                summary=f"CODE: {rel} [{s}-{e}]\n{snip[:1000]}",
                                effects=[],
                                etype="code_chunk",
                                full_text=None,
                                tags=["code", "chunk", rel]
                            )
                            added += 1
                    except Exception as e:
                        print(f"skip {path}: {e}")
                print(f"Indexed {added} code chunks from {local_root}")
                continue
            elif user_input.lower().startswith("net add "):
                url = user_input.split(" ", 2)[2].strip()
                try:
                    txt, sha = _http_get_text(url)
                    title = (txt.splitlines()[0] if txt.splitlines() else url)[:120]
                    pmm_memory.pmm.add_event(
                        summary=f"WEB: {title}\n{txt[:1500]}",
                        effects=[],
                        etype="web_doc",
                        full_text=None,
                        tags=["web", "doc", url]
                    )
                    print(f"Fetched & stored: {url} (sha256={sha[:12]}...)")
                except Exception as e:
                    print(f"net add failed: {e}")
                continue
            elif user_input.lower().startswith("doc find "):
                qraw = user_input.split(" ", 2)[2].strip().lower()
                import re as __re
                q = set(t for t in __re.findall(r"\w+", qraw) if len(t) > 2)
                rows = pmm_memory.pmm.sqlite_store.recent_events(limit=1000)
                hits = []
                for (eid, ts, kind, content, meta, prev, hsh) in rows:
                    try:
                        m = __json.loads(meta) if isinstance(meta, str) else (meta or {})
                    except Exception:
                        m = {}
                    t = (m.get("type") or m.get("etype") or "").lower()
                    if t not in ("code_chunk", "web_doc"):
                        continue
                    src = m.get("path") or m.get("url") or "(unknown)"
                    hay = " ".join([(content or ""), src]).lower()
                    score = sum(1 for w in q if w in hay)
                    if score:
                        preview = (content or "").splitlines()[0][:80]
                        hits.append((score, eid, src, preview))
                hits.sort(reverse=True)
                for score, eid, src, preview in hits[:10]:
                    print(f"E{eid} [{score}] | {src} | {preview}")
                if not hits:
                    print("no matches")
                continue
            elif user_input.lower().startswith("code reflect"):
                # Manual trigger for code explanation on a query
                parts = user_input.split(" ", 2)
                query = parts[2].strip() if len(parts) > 2 else "code"
                explanation = _explain_code_snippet(pmm_memory, query, top=3)
                print("\n" + (explanation or "- (none)").rstrip() + "\n")
                conversation_history.append({"role": "assistant", "content": explanation})
                continue
            elif user_input.lower() == "models":
                print("\n" + "=" * 50)
                # For piped sessions, allow inline model selection
                if stdin_is_pipe:
                    print("🎯 Select a model by typing the number:")
                    available_models = list_available_models()
                    for i, model in enumerate(available_models, 1):
                        marker = "⭐" if model == model_name else f"{i:2d}."
                        config = get_model_config(model)
                        cost_str = (
                            f"${config.cost_per_1k_tokens:.4f}/1K"
                            if config.cost_per_1k_tokens > 0
                            else "Free"
                        )
                        print(f"{marker} {model} ({config.provider}) - {cost_str}")
                    print(
                        f"\n💡 Type a number (1-{len(available_models)}) or press ENTER for current model"
                    )

                    # Get next input for model selection
                    try:
                        model_choice = get_user_input().strip()
                        if not model_choice:
                            new_model = model_name  # Keep current
                            print(f"✅ Keeping current model: {model_name}")
                        elif model_choice.isdigit():
                            idx = int(model_choice)
                            if 1 <= idx <= len(available_models):
                                new_model = available_models[idx - 1]
                                print(f"✅ Selected model {idx}: {new_model}")
                            else:
                                print(
                                    f"❌ Invalid number. Please choose 1-{len(available_models)}"
                                )
                                new_model = None
                        else:
                            print(f"❌ Please enter a number 1-{len(available_models)}")
                            new_model = None
                    except Exception as e:
                        print(f"❌ Error reading model choice: {e}")
                        new_model = None
                else:
                    new_model = show_model_selection(force_tty=not args.noninteractive)

                if new_model and new_model != model_name:
                    print(f"🔄 Switching to {new_model}... Please wait...")

                    # Update model configuration
                    model_name = new_model
                    model_config = get_model_config(model_name)

                    # Recreate LLM with new model based on provider
                    if model_config.provider == "ollama":
                        llm = OllamaLLM(model=model_config.name, temperature=0.7)
                    else:  # openai
                        llm = ChatOpenAI(model=model_config.name, temperature=0.7)

                    # Refresh conversation history with updated system prompt
                    conversation_history[0] = {
                        "role": "system",
                        "content": get_pmm_system_prompt(),
                    }

                    print(
                        f"✅ Successfully switched to {model_name} ({model_config.provider})"
                    )
                    print(f"🔧 Max tokens: {model_config.max_tokens:,}")
                    if model_config.cost_per_1k_tokens > 0:
                        print(
                            f"💰 Cost: ${model_config.cost_per_1k_tokens:.4f}/1K tokens"
                        )
                    else:
                        print("💰 Cost: Free (local model)")
                    print("🧠 PMM context refreshed for new model")
                elif new_model == model_name:
                    print(f"✅ Already using {model_name}")
                else:
                    print("❌ Model selection cancelled")
                print("=" * 50 + "\n")
                continue

            # Add user input to conversation history
            conversation_history.append({"role": "user", "content": user_input})

            # Inject retrieval-augmented PMM Context before each response
            conversation_history[0] = {
                "role": "system",
                "content": (
                    get_pmm_system_prompt(user_input)
                    + ("\n" + (_pmm_facts_block(pmm_memory, k=5) or ""))
                    + (_code_context_block(pmm_memory, user_input, top=3) or "")
                    + (_doc_context_block(pmm_memory, user_input, top=3) or "")
                ),
            }

            # Show API call info
            current_config = get_model_config(model_name)
            provider_name = current_config.provider.upper()
            print(
                f"🤖 PMM: [API] Calling {provider_name} with prompt: {user_input[:50]}..."
            )
            response = invoke_model(conversation_history)

            # Handle response format differences
            if current_config.provider == "ollama":
                response_text = response  # Ollama returns string directly
            else:
                response_text = response.content  # OpenAI returns message object

            print(f"[API] Response received: {len(response_text)} chars")
            print(response_text)

            # Add AI response to conversation history
            conversation_history.append({"role": "assistant", "content": response_text})

            # Save to PMM memory system (async to avoid UI stalls)
            def _persist_context(u: str, r: str):
                try:
                    pmm_memory.save_context({"input": u}, {"response": r})
                except Exception as _e:
                    print(f"[warn] save_context failed: {_e}")

            threading.Thread(target=_persist_context, args=(user_input, response_text), daemon=True).start()

        except KeyboardInterrupt:
            print("\n\n👋 Chat interrupted. Your conversation is saved!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Continuing chat...")

    # Clean up
    if tty_file:
        tty_file.close()


if __name__ == "__main__":
    main()
