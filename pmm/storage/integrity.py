import hashlib
from typing import Optional

def sha256_hex(text: str) -> str:
    """Generate SHA256 hex hash of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def make_linked_hash(prev_hash: Optional[str], kind: str, content: str, meta_json: str) -> str:
    """Create hash linking to previous event in chain."""
    base = (prev_hash or "") + "|" + kind + "|" + content + "|" + meta_json
    return sha256_hex(base)

def verify_chain(rows) -> bool:
    """Verify hash-chain integrity across all events."""
    prev = None
    for _, _, kind, content, meta, prev_hash, hsh in rows:
        if prev_hash != prev:
            return False
        expect = make_linked_hash(prev, kind, content, meta)
        if expect != hsh:
            return False
        prev = hsh
    return True
