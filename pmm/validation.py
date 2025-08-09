from __future__ import annotations
import json
from dataclasses import asdict
from typing import Any, Optional

try:
    from jsonschema import Draft7Validator
except Exception:  # jsonschema not installed yet; make validation a no-op
    Draft7Validator = None  # type: ignore


class SchemaValidator:
    """Wrap jsonschema validation. If schema file or jsonschema is missing, validation is skipped.
    """

    def __init__(self, schema_path: str = "schema/pmm.schema.json"):
        self._enabled = False
        self._validator: Optional[Any] = None
        if Draft7Validator is None:
            return
        try:
            with open(schema_path, "r") as f:
                schema = json.load(f)
            self._validator = Draft7Validator(schema)
            self._enabled = True
        except FileNotFoundError:
            # Schema not present yet: run permissive
            self._enabled = False
        except Exception:
            # Any schema load/parse error disables validation (fail open)
            self._enabled = False

    def validate_dict(self, payload: dict) -> None:
        if not self._enabled or self._validator is None:
            return
        errors = sorted(self._validator.iter_errors(payload), key=lambda e: e.path)
        if errors:
            lines = []
            for err in errors[:10]:  # cap noise
                path = "$" + "".join(
                    f"[{repr(p)}]" if isinstance(p, int) else f".{p}" for p in err.path
                )
                lines.append(f"{path}: {err.message}")
            raise ValueError("JSON Schema validation failed:\n" + "\n".join(lines))

    def validate_model(self, model_obj) -> None:
        try:
            self.validate_dict(asdict(model_obj))
        except TypeError:
            # asdict failed (not a dataclass) — skip
            return
