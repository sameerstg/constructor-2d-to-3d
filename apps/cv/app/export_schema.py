"""Export the IR JSON Schema from the Pydantic models.

The Pydantic models in `app.ir` are the single source of truth. This writes the
schema out so the TypeScript side has something to check itself against, and so
the contract is reviewable in a diff when it changes.

    python -m app.export_schema
"""

from __future__ import annotations

import json
from pathlib import Path

from app.ir import FloorPlanIR

# apps/cv/app/export_schema.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT = REPO_ROOT / "packages" / "schema" / "ir.schema.json"


def main() -> None:
    schema = FloorPlanIR.model_json_schema(by_alias=True)
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://teknardo.dev/schemas/floorplan-ir-1.0.0.json"
    schema["title"] = "FloorPlanIR"

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
