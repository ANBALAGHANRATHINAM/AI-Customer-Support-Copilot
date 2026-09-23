"""Run one Segment 4 end-to-end response-engine request from the project root."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "4_AI_Response_Engine" / "code"))

from llm import StaticAcknowledgementLLM
from response_engine import ZendsResponseEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a source-grounded ZENDS response recommendation.")
    parser.add_argument("query", help="Customer query to analyze.")
    parser.add_argument("--offline", action="store_true", help="Use the deterministic acknowledgement adapter instead of loading FLAN-T5.")
    args = parser.parse_args()
    engine = ZendsResponseEngine.from_project_assets(llm=StaticAcknowledgementLLM() if args.offline else None)
    print(json.dumps(engine.respond(args.query), indent=2))


if __name__ == "__main__":
    main()
