"""Run existing unit tests without model calls; report optional archive skips.

This release helper does not change any application or test implementation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


def cases(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from cases(item)
        else:
            yield item


def archive_available():
    manifest = ROOT / "evaluation/data/research_v5/source_manifest.json"
    records = json.loads(manifest.read_text())["sources"]
    return all((ROOT / record[field]).is_file() for record in records for field in ("pdf_path", "text_path"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-sources", action="store_true", help="Fail instead of skipping when downloaded paper archives are absent")
    args = parser.parse_args()
    for name, value in {"LANGSMITH_TRACING": "false", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"}.items():
        os.environ[name] = value
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT))
    present = archive_available()
    if args.require_sources and not present:
        parser.error("Restore paper archives with scripts/fetch_research_sources.py --fetch first")
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py")
    selected = unittest.TestSuite()
    for case in cases(suite):
        if not present and case.id().endswith(".test_rebuild_and_exact_pdf_text_provenance"):
            def unavailable_archive():
                raise unittest.SkipTest("Optional original paper archives are not distributed; restore with fetch_research_sources.py --fetch")
            selected.addTest(unittest.FunctionTestCase(unavailable_archive, description=case.id()))
        else:
            selected.addTest(case)
    result = unittest.TextTestRunner(verbosity=2).run(selected)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
