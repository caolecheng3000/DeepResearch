"""Keep generated documentation and its relative artifact links under docs/."""
from pathlib import Path
import os
import re

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
REPORT = DOCS / "CAPABILITY_EVALUATION_20261018.md"
RETIRED = {
    "EVALUATION_VALIDATION_20261018.md": "追踪体系初始验证",
    "REPORT_EVALUATION_VALIDATION_20261018.md": "报告测评与对抗实验",
}


def rebase_links(text: str, source: Path, destination: Path) -> str:
    """Relocate Markdown links, preserving external URLs and local fragments."""
    def replace(match):
        label, target = match.groups()
        if not target or target.startswith("#") or re.match(r"[a-zA-Z][\w+.-]*:", target):
            return match.group(0)
        wrapped = target.startswith("<") and target.endswith(">")
        path, separator, fragment = target.strip("<>").partition("#")
        resolved = (source / path).resolve()
        if resolved.parent == ROOT:
            if resolved.name in RETIRED:
                resolved = DOCS / "历史测评记录.md"
                if not separator:
                    separator, fragment = "#", RETIRED[Path(path).name]
            elif (DOCS / resolved.name).is_file():
                resolved = DOCS / resolved.name
        relative = Path(os.path.relpath(resolved, destination)).as_posix()
        target = relative + (separator + fragment if separator else "")
        if wrapped:
            target = "<" + target + ">"
        return label + "(" + target + ")"
    return re.sub(r"(!?\[[^\]\n]*\])\(([^)\n]*)\)", replace, text)


def write_report(base: Path, text: str) -> None:
    DOCS.mkdir(exist_ok=True)
    REPORT.write_text(rebase_links(text, ROOT, DOCS), encoding="utf-8")
    (base / "final_report.md").write_text(rebase_links(text, ROOT, base), encoding="utf-8")
