"""Audit and export a source-only Deep-Research snapshot, without Git history.

Never deletes local files, rewrites history, publishes, or calls a model service.
Uses the repository ignore rules even for files already tracked by Git.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def git(*args, input=None):
    result = subprocess.run(["git", *args], cwd=ROOT, input=input, capture_output=True)
    if result.returncode not in (0, 1):
        raise RuntimeError("Git file enumeration failed; no release was exported")
    return result.stdout


def release_files():
    repository = git("rev-parse", "--show-toplevel").decode().strip()
    if not repository or Path(repository).resolve() != ROOT.resolve():
        raise ValueError("Run in the project Git root; initialise a new repository before auditing an exported snapshot")
    candidates = sorted(set(filter(None, git("ls-files", "--cached", "--others", "--exclude-standard", "-z").decode().split("\0"))))
    ignored = set(filter(None, git("check-ignore", "--no-index", "-z", "--stdin", input=("\0".join(candidates) + "\0").encode()).decode().split("\0")))
    selected = []
    for relative in candidates:
        path = ROOT / relative
        if relative in ignored or not path.exists():
            continue
        if path.is_symlink() or ROOT not in path.resolve().parents:
            raise ValueError(f"Release does not follow symlinks or external paths: {relative}")
        if path.is_file():
            selected.append(relative)
    local_ignored = set(filter(None, git("ls-files", "--others", "--ignored", "--exclude-standard", "-z").decode().split("\0")))
    return selected, ignored | local_ignored


def known_secrets():
    result = []
    for path in (ROOT / ".env", ROOT / ".env.local"):
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            match = re.match(r"\s*(?:export\s+)?([A-Z0-9_]+)\s*=\s*(.*?)\s*$", line)
            if not match or not re.search(r"KEY|TOKEN|PASSWORD|SECRET", match[1]):
                continue
            value = match[2].strip("\"'")
            if len(value) >= 16 and not re.search(r"your|placeholder|example|^EMPTY$", value, re.I):
                result.append((match[1], value.encode()))
    return result


def audit(files):
    credentials = known_secrets()
    issues = []
    protected = (".git/", "outputs/", "data/", "evaluation/data/external/", "evaluation/data/research_v5/sources/")
    for relative in files:
        path = ROOT / relative
        if (relative.startswith(protected) and path.name != "README.md") or (path.name.startswith(".env") and path.name not in (".env.template", ".env.tools.template")):
            issues.append({"path": relative, "reason": "local/private artifact selected"})
        data = path.read_bytes()
        for variable, value in credentials:
            if value in data:
                issues.append({"path": relative, "reason": f"configured credential matched: {variable}"})
        if re.search(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", data):
            issues.append({"path": relative, "reason": "private key marker"})
        for label, pattern in (
            ("provider API token", rb"\bsk-[A-Za-z0-9_-]{24,}\b"),
            ("GitHub token", rb"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
            ("AWS access key", rb"\bAKIA[A-Z0-9]{16}\b"),
        ):
            if re.search(pattern, data):
                issues.append({"path": relative, "reason": label + " pattern; review without printing value"})
        if path.stat().st_size > 50 * 1024 * 1024:
            issues.append({"path": relative, "reason": "unexpected file larger than 50 MiB"})
    commits = git("log", "--all", "--format=%h", "--", ".env", ".env.local").decode().splitlines()
    return {"snapshot_status": "passed" if not issues else "blocked", "issues": issues,
            "selected_files": len(files), "selected_bytes": sum((ROOT / p).stat().st_size for p in files),
            "historical_dotenv_commits": len(set(commits)),
            "history_notice": "Snapshot audit does not certify old history. Export excludes .git; revoke/rotate any previously committed real credentials."}


def export(files, target: Path, report: dict, archive: Path | None):
    target = target.resolve()
    if target == ROOT or ROOT in target.parents and target.relative_to(ROOT).parts[0] != "dist":
        raise ValueError("Use an external directory or a new directory under dist/")
    if target.exists():
        raise FileExistsError("Release destination already exists; use a new directory")
    if archive is not None and archive.resolve().exists():
        raise FileExistsError("Archive already exists; use a new name")
    target.mkdir(parents=True)
    hashes = {}
    for relative in files:
        source, destination = ROOT / relative, target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[relative] = hashlib.sha256(destination.read_bytes()).hexdigest()
    manifest = {"project": "Deep-Research", "git_history_included": False, "audit": report,
                "files_sha256": hashes,
                "optional_archives": "Original paper PDFs/full text, external dataset cache, Word exports, runtime state and raw experiment outputs are not included."}
    (target / "PUBLIC_SNAPSHOT.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    if archive:
        archive = archive.resolve()
        archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(target.rglob("*")):
                if path.is_file():
                    bundle.write(path, Path("Deep-Research") / path.relative_to(target))
    return str(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Audit only; this is also the default")
    parser.add_argument("--output", type=Path, help="Export a new directory with no .git or local artifacts")
    parser.add_argument("--archive", type=Path, help="Also write a new ZIP archive")
    args = parser.parse_args()
    if args.archive and not args.output:
        parser.error("--archive requires --output")
    if args.check and args.output:
        parser.error("Use either --check or --output")
    files, ignored = release_files()
    report = audit(files)
    report["excluded_file_groups"] = dict(Counter(p.split("/")[0] for p in ignored))
    if report["issues"]:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(1)
    if args.output:
        report["exported_to"] = export(files, args.output, report, args.archive)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
