"""
Release check: what a push tells the people who use the nodes must match the version it carries.

The version in pyproject.toml has to be named in the README's "What's new" section and has to have a dated entry in
CHANGELOG.md (not "not released yet"). Run by the git pre-push hook next to the privacy check:

    python tests/check_release.py
Exit code 1 and the reason when something is missing.
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(name):
    with open(os.path.join(ROOT, name), encoding="utf-8") as f:
        return f.read()


def problems():
    m = re.search(r'(?m)^version\s*=\s*"([^"]+)"', read("pyproject.toml"))
    if not m:
        return ["pyproject.toml has no version"]
    version, out = m.group(1), []
    readme = read("README.md")
    new = re.search(r"(?ms)^## What's new\s*\n(.*?)(?=^## )", readme)
    if not new:
        out.append("README.md has no \"## What's new\" section")
    elif f"**{version}**" not in new.group(1):
        out.append(f"README.md \"What's new\" does not name version {version}: rewrite it for this version")
    elif "CHANGELOG.md" not in new.group(1):
        out.append("README.md \"What's new\" no longer links to CHANGELOG.md")
    entry = re.search(rf"(?m)^## {re.escape(version)} \((.+?)\)\s*$", read("CHANGELOG.md"))
    if not entry:
        out.append(f"CHANGELOG.md has no entry '## {version} (date)'")
    elif not re.fullmatch(r"\d{4}-\d{2}-\d{2}", entry.group(1)):
        out.append(f"CHANGELOG.md: the entry of {version} has no date yet ('{entry.group(1)}')")
    return out


if __name__ == "__main__":
    found = problems()
    for p in found:
        print("release check:", p)
    print("release check passed" if not found else "release check failed")
    sys.exit(1 if found else 0)
