"""
Privacy check: nothing that belongs to one machine or one person may be committed.

Flags absolute drive paths (C:\\..., D:/...), network paths, user folders (/Users/<name>, /home/<name>, \\Users\\<name>),
and e-mail addresses in every file git would commit. Words that are private to the owner of a checkout (names,
project names, folders) go into private/privacy_terms.txt, one per line (the private/ folder is git-ignored, so the
words themselves are never published). Lines listed in tests/privacy_allow.txt (substring matches) are allowed.

    python tests/check_private.py            # the working tree (tracked + new files)
    python tests/check_private.py --staged   # what the next commit contains (git pre-commit hook)
    python tests/check_private.py --pushed <range>   # the commits a push sends (git pre-push hook)
Exit code 1 and a list of file:line hits when something is found.
"""

import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GENERIC = [
    (r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]+(?![\\/])(?!Program Files|Windows[\\/]|ProgramData)[^\s\"'<>|]*", "absolute drive path"),
    (r"\\\\[A-Za-z0-9_.-]+\\[A-Za-z0-9_$.-]+", "network path"),
    (r"(?i)[\\/](?:Users|home)[\\/](?!Public\b|<|\{|\$|%|name\b|you\b|me\b)[A-Za-z0-9_.-]+", "user folder"),
    (r"[A-Za-z0-9_.+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9.-]+", "e-mail address"),
]
EMAIL_OK = ("noreply@anthropic.com", "@example.com", "@example.org", "git@github.com")
BINARY = (".png", ".jpg", ".jpeg", ".webp", ".exr", ".woff", ".woff2", ".ttf", ".mp4", ".mov", ".wav", ".mp3", ".glb",
          ".fbx", ".blend", ".npy", ".pdf", ".ai", ".onnx", ".safetensors", ".zip", ".ico", ".gif")


def _lines(path):
    p = os.path.join(ROOT, path)
    try:
        return open(p, encoding="utf-8", errors="replace").read().splitlines()
    except OSError:
        return []


def _git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace").stdout


def rules():
    out = [(re.compile(rx), why) for rx, why in GENERIC]
    terms = os.path.join(ROOT, "private", "privacy_terms.txt")
    if os.path.isfile(terms):
        for t in open(terms, encoding="utf-8").read().splitlines():
            t = t.strip()
            if t and not t.startswith("#"):
                out.append((re.compile(re.escape(t), re.I), "private word"))
    return out


def allowed():
    p = os.path.join(HERE, "privacy_allow.txt")
    return [a.strip() for a in open(p, encoding="utf-8").read().splitlines() if a.strip() and not a.startswith("#")] \
        if os.path.isfile(p) else []


def scan(files, reader):
    rs, allow, hits = rules(), allowed(), []
    for f in files:
        if f.lower().endswith(BINARY) or f.startswith("private/") or f in ("tests/privacy_allow.txt", "tests/check_private.py"):
            continue
        for n, line in enumerate(reader(f), 1):
            for rx, why in rs:
                for m in rx.finditer(line):
                    s = m.group(0)
                    if why == "e-mail address" and any(ok in s for ok in EMAIL_OK):
                        continue
                    if any(a in line for a in allow):
                        continue
                    hits.append(f"{f}:{n}: {why}: {s[:80]}")
    return hits


def main(argv):
    if "--staged" in argv:
        files = [f for f in _git("diff", "--cached", "--name-only", "--diff-filter=ACMR").splitlines() if f]
        hits = scan(files, lambda f: _git("show", f":{f}").splitlines())
    elif "--pushed" in argv:
        rng = argv[argv.index("--pushed") + 1]
        files = sorted({f for f in _git("log", "--name-only", "--format=", rng).splitlines() if f})
        tip = rng.split("..")[-1] or "HEAD"
        hits = scan(files, lambda f: _git("show", f"{tip}:{f}").splitlines())
        msgs = _git("log", "--format=%B", rng).splitlines()      # commit messages too
        hits += [h.replace("<messages>", "commit message") for h in scan(["<messages>"], lambda f: msgs)]
    else:
        files = [f for f in _git("ls-files", "-co", "--exclude-standard").splitlines() if f]
        hits = scan(files, _lines)
    # private/ is git-ignored, so a path from it in any of these lists is a file git tracks (e.g. after `git mv`)
    hits += [f"{f}: the private folder must never be tracked (git rm --cached)" for f in files if f.startswith("private/")]
    if hits:
        print("privacy check FAILED - these would publish something private (fix, or allow in tests/privacy_allow.txt):")
        print("\n".join("  " + h for h in hits[:200]) + (f"\n  ... {len(hits) - 200} more" if len(hits) > 200 else ""))
        return 1
    print("privacy check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
