"""Check docs, notebook markdown and code comments for dashes, semicolons and hyphens in prose.

Usage:
    python tools/check_prose.py README.md docs/index.md notebooks/hybrid_yolo_workshop.ipynb
"""

import json
import re
import sys
from pathlib import Path


def clean_md(text):
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL)
    text = re.sub(r"`[^`]*`", "", text)
    text = re.sub(r"\]\([^)]*\)", "]", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"<code>.*?</code>", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r":[a-z0-9_+-]+:", "", text)
    text = re.sub(r"\{[^}]*\}", "", text)
    text = re.sub(r"^\s*\|?[-:| ]+\|?\s*$", "", text, flags=re.MULTILINE)  # table separators
    text = re.sub(r"^\s*[-*] ", "", text, flags=re.MULTILINE)  # bullets
    text = re.sub(r"^\s*- \[[ x]\] ", "", text, flags=re.MULTILINE)
    return text


def comments(path):
    src = path.read_text()
    out = []
    if path.suffix == ".py":
        out += re.findall(r'"""(.*?)"""', src, flags=re.DOTALL)
        out += re.findall(r"#\s(.*)", src)
        out += re.findall(r'help="([^"]*)"', src)
    elif path.suffix in (".sh", ".yml", ".yaml", ".toml"):
        out += re.findall(r"#\s(.*)", src)
        out += re.findall(r'(?:say|echo) "([^"]*)"', src)
    elif path.suffix == ".rs":
        out += re.findall(r"//+!?\s?(.*)", src)
    elif path.suffix == ".css":
        out += re.findall(r"/\*(.*?)\*/", src, flags=re.DOTALL)
    return "\n".join(out)


files = [Path(f) for f in sys.argv[1:]]
bad = 0
for f in files:
    if f.suffix == ".md":
        text = clean_md(f.read_text())
    elif f.suffix == ".ipynb":
        nb = json.loads(f.read_text())
        text = clean_md("\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown"))
        text += "\n" + "\n".join(
            re.findall(r"#\s(.*)", "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"))
        )
    else:
        text = comments(f)
    text = re.sub(r"`[^`]*`", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"(?<![\w])--?[a-z][a-z-]*", "", text)  # cli flags quoted in usage lines
    for i, line in enumerate(text.splitlines()):
        hits = []
        if "—" in line or "–" in line:
            hits.append("dash")
        if ";" in line:
            hits.append("semicolon")
        for m in re.finditer(r"[A-Za-z0-9]-[A-Za-z]", line):
            hits.append("hyphen:" + line[max(0, m.start() - 12) : m.end() + 12])
        if hits:
            bad += 1
            print(f"{f}: {hits} :: {line.strip()[:140]}")
print("issues", bad)
