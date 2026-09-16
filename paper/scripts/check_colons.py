"""Colons the reader actually sees: strip LaTeX internals first, then look."""
import io
import re

P = r"C:/Users/cak7496/Desktop/first paper/MicroCT-paper/main.tex"
lines = io.open(P, encoding="utf-8").read().split("\n")

CMDS = ("label", "ref", "eqref", "cite", "autoref", "Cref", "newcommand",
        "usepackage", "documentclass", "input", "includegraphics",
        "bibliography", "bibliographystyle")
strip = re.compile(r"\\(?:" + "|".join(CMDS) + r")\s*\{[^}]*\}")

for i, l in enumerate(lines, 1):
    if l.strip().startswith("%"):
        continue
    t = strip.sub("", l)
    t = re.sub(r"https?://\S+", "", t)
    if ":" in t:
        print("%5d  %s" % (i, l.strip()[:96]))
