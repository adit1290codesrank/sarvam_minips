"""List every source linked in a markdown report, for the appendix source table.

Usage:
    python tools/list_cited_sources.py "reports/Sarvam mini PS research.md"            # print a table
    python tools/list_cited_sources.py report.md --stop "## Appendix"                  # ignore links after a heading

Prints one markdown row per distinct URL: number, link text, and the section where it is first cited.
Fill in a one-line "what it told us" for each row before submitting.
"""
import argparse
import re

LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
HEADING = re.compile(r"^#{2,4}\s+(.*)")


def cited_sources(text, stop=None):
    seen, rows, section = set(), [], "Opening summary"
    for line in text.splitlines():
        if stop and line.startswith(stop):
            break
        h = HEADING.match(line)
        if h:
            section = h.group(1).split(".")[0].split(" — ")[0].strip()
            continue
        for label, url in LINK.findall(line):
            if url in seen or url.startswith("#"):
                continue
            seen.add(url)
            rows.append((label.replace("|", "\\|"), url, section))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("report")
    ap.add_argument("--stop", help="stop at the first line starting with this text, e.g. '## Appendix'")
    args = ap.parse_args()
    rows = cited_sources(open(args.report, encoding="utf8").read(), args.stop)
    print("| # | Source | First cited in |")
    print("|---|---|---|")
    for n, (label, url, section) in enumerate(rows, 1):
        print(f"| {n} | [{label}]({url}) | {section} |")
