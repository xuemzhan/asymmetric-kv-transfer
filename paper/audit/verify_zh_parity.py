#!/usr/bin/env python3
"""W33 guard: the Chinese edition must mirror the English edition exactly.

Checks paper/main_zh.tex against paper/main.tex on the quantities that carry
meaning across a translation:

  * every numeric literal (a number is never translated),
  * every \\label / \\ref / \\eqref target and every \\cite key,
  * the bibliography entries and their order,
  * the included figure files, in order,
  * the table/figure environment counts and the row count of every table,
  * the section count.

Exit code 0 means parity holds; otherwise every difference is listed.
"""
from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EN_PATH = os.path.join(ROOT, "paper", "main.tex")
ZH_PATH = os.path.join(ROOT, "paper", "main_zh.tex")

NUMBER = re.compile(r"(?<![A-Za-z0-9_.])\d+(?:\.\d+)?(?![0-9])")
TABLE = re.compile(r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}", re.S)


def read(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def numbers(text: str) -> list[str]:
    stripped = re.sub(r"\\text\{[^}]*\}", " ", text)
    stripped = re.sub(r"\\[,;!]\s*|\s+|--", " ", stripped)
    return sorted(NUMBER.findall(stripped))


def captures(pattern: str, text: str) -> list[str]:
    return re.findall(pattern, text)


def main() -> int:
    en = read(EN_PATH)
    zh = read(ZH_PATH)
    failures: list[str] = []

    def compare(name: str, left: list[str], right: list[str]) -> None:
        if left == right:
            print("  ok   %-18s (%d)" % (name, len(left)))
            return
        missing = sorted(set(left) - set(right))
        extra = sorted(set(right) - set(left))
        failures.append("%s: %d vs %d; missing=%s extra=%s"
                        % (name, len(left), len(right), missing[:8], extra[:8]))

    def compare_sorted(name: str, pattern: str) -> None:
        compare(name, sorted(captures(pattern, en)), sorted(captures(pattern, zh)))

    print("zh parity guard: %s" % os.path.relpath(ZH_PATH, ROOT))
    compare("numeric literals", numbers(en), numbers(zh))
    compare_sorted("labels", r"\\label\{([^}]*)\}")
    compare_sorted("refs", r"\\ref\{([^}]*)\}")
    compare_sorted("eqrefs", r"\\eqref\{([^}]*)\}")
    compare_sorted("cite keys", r"\\cite\{([^}]*)\}")
    compare("bibitem order", captures(r"\\bibitem\{([^}]*)\}", en),
            captures(r"\\bibitem\{([^}]*)\}", zh))
    compare("includegraphics", captures(r"\\includegraphics\[[^\]]*\]\{([^}]*)\}", en),
            captures(r"\\includegraphics\[[^\]]*\]\{([^}]*)\}", zh))
    compare("figure environments", captures(r"\\begin\{figure\}", en),
            captures(r"\\begin\{figure\}", zh))
    compare("table environments", captures(r"\\begin\{table\}", en),
            captures(r"\\begin\{table\}", zh))

    en_rows = [len(re.findall(r"\\\\", block)) for block in TABLE.findall(en)]
    zh_rows = [len(re.findall(r"\\\\", block)) for block in TABLE.findall(zh)]
    compare("table rows", [str(value) for value in en_rows],
            [str(value) for value in zh_rows])

    en_sections = len(captures(r"\\section\{", en))
    zh_sections = len(captures(r"\\section\{", zh))
    if en_sections == zh_sections:
        print("  ok   %-18s (%d)" % ("sections", en_sections))
    else:
        failures.append("sections: %d vs %d" % (en_sections, zh_sections))

    if failures:
        print("\nFAILURES")
        for item in failures:
            print("  - " + item)
        return 1
    print("\nall parity checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
