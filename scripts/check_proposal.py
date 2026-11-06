"""Check the research proposal against the supervisor's required points.

    python scripts/build_consult_pdf.py research/proposal/proposal_v2.md \
        research/proposal/L25020007_Proposal_v2.pdf
    python scripts/check_proposal.py
"""
import io
import os
import re
import pdfplumber

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD = os.path.join(ROOT, "research", "proposal", "proposal_v2.md")
PDF = os.path.join(ROOT, "research", "proposal", "L25020007_Proposal_v2.pdf")
md = io.open(MD, encoding="utf-8").read()
with pdfplumber.open(PDF) as p:
    pages = [(pg.extract_text() or "") for pg in p.pages]
txt = "\n".join(pages)

rows = []


def chk(n, item, ok, note=""):
    rows.append((n, "PASS" if ok else "FAIL", item, note))


chk(1, "Research plan (method + procedure + schedule)",
    "PROCEDURE" in md.upper() and "SCHEDULE" in md.upper())
chk(2, "Problem stated and marked", "**Problem.**" in md)
chk(3, "Solution stated and marked", "**Solution.**" in md)
chk(4, "Decisions with explicit rationale",
    "DESIGN DECISIONS AND THEIR RATIONALE" in md.upper()
    and len(re.findall(r"\*\*D\d,", md)) >= 5)
chk(5, "Blueprint of the solution",
    "SOLUTION BLUEPRINT" in md.upper() and "controlled apparatus" in md)
chk(6, "Research design (IV / DV / controls)",
    "independent variable" in md and "dependent variable" in md
    and "**control**" in md)
chk(7, "How objectives will be achieved",
    all(("O%d" % k) in md for k in range(1, 6))
    and "Experiment 1 (RQ1, O3)" in md)
chk(8, "Benefit / contribution",
    "EXPECTED CONTRIBUTIONS AND BENEFICIARIES" in md.upper()
    and "Beneficiaries" in md)
chk(9, "Timeline to defence, end of May 2027",
    "end of May 2027" in md and "May 2027" in txt and "Nov 2026" in txt)
chk(10, "Where the problem comes from is reported",
    "documented methodological failure" in md and "originates in the published literature" in md)
chk(11, "Gap supported by literature, not assumption",
    "**Research gap.**" in md and "Table I" in txt.replace("TABLE I", "Table I"))
chk(12, "States what others have not done",
    "none compares paradigms" in md or "No controlled, Elo-anchored comparison" in md)
chk(13, "Problem and solution visually marked",
    "**Problem.**" in md and "**Solution.**" in md
    and "**Research gap.**" in md)
refs = re.findall(r"^\[\d+\]", md, re.M)
nonpeer = len(re.findall(r"arXiv:\d", md))
chk(14, "Most references peer-reviewed",
    len(refs) >= 18 and nonpeer <= len(refs) * 0.3,
    "%d refs, %d arXiv-only" % (len(refs), nonpeer))
bad_cite = re.findall(r"\[\d+\]\s*\n\n", md)
chk(15, "Citations sit at the sentence that uses them", not bad_cite)
chk(16, "Sources identified and their useful point summarised",
    len(re.findall(r"\*Useful point:?\*", md)) >= 5,
    "%d summaries" % len(re.findall(r"\*Useful point:?\*", md)))
chk(17, "Persuasive closing for the presentation",
    "worth doing" in md and "CONCLUSION" in md.upper())

# production checks
chk("A", "No unresolved cross-references",
    txt.count("Fig. ?") == 0 and txt.count("Table ?") == 0 and "{{" not in txt)
chk("B", "No markup leaked into the text",
    all(t not in txt for t in ("<sub>", "<super>", "&lt;", "&#", "|E")))
sparse = [i + 1 for i, t in enumerate(pages) if len(t) < 1200]
chk("C", "No sparse pages", not sparse, "pages=%d sparse=%s" % (len(pages), sparse))
# a lone letter is a subscript artefact of PDF text extraction (E_A, R_A),
# not a broken word, so it is excluded
short = [l.strip() for t in pages for l in t.split("\n")
         if 0 < len(l.strip()) <= 2 and not l.strip().isdigit()
         and not re.fullmatch(r"[A-Za-z]", l.strip())]
chk("D", "No broken words in tables", not short, str(short[:6]))

w = max(len(str(r[0])) for r in rows)
fails = sum(1 for r in rows if r[1] == "FAIL")
for n, st, item, note in rows:
    print("%-*s  %-4s  %-54s %s" % (w, n, st, item, note if st == "FAIL" or note else ""))
print("\n%d checks, %d failed  (%d pages)" % (len(rows), fails, len(pages)))
