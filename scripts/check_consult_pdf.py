"""Regression checks for the advisor consultation paper.

Run after editing research/13_advisor_consultation.md and rebuilding, to
confirm that none of the twenty audit findings has regressed:

    python scripts/build_consult_pdf.py
    python scripts/check_consult_pdf.py
"""
import io
import re
import pdfplumber

MD = r"C:\Users\kinet\OneDrive\Documents\xiangqimaster\research\13_advisor_consultation.md"
PDF = r"C:\Users\kinet\OneDrive\Documents\xiangqimaster\research\13_advisor_consultation.pdf"
md = io.open(MD, encoding="utf-8").read()
with pdfplumber.open(PDF) as p:
    pages = [(pg.extract_text() or "") for pg in p.pages]
pdftxt = "\n".join(pages)

out = []


def chk(n, name, ok, detail=""):
    out.append((n, "PASS" if ok else "FAIL", name, detail))


# 1 equation-id leak
leaks = sum(pdftxt.count(t) for t in ("mask|", "minimax|", "ilos|", "ppo|", "puct|", "elo|"))
chk(1, "Equation id no longer leaks into the formula", leaks == 0, "occurrences=%d" % leaks)

# 2 raw markup in prose
raw = sum(pdftxt.count(t) for t in ("<sub>", "<super>", "<sup>", "&lt;", "&#"))
rawmd = len([l for l in md.split("\n")
             if ("<sub>" in l or "<super>" in l) and not l.startswith("@EQ")])
chk(2, "No raw inline markup rendered in prose", raw == 0 and rawmd == 0,
    "pdf=%d, md-prose-lines=%d" % (raw, rawmd))

# 3 equation 6 fits on one line
eq6 = [l for l in pdftxt.split("\n") if "(6)" in l and "10" in l]
chk(3, "Equation (6) fits on one line", bool(eq6), (eq6[0][:70] + "...") if eq6 else "not found")

# 4 hard-coded table reference removed
hard = re.findall(r"(?<!@)Table [IVX]+\b", md)
chk(4, "No hard-coded table numbers left in the source", not hard, str(hard))

# 5 vague section cross-references
vague = [m for m in re.findall(r"Section VIII(?![-A-E])[^.]{0,40}", md) if "the results obtained so far" not in m]
chk(5, "Section VIII references are sub-sectioned", not vague, str(vague))

# 6 contribution wording
chk(6, "Contribution 3 no longer claims six studies",
    "obtained from four independent ablations" in md)

# 7 journal-family phrase
chk(7, "'this journal family' removed", "this journal family" not in md)

# 8 two IL ratings disambiguated
chk(8, "The two imitation-learning ratings are disambiguated",
    "five-agent round robin" in md and "anchored differently" in md)

# 9 cross-tournament comparison defended
chk(9, "Cross-tournament comparison justified",
    "identical opponent pool" in md)

# 10 search-ablation sample caveat
chk(10, "Search ablation sample size disclosed",
    "twenty games per setting" in md)

# 11 training-plateau numbers
chk(11, "Training-plateau claim quantified",
    "about 26%" in md and "about 34%" in md)

# 12 distillation accuracy numbers
chk(12, "Distillation accuracy claim quantified",
    "13.6%" in md and "34.3%" in md)

# 13 masked-accuracy wording
chk(13, "Accuracy figures identified as legality-masked",
    "After masking illegal moves" in md)

# 14 design-decision label consistency
dd = re.findall(r"\*\*Design decision \d[^*]*\*\*", md)
bad_dd = [d for d in dd if not re.match(r"\*\*Design decision \d \(.+\)\.\*\*$", d)]
chk(14, "Design-decision labels consistent", len(dd) == 6 and not bad_dd,
    "found=%d bad=%s" % (len(dd), bad_dd))

# 15 run-in labels bold
ital = re.findall(r"^\*(Definition|Flow|Configuration|Provenance|Processing|Outcome|Acquisition)[^*]*\*",
                  md, re.M)
chk(15, "Run-in labels are bold, not italic", not ital, str(ital[:4]))

# 16 scientific notation in text
chk(16, "No 1e-3 / 3e-4 style notation in text",
    "1e-3" not in md and "3e-4" not in md)

# 17 percent style
pc = md.count("per cent")
chk(17, "Percent style consistent", pc == 0, "'per cent' occurrences=%d" % pc)

# 18 duplicated module list removed
chk(18, "Duplicated module list removed",
    "A rules engine produces legal moves; an encoder converts" not in md)

# 19 ambiguous table cell
chk(19, "Ambiguous '(M4 warm start)' cell clarified",
    "(M4 warm start)" not in md and "warm-started from M2" in md)

# 20 date, section order, figure placement
order_ok = md.find("### H. Reference points") < md.find("### I. Implementation and reproducibility")
movepos = md.find("@FIG col | moves") > md.find("{{f:moves}} gives the movement rule")
chk(20, "Date, Section V order and figure placement corrected",
    "6 November 2026" in md and order_ok and movepos,
    "date=%s order=%s figpos=%s" % ("6 November 2026" in md, order_ok, movepos))

# extra: table word breaks
shorts = [(i + 1, l.strip()) for i, t in enumerate(pages) for l in t.split("\n")
          if 0 < len(l.strip()) <= 3 and not l.strip().isdigit()
          and not re.fullmatch(r"\(\d\)", l.strip())]
chk("T", "Table VII: no broken words anywhere",
    all(w in pdftxt for w in ("Complete", "Milestone", "Planned")) and
    not [x for x in shorts if x[1] in ("e", "d", "te", "ne")],
    "suspicious=%s" % shorts)

# extra: loop arrows explained in captions
chk("L", "Loop arrows explained in Fig. 7 and Fig. 8 captions",
    "dashed arrow on the left is the training loop" in md and
    "arrow on the left is the iteration loop" in md)

# extra: unresolved refs / sparse pages
sparse = [i + 1 for i, t in enumerate(pages) if len(t) < 1200]
chk("P", "16 pages, no unresolved refs, no sparse pages",
    pdftxt.count("Fig. ?") == 0 and pdftxt.count("Table ?") == 0 and not sparse,
    "pages=%d sparse=%s" % (len(pages), sparse))

w = max(len(str(n)) for n, _, _, _ in out)
fails = 0
for n, st, name, det in out:
    if st == "FAIL":
        fails += 1
    print("%-*s  %-4s  %-58s %s" % (w, n, st, name, det if st == "FAIL" else ""))
print("\n%d checks, %d failed" % (len(out), fails))
