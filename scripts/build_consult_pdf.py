"""Typeset the consultation report as a two-column IEEE paper.

Typography is matched to the project's own IEEEtran proposal, measured from
that PDF: Times 16 pt bold title, 11 pt author, 10 pt affiliation, 9 pt bold
abstract, 10/12 pt body, 10 pt centred small-caps section heads, 8 pt figure
captions, 7.8 pt table bodies, two 248 pt columns with an 18 pt gutter on US
Letter with 49 pt side margins.
"""
import os
import re
import io

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph,
                                Spacer, Image, Table, TableStyle, ListFlowable,
                                ListItem, KeepTogether, PageBreak, FrameBreak,
                                NextPageTemplate)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image as PILImage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "research", "13_advisor_consultation.md")
OUT = os.path.join(ROOT, "research", "13_advisor_consultation.pdf")
WF = r"C:\Windows\Fonts"

pdfmetrics.registerFont(TTFont("TM", os.path.join(WF, "times.ttf")))
pdfmetrics.registerFont(TTFont("TM-B", os.path.join(WF, "timesbd.ttf")))
pdfmetrics.registerFont(TTFont("TM-I", os.path.join(WF, "timesi.ttf")))
pdfmetrics.registerFont(TTFont("TM-BI", os.path.join(WF, "timesbi.ttf")))
pdfmetrics.registerFontFamily("TM", normal="TM", bold="TM-B", italic="TM-I",
                              boldItalic="TM-BI")
CJK = "TM"
for cand, idx in (("msyh.ttc", 0), ("simsun.ttc", 0), ("simhei.ttf", None)):
    pth = os.path.join(WF, cand)
    if os.path.exists(pth):
        try:
            pdfmetrics.registerFont(TTFont("CJK", pth) if idx is None
                                    else TTFont("CJK", pth, subfontIndex=idx))
            CJK = "CJK"
            break
        except Exception:
            pass

PW, PH = letter
LM = RM = 49.0
TM_ = 52.0
BM = 56.0
GUT = 18.0
COLW = (PW - LM - RM - GUT) / 2.0
SPANW = PW - LM - RM
BODYH = PH - TM_ - BM
BK = colors.black
MUTE = colors.HexColor("#3f3f3f")

S = dict(
    title=ParagraphStyle("title", fontName="TM-B", fontSize=16, leading=19,
                         alignment=TA_CENTER, spaceAfter=10),
    auth=ParagraphStyle("auth", fontName="TM", fontSize=11, leading=13,
                        alignment=TA_CENTER),
    meta=ParagraphStyle("meta", fontName="TM", fontSize=10, leading=13,
                        alignment=TA_CENTER),
    abs=ParagraphStyle("abs", fontName="TM-B", fontSize=9, leading=10.4,
                       alignment=TA_JUSTIFY, spaceAfter=5),
    h1=ParagraphStyle("h1", fontName="TM", fontSize=10, leading=12,
                      alignment=TA_CENTER, spaceBefore=11, spaceAfter=4),
    h2=ParagraphStyle("h2", fontName="TM-I", fontSize=10, leading=12,
                      spaceBefore=7, spaceAfter=2),
    h3=ParagraphStyle("h3", fontName="TM-I", fontSize=9.6, leading=11.5,
                      spaceBefore=5, spaceAfter=1),
    body=ParagraphStyle("body", fontName="TM", fontSize=10, leading=12,
                        alignment=TA_JUSTIFY, firstLineIndent=10, spaceAfter=0),
    flat=ParagraphStyle("flat", fontName="TM", fontSize=10, leading=12,
                        alignment=TA_JUSTIFY, spaceAfter=0),
    eq=ParagraphStyle("eq", fontName="TM", fontSize=9.6, leading=12,
                      alignment=TA_CENTER),
    eqn=ParagraphStyle("eqn", fontName="TM", fontSize=9.6, leading=12,
                       alignment=TA_LEFT),
    cap=ParagraphStyle("cap", fontName="TM", fontSize=8, leading=9.3,
                       alignment=TA_JUSTIFY, spaceBefore=3),
    tnum=ParagraphStyle("tnum", fontName="TM", fontSize=8, leading=9.4,
                        alignment=TA_CENTER, spaceBefore=4),
    tcap=ParagraphStyle("tcap", fontName="TM", fontSize=8, leading=9.4,
                        alignment=TA_CENTER, spaceAfter=2),
    cell=ParagraphStyle("cell", fontName="TM", fontSize=7.8, leading=9.0,
                        alignment=TA_LEFT),
    cellh=ParagraphStyle("cellh", fontName="TM-B", fontSize=7.8, leading=9.0,
                         alignment=TA_LEFT),
    bul=ParagraphStyle("bul", fontName="TM", fontSize=10, leading=12,
                       alignment=TA_JUSTIFY, spaceAfter=1),
    ref=ParagraphStyle("ref", fontName="TM", fontSize=8, leading=9.4,
                       alignment=TA_JUSTIFY, leftIndent=13,
                       firstLineIndent=-13, spaceAfter=1.5),
)
ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
         "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII"]
FIGN, TABN, EQN = {}, {}, {}


def resolve(t):
    def f(m):
        k, i = m.group(1), m.group(2)
        if k == "f":
            return "Fig. %s" % FIGN.get(i, "?")
        return "Table %s" % TABN.get(i, "?")
    return re.sub(r"\{\{([ft]):([A-Za-z0-9_]+)\}\}", f, t)


def inline(t, esc=True):
    t = resolve(t)
    if esc:
        t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"`([^`]+)`", r'<font face="Courier" size="8">\1</font>', t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", t)
    t = re.sub(r"(\d)\^(\d+)", r"\1<super>\2</super>", t)
    if CJK != "TM":
        t = re.sub(r"([\u3000-\u9fff\uff00-\uffef]+)",
                   r'<font face="%s">\1</font>' % CJK, t)
    return t


def mk_table(caption, rows, num, width):
    head, body = rows[0], rows[1:]
    nc = len(head)
    wts = []
    for i in range(nc):
        m = max([len(head[i])] + [len(r[i]) if i < len(r) else 0 for r in body])
        wts.append(max(m, 5) ** 0.60)
    tot = sum(wts)
    widths = [width * w / tot for w in wts]
    data = [[Paragraph(inline(c), S["cellh"]) for c in head]]
    for r in body:
        r = (r + [""] * nc)[:nc]
        data.append([Paragraph(inline(c), S["cell"]) for c in r])
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="CENTER")
    t.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.9, BK),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, BK),
        ("LINEBELOW", (0, -1), (-1, -1), 0.9, BK),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.8),
        ("LEFTPADDING", (0, 0), (-1, -1), 2.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    out = [Paragraph("TABLE %s" % ROMAN[num], S["tnum"]),
           Paragraph(inline(caption).upper(), S["tcap"]), t, Spacer(1, 6)]
    return out


def mk_eq(markup, num, width):
    body = Paragraph(resolve(markup), S["eq"])
    tag = Paragraph("(%d)" % num, S["eqn"])
    t = Table([[body, tag]], colWidths=[width - 24, 24], hAlign="CENTER")
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                           ("TOPPADDING", (0, 0), (-1, -1), 3),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    return [t]


def fig_parts(path, width, num, caption):
    full = os.path.join(ROOT, path.replace("/", os.sep))
    with PILImage.open(full) as im:
        iw, ih = im.size
    w = width
    h = w * ih / iw
    if h > 560:
        h = 560; w = h * iw / ih
    img = Image(full, width=w, height=h)
    img.hAlign = "CENTER"
    cap = Paragraph("Fig. %d.&nbsp; %s" % (num, inline(caption)), S["cap"])
    return img, cap, h


def cols(top):
    h = BODYH - top
    return [Frame(LM, BM, COLW, h, id="L", leftPadding=0, rightPadding=0,
                  topPadding=0, bottomPadding=0),
            Frame(LM + COLW + GUT, BM, COLW, h, id="R", leftPadding=0,
                  rightPadding=0, topPadding=0, bottomPadding=0)]


def deco(canv, doc):
    canv.saveState()
    canv.setFont("TM", 9)
    canv.setFillColor(MUTE)
    canv.drawCentredString(PW / 2, 32, "%d" % doc.page)
    canv.restoreState()


def prescan(lines):
    f = t = e = 0
    for ln in lines:
        s2 = ln.strip()
        if s2.startswith("@FIG "):
            f += 1
            FIGN[s2.split("|")[1].strip()] = f
        elif s2.startswith("@TABLE "):
            t += 1
            TABN[s2.split("|")[0][7:].strip()] = ROMAN[t]
        elif s2.startswith("@EQ "):
            e += 1
            EQN[s2.split("|")[1].strip()] = e


def build():
    lines = io.open(SRC, encoding="utf-8").read().split("\n")
    prescan(lines)
    story, templates = [], []
    i, n = 0, len(lines)
    fign = tabn = eqn = spann = 0
    para, meta, abstract, titleblk = [], [], [], []
    mode, in_title = None, True

    def flush(first=True):
        if para:
            story.append(Paragraph(inline(" ".join(para)),
                                   S["body"] if first else S["flat"]))
            para.clear()

    while i < n:
        raw = lines[i]
        s2 = raw.strip()
        if s2.startswith("# ") and not s2.startswith("##"):
            titleblk.append(Paragraph(inline(s2[2:]), S["title"])); i += 1
        elif s2.startswith("@META "):
            meta.append(s2[6:]); i += 1
        elif s2 == "@ABSTRACT":
            mode = "abs"; i += 1
        elif s2.startswith("@KEYWORDS "):
            for k, m in enumerate(meta):
                titleblk.append(Paragraph(inline(m), S["auth"] if k == 0 else S["meta"]))
            meta = []
            story.extend(titleblk); titleblk = []
            story.append(NextPageTemplate("main"))
            story.append(FrameBreak())
            if abstract:
                story.append(Paragraph("<i>Abstract</i>&mdash;" + inline(" ".join(abstract)),
                                       S["abs"]))
                abstract = []
            story.append(Paragraph("<i>Index Terms</i>&mdash;" + inline(s2[10:]), S["abs"]))
            mode, in_title = None, False
            i += 1
        elif s2.startswith("@FIG "):
            flush()
            p = [x.strip() for x in s2[5:].split("|")]
            kind, fid, path, cap = p[0], p[1], p[2], "|".join(p[3:])
            fign += 1
            if kind == "span":
                spann += 1
                img, capp, h = fig_parts(path, SPANW, fign, cap)
                _, ch = capp.wrap(SPANW, 400)
                top = h + ch + 13
                tid = "sp%d" % spann
                templates.append(PageTemplate(
                    id=tid,
                    frames=[Frame(LM, PH - TM_ - top, SPANW, top, id="T",
                                  leftPadding=0, rightPadding=0, topPadding=0,
                                  bottomPadding=0)] + cols(top), onPage=deco))
                story += [NextPageTemplate(tid), PageBreak(), img, capp,
                          FrameBreak(), NextPageTemplate("main")]
            else:
                img, capp, h = fig_parts(path, COLW, fign, cap)
                story += [Spacer(1, 4), KeepTogether([img, capp]), Spacer(1, 7)]
            i += 1
        elif s2.startswith("@EQ "):
            flush()
            p = [x.strip() for x in s2[4:].split("|")]
            eqn += 1
            story.extend(mk_eq("|".join(p[1:]), eqn, COLW))
            i += 1
        elif s2.startswith("@TABLE "):
            flush()
            cap = s2.split("|", 1)[1].strip()
            i += 1
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                c = [x.strip() for x in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", x or "-") for x in c):
                    rows.append(c)
                i += 1
            tabn += 1
            story.extend(mk_table(cap, rows, tabn, COLW))
        elif s2.startswith("|"):
            flush()
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                c = [x.strip() for x in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", x or "-") for x in c):
                    rows.append(c)
                i += 1
            tabn += 1
            story.extend(mk_table("Anticipated questions and prepared responses",
                                  rows, tabn, COLW))
        elif s2.startswith("#### "):
            flush(); story.append(Paragraph(inline(s2[5:]), S["h3"])); i += 1
        elif s2.startswith("### "):
            flush(); story.append(Paragraph(inline(s2[4:]), S["h2"])); i += 1
        elif s2.startswith("## "):
            flush()
            story.append(Paragraph(inline(s2[3:]).upper(), S["h1"])); i += 1
        elif re.match(r"^[-*] ", s2):
            flush()
            items = []
            while i < n and re.match(r"^[-*] ", lines[i].strip()):
                items.append(ListItem(Paragraph(inline(lines[i].strip()[2:]), S["bul"]),
                                      leftIndent=12))
                i += 1
            story.append(ListFlowable(items, bulletType="bullet", start="square",
                                      bulletFontSize=3.4, leftIndent=11))
            story.append(Spacer(1, 3))
        elif re.match(r"^\d+\. ", s2):
            flush()
            items = []
            while i < n and re.match(r"^\d+\. ", lines[i].strip()):
                items.append(ListItem(Paragraph(
                    inline(re.sub(r"^\d+\. ", "", lines[i].strip())), S["bul"]),
                    leftIndent=14))
                i += 1
            story.append(ListFlowable(items, bulletType="1", leftIndent=13))
            story.append(Spacer(1, 3))
        elif re.match(r"^\[\d+\] ", s2):
            flush(); story.append(Paragraph(inline(s2), S["ref"])); i += 1
        elif s2 in ("---", ""):
            flush(); i += 1
        else:
            if mode == "abs":
                abstract.append(s2)
            elif in_title:
                titleblk.append(Paragraph(inline(s2), S["meta"]))
            else:
                para.append(s2)
            i += 1
    flush()

    doc = BaseDocTemplate(OUT, pagesize=letter, leftMargin=LM, rightMargin=RM,
                          topMargin=TM_, bottomMargin=BM,
                          title="Benchmarking Four AI Paradigms in Xiangqi",
                          author="Alma Nurul Salma (L25020007)")
    TH = 112.0
    first = PageTemplate(id="first",
                         frames=[Frame(LM, PH - TM_ - TH, SPANW, TH, id="T",
                                       leftPadding=0, rightPadding=0,
                                       topPadding=0, bottomPadding=0)] + cols(TH),
                         onPage=deco)
    main = PageTemplate(id="main", frames=cols(0), onPage=deco)
    doc.addPageTemplates([first, main] + templates)
    doc.build(story)
    print("wrote", OUT, os.path.getsize(OUT), "bytes")
    print("figures %d (span %d) | tables %d | equations %d" % (fign, spann, tabn, eqn))


if __name__ == "__main__":
    build()
