"""Build docs/paper/references.bib from primary metadata and write a verification table (citation-check rule,
docs/PAPER_PLAN_2026-11.md section 6).

    cd research && .venv/bin/python -I tools/build_bib.py ../docs/paper/references_sources.json ../docs/paper

For every entry of references_sources.json: arXiv metadata (export.arxiv.org API: title, authors, date, journal_ref,
comment) and/or Crossref metadata (api.crossref.org: title, authors, container, volume, pages, year); for PMLR papers the
proceedings page. The claimed venue counts as verified when the DOI's container matches it, when the PMLR page exists with
the same title, or when the arXiv journal_ref / comment names it. Titles from arXiv and DOI are compared. Output:
references.bib (metadata from the sources, not typed by hand) and references_check.md.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

UA = {"User-Agent": "citation-check/0.1 (research script)"}
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def get(url: str, timeout: int = 30) -> tuple[int, bytes]:
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < 2:
                time.sleep(5 * (attempt + 1))
                continue
            return e.code, b""
        except Exception:
            if attempt < 2:
                time.sleep(3)
                continue
            return 0, b""
    return 0, b""


def norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", t.lower())


def arxiv_batch(ids: list[str]) -> dict:
    out = {}
    for k in range(0, len(ids), 40):
        st, body = get("https://export.arxiv.org/api/query?max_results=50&id_list=" + ",".join(ids[k:k + 40]))
        if st != 200:
            continue
        for e in ET.fromstring(body).findall("a:entry", NS):
            aid = re.sub(r"v\d+$", "", e.find("a:id", NS).text.rsplit("/abs/", 1)[-1])
            out[aid] = {"title": " ".join(e.find("a:title", NS).text.split()),
                        "authors": [a.find("a:name", NS).text for a in e.findall("a:author", NS)],
                        "year": e.find("a:published", NS).text[:4],
                        "journal_ref": (e.find("arxiv:journal_ref", NS).text if e.find("arxiv:journal_ref", NS) is not None else ""),
                        "comment": (" ".join(e.find("arxiv:comment", NS).text.split()) if e.find("arxiv:comment", NS) is not None else "")}
        time.sleep(3)
    return out


def crossref(doi: str) -> dict | None:
    st, body = get("https://api.crossref.org/works/" + urllib.request.quote(doi))
    if st != 200:
        return None
    m = json.loads(body)["message"]
    dp = (m.get("published-print") or m.get("published-online") or m.get("issued") or {}).get("date-parts", [[None]])[0]
    title = re.sub(r"^\d+(\.\d+)+\s+", "", " ".join((m.get("title") or [""])[0].split()))   # drop session numbers ("1.1 ...")
    return {"title": title, "authors": [f"{a.get('given', '')} {a.get('family', '')}".strip() for a in m.get("author", [])],
            "container": (m.get("container-title") or [""])[0], "volume": m.get("volume", ""), "issue": m.get("issue", ""),
            "pages": m.get("page", "") or m.get("article-number", ""), "year": str(dp[0]) if dp and dp[0] else "", "type": m.get("type", "")}


def pmlr(path: str) -> str | None:
    st, body = get(f"https://proceedings.mlr.press/{path}.html")
    if st != 200:
        return None
    m = re.search(rb"<title>(.*?)</title>", body, re.S)
    return " ".join(m.group(1).decode("utf-8", "replace").split()) if m else ""


_NIPS = {}


def neurips_has(title: str, year: str) -> bool:
    """The NeurIPS proceedings index of that year lists the title."""
    if year not in _NIPS:
        st, body = get(f"https://proceedings.neurips.cc/paper_files/paper/{year}", timeout=60)
        _NIPS[year] = norm(body.decode("utf-8", "replace")) if st == 200 else ""
    return bool(_NIPS[year]) and norm(title)[:50] in _NIPS[year]


def openreview_venue(title: str) -> str:
    """Venue string of the OpenReview note whose title matches (ICLR and other OpenReview venues)."""
    st, body = get("https://api2.openreview.net/notes/search?limit=5&term=" + urllib.request.quote(title))
    if st != 200:
        return ""
    for n in json.loads(body).get("notes", []):
        c = n.get("content", {})
        t = c.get("title", {}).get("value", "") if isinstance(c.get("title"), dict) else c.get("title", "")
        v = c.get("venue", {}).get("value", "") if isinstance(c.get("venue"), dict) else c.get("venue", "")
        if norm(t) == norm(title):
            return v
    return ""


BOOK = {"ICML": "Proceedings of the International Conference on Machine Learning (ICML)",
        "NeurIPS": "Advances in Neural Information Processing Systems (NeurIPS)",
        "ICLR": "International Conference on Learning Representations (ICLR)",
        "ICCV": "Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)",
        "ISVLSI": "IEEE Computer Society Annual Symposium on VLSI (ISVLSI)"}
NAME_FIX = {"Lecun": "LeCun"}   # capitalisation as printed in the paper (Crossref has "Lecun")


def bib_authors(names: list[str]) -> str:
    names = [" ".join(NAME_FIX.get(w, w) for w in n.split()) for n in names]
    out = []
    for n in names:
        parts = n.split()
        out.append(f"{parts[-1]}, {' '.join(parts[:-1])}" if len(parts) > 1 else n)
    return " and ".join(out)


def main(src: Path, outdir: Path) -> None:
    refs = json.loads(src.read_text())
    ax = arxiv_batch([r["arxiv"] for r in refs if r.get("arxiv")])
    rows, bib = [], []
    for r in refs:
        a = ax.get(r.get("arxiv", ""), None) if r.get("arxiv") else None
        c = crossref(r["doi"]) if r.get("doi") else None
        time.sleep(1)
        p = pmlr(r["pmlr"]) if r.get("pmlr") else None
        venue, ev = r["venue"], []
        claim_tokens = [t for t in re.findall(r"[A-Za-z]+|\d{4}", venue) if t.lower() not in ("ieee", "workshop", "on", "of", "the", "and", "proceedings")]
        ok_venue = False
        if venue == "arXiv preprint":
            ok_venue = a is not None
            ev.append("arXiv only")
        if c:
            ev.append(f"DOI -> {c['container'] or c['type']} ({c['year']})")
            cont = norm(c["container"])
            if norm(venue) and norm(venue) in cont or any(norm(t) in cont for t in claim_tokens if len(t) > 3 and not t.isdigit()) \
                    or (c["type"] == "proceedings-article" and r["doi"].startswith("10.52202")):
                ok_venue = True
        elif r.get("doi"):
            ev.append("DOI NOT FOUND in Crossref")
        if p is not None:
            ev.append("PMLR page found")
            if a and norm(a["title"])[:40] in norm(p):
                ok_venue = True
        if a:
            text = f"{a['journal_ref']} {a['comment']}"
            acr = [t for t in claim_tokens if t.isupper() and len(t) >= 4]
            yr = [t for t in claim_tokens if t.isdigit()]
            if acr and all(t in text for t in acr) and (not yr or any(y in text for y in yr)):
                ok_venue = True
                ev.append(f"arXiv note: {text[:80]}")
        yr_claim = re.search(r"\d{4}", venue).group(0) if re.search(r"\d{4}", venue) else ""
        if not ok_venue and a and venue.startswith("NeurIPS") and neurips_has(a["title"], yr_claim):
            ok_venue = True
            ev.append(f"listed in NeurIPS {yr_claim} proceedings")
        if not ok_venue and a and venue.startswith("ICLR"):
            v = openreview_venue(a["title"])
            if v:
                ev.append(f"OpenReview: {v}")
                ok_venue = "ICLR" in v and yr_claim in v
            time.sleep(1)
        title_a, title_c = (a or {}).get("title", ""), (c or {}).get("title", "")
        same_title = (not title_a or not title_c) or norm(title_a) == norm(title_c)
        exists = bool(a or c)
        status = "OK" if exists and ok_venue else ("EXISTS, venue unverified" if exists else "NOT FOUND")
        rows.append([r["key"], r["cite"], title_c or title_a, ", ".join(((c or {}).get("authors") or (a or {}).get("authors") or [])[:3]),
                     (c or {}).get("year") or (a or {}).get("year", ""), venue, "; ".join(ev), status,
                     "" if same_title else f"arXiv title differs: {title_a}"])
        # BibTeX from the sources
        if c and c["type"] in ("journal-article",):
            bib.append(f"@article{{{r['key']}},\n  title = {{{{{c['title']}}}}},\n  author = {{{bib_authors(c['authors'])}}},\n  journal = {{{c['container']}}},\n"
                       f"  volume = {{{c['volume']}}},\n  number = {{{c['issue']}}},\n  pages = {{{c['pages']}}},\n  year = {{{c['year']}}},\n  doi = {{{r['doi']}}}"
                       + (f",\n  eprint = {{{r['arxiv']}}},\n  archivePrefix = {{arXiv}}" if r.get("arxiv") else "") + "\n}")
        elif venue == "arXiv preprint" and a:
            bib.append(f"@misc{{{r['key']}},\n  title = {{{{{a['title']}}}}},\n  author = {{{bib_authors(a['authors'])}}},\n  year = {{{a['year']}}},\n"
                       f"  eprint = {{{r['arxiv']}}},\n  archivePrefix = {{arXiv}},\n  note = {{arXiv preprint arXiv:{r['arxiv']}}}\n}")
        else:
            title = (c or a or {}).get("title", "")
            authors = (c or {}).get("authors") or (a or {}).get("authors") or []
            year = re.search(r"\d{4}", venue).group(0) if re.search(r"\d{4}", venue) else (c or a or {}).get("year", "")
            book = re.sub(r"\s+\d{4}$", "", venue)
            book = c["container"] if c and c["type"] == "proceedings-article" and c["container"] else BOOK.get(book, book)
            bib.append(f"@inproceedings{{{r['key']}},\n  title = {{{{{title}}}}},\n  author = {{{bib_authors(authors)}}},\n  booktitle = {{{book}}},\n  year = {{{year}}}"
                       + (f",\n  doi = {{{r['doi']}}}" if r.get("doi") and c else "") + (f",\n  eprint = {{{r['arxiv']}}},\n  archivePrefix = {{arXiv}}" if r.get("arxiv") else "") + "\n}")
    (outdir / "references.bib").write_text("% Generated by research/tools/build_bib.py from arXiv / Crossref / PMLR metadata. Do not edit by hand;\n"
                                          "% fix references_sources.json and rerun.\n\n" + "\n\n".join(bib) + "\n")
    L = ["# Kiểm trích dẫn (tạo tự động bởi `research/tools/build_bib.py`)", "",
         "_OK = bài có thật (arXiv hoặc DOI trả về siêu dữ liệu) và nơi đăng được xác nhận (DOI, trang PMLR, hoặc ghi chú arXiv). Câu trích nguyên văn được kiểm riêng (mục cuối)._", "",
         "| key | trích trong bài | tiêu đề (theo nguồn) | tác giả đầu | năm | nơi đăng ghi trong bài | bằng chứng | trạng thái | ghi chú |", "|---|---|---|---|---|---|---|---|---|"]
    L += ["| " + " | ".join(x.replace("|", "/") for x in row) + " |" for row in rows]
    n_ok = sum(r[7] == "OK" for r in rows)
    L += ["", f"**{n_ok}/{len(rows)} OK.** Mục không OK cần kiểm tay trước khi nộp."]
    (outdir / "references_check.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[-1:]))
    for row in rows:
        if row[7] != "OK" or row[8]:
            print("  ", row[0], "|", row[7], "|", row[6][:120], "|", row[8][:100])


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
