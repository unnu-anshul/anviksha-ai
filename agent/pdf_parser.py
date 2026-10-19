import re

import pymupdf


def load(pdf_bytes):
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    pages, tables = [], []
    for i, page in enumerate(doc, start=1):
        pages.append({"page": i, "text": page.get_text()})
        try:
            found = page.find_tables()
            for j, t in enumerate(found.tables, start=1):
                rows = [[(c or "").strip() for c in r] for r in t.extract()]
                if rows:
                    tables.append({"id": f"P{i}-T{j}", "page": i, "rows": rows})
        except Exception:
            pass  # table detection is best-effort
    doc.close()
    return pages, tables


def full_text(pages):
    return "\n".join(f"[page {p['page']}]\n{p['text']}" for p in pages)


def tables_as_text(tables):
    out = []
    for t in tables:
        out.append(f"TABLE {t['id']} (page {t['page']})")
        for r in t["rows"]:
            out.append(" | ".join(r))
        out.append("")
    return "\n".join(out)


def find_github_urls(text):
    urls = re.findall(r"https?://github\.com/[\w.-]+/[\w.-]+", text)
    return sorted({u.rstrip(".,);") for u in urls})
