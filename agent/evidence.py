import os
import re

import requests

from agent import pdf_parser

TIMEOUT = 10


def guess_title(pages):
    if not pages:
        return ""
    for line in pages[0]["text"].splitlines():
        line = line.strip()
        if len(line) > 15:
            return line
    return ""


def crossref(title):
    r = requests.get("https://api.crossref.org/works",
                     params={"query.bibliographic": title, "rows": 3}, timeout=TIMEOUT)
    r.raise_for_status()
    out = []
    for it in r.json()["message"]["items"]:
        out.append({"title": (it.get("title") or [""])[0], "doi": it.get("DOI"),
                    "url": it.get("URL")})
    return out


def openalex(title):
    r = requests.get("https://api.openalex.org/works",
                     params={"search": title, "per_page": 3}, timeout=TIMEOUT)
    r.raise_for_status()
    return [{"title": w.get("display_name"), "year": w.get("publication_year"),
             "doi": w.get("doi"), "id": w.get("id")} for w in r.json().get("results", [])]


def github_search(query):
    headers = {"Accept": "application/vnd.github+json"}
    if os.getenv("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.getenv('GITHUB_TOKEN')}"
    q = re.sub(r"[^A-Za-z0-9 ]", " ", query)[:120]
    r = requests.get("https://api.github.com/search/repositories",
                     params={"q": q, "per_page": 3}, headers=headers, timeout=TIMEOUT)
    r.raise_for_status()
    return [{"name": i["full_name"], "url": i["html_url"], "stars": i["stargazers_count"],
             "description": i.get("description")} for i in r.json().get("items", [])]


def check_url(url):
    try:
        r = requests.head(url, allow_redirects=True, timeout=TIMEOUT)
        return r.status_code
    except Exception as e:
        return f"error: {type(e).__name__}"


def discover(pages, text):
    title = guess_title(pages)
    ev = {"title_guess": title, "crossref": [], "openalex": [],
          "repos_in_paper": [], "github_search": [], "errors": []}
    for name, fn in (("crossref", crossref), ("openalex", openalex),
                     ("github_search", github_search)):
        try:
            ev[name] = fn(title) if title else []
        except Exception as e:
            ev["errors"].append(f"{name}: {type(e).__name__}: {e}")
    for url in pdf_parser.find_github_urls(text):
        ev["repos_in_paper"].append({"url": url, "http_status": check_url(url)})
    return ev
