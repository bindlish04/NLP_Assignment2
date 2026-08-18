"""
Task 0: Domain document collection.

Downloads publicly available Wikipedia articles for the climate domain and saves
them as plain-text files. Cached files in data/documents/ allow fully offline
re-runs without hitting the Wikipedia API again.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Iterable

import requests

from .config import (
    MIN_DOCUMENTS,
    MIN_WORDS_PER_DOCUMENT,
    WIKIPEDIA_REQUEST_DELAY_SEC,
)

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
# Wikipedia requires a descriptive User-Agent; replace contact email before submission.
USER_AGENT = "NLP-PS4-Assignment/1.0 (Educational; contact: student@example.com)"

# Curated list of climate-related Wikipedia article titles (all public sources).
CLIMATE_ARTICLE_TITLES = [
    "Climate change",
    "Global warming",
    "Greenhouse effect",
    "Carbon dioxide in Earth's atmosphere",
    "Methane emissions",
    "Kyoto Protocol",
    "Paris Agreement",
    "Intergovernmental Panel on Climate Change",
    "Renewable energy",
    "Solar power",
    "Wind power",
    "Fossil fuel",
    "Deforestation",
    "Sea level rise",
    "Ocean acidification",
    "Extreme weather",
    "Carbon capture and storage",
    "Electric vehicle",
    "Climate change mitigation",
    "Climate change adaptation",
    "IPCC Sixth Assessment Report",
    "El Nino-Southern Oscillation",
    "Arctic sea ice decline",
    "Amazon rainforest",
    "Carbon footprint",
    "Emissions trading",
    "Climate justice",
    "Climate change and agriculture",
    "Climate change feedback",
    "Attribution of recent climate change",
    "Climate model",
    "Paleoclimatology",
    "Effects of climate change on agriculture",
    "Urban heat island",
    "Climate change in India",
    "Climate emergency",
    "Climate variability and change",
    "Climate change denial",
]


def _clean_wikipedia_text(text: str) -> str:
    """Remove citation markers like [12] and collapse extra whitespace."""
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fetch_wikipedia_article(title: str, session: requests.Session) -> dict:
    """Fetch one article's plain-text extract and canonical URL via the MediaWiki API."""
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts|info",
        "explaintext": True,
        "titles": title,
        "inprop": "url",
    }
    response = session.get(WIKIPEDIA_API, params=params, timeout=30)
    response.raise_for_status()
    page = next(iter(response.json()["query"]["pages"].values()))
    if "missing" in page:
        raise ValueError(f"Article not found: {title}")

    extract = _clean_wikipedia_text(page.get("extract", ""))
    return {
        "title": page.get("title", title),
        "url": page.get("fullurl", f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"),
        "text": extract,
        "word_count": len(extract.split()),
    }


def collect_documents(
    output_dir: Path,
    titles: Iterable[str] | None = None,
    min_documents: int = MIN_DOCUMENTS,
) -> list[dict]:
    """
    Download articles and write one .txt file per document.

    Each file stores Title, Source URL, and body text so provenance is preserved
    for the assignment report.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    titles = list(titles or CLIMATE_ARTICLE_TITLES)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    documents: list[dict] = []
    for title in titles:
        try:
            doc = fetch_wikipedia_article(title, session)
            if doc["word_count"] < MIN_WORDS_PER_DOCUMENT:
                continue

            slug = re.sub(r"[^a-zA-Z0-9]+", "_", doc["title"]).strip("_").lower()
            file_path = output_dir / f"{slug}.txt"
            header = f"Title: {doc['title']}\nSource: {doc['url']}\n\n"
            file_path.write_text(header + doc["text"], encoding="utf-8")

            doc["file_name"] = file_path.name
            doc["file_path"] = str(file_path)
            documents.append(doc)
            time.sleep(WIKIPEDIA_REQUEST_DELAY_SEC)  # polite rate limiting
        except Exception as exc:  # noqa: BLE001 - skip failed titles, continue collecting
            print(f"Skipping '{title}': {exc}")

    (output_dir / "documents_metadata.json").write_text(
        json.dumps(documents, indent=2), encoding="utf-8"
    )

    if len(documents) < min_documents:
        raise RuntimeError(
            f"Collected only {len(documents)} documents; need at least {min_documents}."
        )
    return documents


def load_local_documents(documents_dir: Path) -> list[dict]:
    """Load previously saved .txt files (preferred path for grading / offline use)."""
    documents: list[dict] = []
    for path in sorted(documents_dir.glob("*.txt")):
        content = path.read_text(encoding="utf-8")
        title_match = re.search(r"^Title:\s*(.+)$", content, re.MULTILINE)
        source_match = re.search(r"^Source:\s*(.+)$", content, re.MULTILINE)
        body = content.split("\n\n", 2)[-1]
        documents.append(
            {
                "title": title_match.group(1) if title_match else path.stem,
                "url": source_match.group(1) if source_match else "",
                "text": body,
                "file_name": path.name,
                "file_path": str(path),
                "word_count": len(body.split()),
            }
        )
    return documents
