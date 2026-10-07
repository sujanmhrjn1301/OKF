"""
ingest.py — Constitution of Nepal → Google OKF (Open Knowledge Format) converter

Reads  :  const.pdf  (Constitution of Nepal, 202 pages)
Writes :  okf_output/
            ├── index.md                           # master manifest / TOC
            ├── preamble.md                        # constitutional preamble
            ├── parts/
            │     ├── part-01-preliminary.md        # Part-level roll-up
            │     ├── part-02-citizenship.md
            │     └── …
            ├── articles/
            │     ├── article-001.md                # Per-article granular files
            │     ├── article-002.md
            │     └── …
            └── schedules/
                  ├── schedule-01.md
                  └── …

OKF spec  : Each file is Markdown with YAML front-matter.
            Required field: `type`.
            Recommended   : title, description, tags, timestamp, resource, source.
"""

from __future__ import annotations

import os
import re
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import fitz  # PyMuPDF
except ImportError:
    sys.exit(
        "ERROR: PyMuPDF is required.  Install it with:\n"
        "  pip install PyMuPDF\n"
    )

# ── Configuration ──────────────────────────────────────────────────────────────
BASE_PROJECT_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = BASE_PROJECT_DIR / "data" / "raw_pdf" / "const.pdf"
OUTPUT_DIR = BASE_PROJECT_DIR / "data" / "okf" / "constitution"

SOURCE_TITLE = "The Constitution of Nepal"
SOURCE_DATE = "2015-09-20"
SOURCE_URL = "https://www.lawcommission.gov.np"
TIMESTAMP = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Data Models ────────────────────────────────────────────────────────────────
@dataclass
class Clause:
    """A clause or sub-clause within an Article."""
    id: str           # e.g. "(1)", "(2)(a)", "Proviso-1"
    clause_type: str  # "clause", "sub-clause", "proviso"
    number: str       # e.g. "1", "a", "Proviso"
    text: str
    parent_id: str = ""  # parent clause id for sub-clauses


@dataclass
class Article:
    number: int
    title: str
    body: str        # full raw text including sub-clauses
    part_number: int = 0
    part_title: str = ""
    page: int = 0    # 1-based PDF page number
    clauses: list[Clause] = field(default_factory=list)


@dataclass
class Part:
    number: int
    title: str
    start_page: int
    articles: list[Article] = field(default_factory=list)
    raw_text: str = ""


@dataclass
class Schedule:
    number: int
    title: str
    body: str
    page: int = 0


# ── PDF Extraction ─────────────────────────────────────────────────────────────
def extract_full_text(pdf_path: Path) -> list[tuple[int, str]]:
    """Return list of (1-based page number, page text)."""
    doc = fitz.open(str(pdf_path))
    pages = []
    for i in range(len(doc)):
        text = doc[i].get_text()
        # Strip the recurring header
        text = re.sub(r"www\.lawcommission\.gov\.np\s*\n\s*\d+\s*\n?", "", text)
        pages.append((i + 1, text))
    doc.close()
    return pages


def merge_text(pages: list[tuple[int, str]]) -> str:
    """Merge all page texts into a single document string."""
    return "\n".join(text for _, text in pages)


def build_page_offset_map(pages: list[tuple[int, str]]) -> list[tuple[int, int]]:
    """Build a list of (char_offset, page_number) for mapping text positions to pages.

    When the per-page texts are joined with '\n', each page's text starts at a
    known character offset.  Given a character position in the merged string we
    can binary-search this list to find the source page.
    """
    offsets: list[tuple[int, int]] = []
    pos = 0
    for page_num, text in pages:
        offsets.append((pos, page_num))
        pos += len(text) + 1  # +1 for the joining '\n'
    return offsets


def offset_to_page(offsets: list[tuple[int, int]], char_pos: int) -> int:
    """Return the 1-based page number for a character position in the merged text."""
    import bisect
    idx = bisect.bisect_right(offsets, (char_pos,)) - 1
    if idx < 0:
        return offsets[0][1] if offsets else 0
    return offsets[idx][1]


# ── Structure Detection ───────────────────────────────────────────────────────
PART_RE = re.compile(
    r"(?:^|\n)\s*Part-(\d+)\s*\n\s*(.+?)(?:\n|$)", re.IGNORECASE
)

# Pattern: article number on its own line (e.g. "5. \n") or inline ("5. Title:")
# We use a two-pass approach for reliability.
ARTICLE_START_RE = re.compile(
    r"(?:^|\n)(\d{1,3})\.\s*\n",
)

SCHEDULE_RE = re.compile(
    r"(?:^|\n)\s*(Schedule-(\d+))\s*\n(.*?)(?=\nSchedule-\d+|\nPart-\d+|\Z)",
    re.DOTALL | re.IGNORECASE,
)


def detect_parts(full_text: str) -> list[Part]:
    """Detect Part boundaries in the full text."""
    matches = list(PART_RE.finditer(full_text))
    parts: list[Part] = []

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        part_text = full_text[start:end]

        parts.append(
            Part(
                number=int(m.group(1)),
                title=m.group(2).strip(),
                start_page=0,
                raw_text=part_text,
            )
        )

    return parts


def detect_articles(part: Part) -> list[Article]:
    """Extract individual articles from a Part's raw text.

    Two-pass approach:
      1. Find every article-start position via `N. \n` pattern.
      2. For each start, extract the title (text before the first colon)
         and body (everything until the next article start).

    Filters out false positives (e.g. numbered list items inside
    Schedule content) by requiring a colon in a title-like position
    within the first 200 characters of the content block.
    """
    # Only look at text before any Schedule boundary within this Part's text
    text = part.raw_text
    schedule_boundary = re.search(r"\nSchedule-\d+", text)
    if schedule_boundary:
        text = text[:schedule_boundary.start()]

    starts = list(ARTICLE_START_RE.finditer(text))

    articles: list[Article] = []
    for i, m in enumerate(starts):
        num = int(m.group(1))
        # Text from just after "N. \n" to the next article or end
        content_start = m.end()
        content_end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        content = text[content_start:content_end].strip()

        if not content:
            continue

        # Real articles have the pattern "Title: (1) ..." where the colon
        # appears within the first ~200 chars (title length). Numbered list
        # items in schedules (e.g. "1.\nCooperatives") lack this pattern.
        colon_pos = content.find(":")
        if colon_pos == -1 or colon_pos > 200:
            continue  # Not a real article — skip

        title = content[:colon_pos].strip()
        body = content[colon_pos + 1:].strip()

        # Clean up multi-line title (join wrapped lines)
        title = " ".join(title.split())
        body = _clean_body(body)

        # Parse clauses and sub-clauses
        clauses = parse_clauses(body)

        articles.append(
            Article(
                number=num,
                title=title,
                body=body,
                part_number=part.number,
                part_title=part.title,
                clauses=clauses,
            )
        )

    return articles


# ── Clause / Sub-clause Parsing ────────────────────────────────────────────────
# Clause pattern: (1), (2), etc. at start of line or after whitespace
CLAUSE_RE = re.compile(r"(?:^|\n)\s*\((\d+)\)\s*")
# Sub-clause pattern: (a), (b), etc.
SUB_CLAUSE_RE = re.compile(r"(?:^|\n)\s*\(([a-z])\)\s*")
# Proviso pattern
PROVISO_RE = re.compile(r"(?:^|\n)\s*Provided\s+that[:\s]", re.IGNORECASE)


def parse_clauses(body: str) -> list[Clause]:
    """Parse clause structure from an article body.

    Detects:
      - Clauses:      (1), (2), (3), ...
      - Sub-clauses:  (a), (b), (c), ... nested under the preceding clause
      - Provisos:     'Provided that ...' blocks
    """
    clauses: list[Clause] = []

    # Find all clause starts with their positions
    clause_positions: list[tuple[int, str, str, str]] = []  # (pos, id, type, number)

    for m in CLAUSE_RE.finditer(body):
        clause_positions.append(
            (m.start(), f"({m.group(1)})", "clause", m.group(1))
        )

    for m in SUB_CLAUSE_RE.finditer(body):
        clause_positions.append(
            (m.start(), f"({m.group(1)})", "sub-clause", m.group(1))
        )

    proviso_count = 0
    for m in PROVISO_RE.finditer(body):
        proviso_count += 1
        clause_positions.append(
            (m.start(), f"Proviso-{proviso_count}", "proviso", f"Proviso-{proviso_count}")
        )

    # Sort by position in text
    clause_positions.sort(key=lambda x: x[0])

    # Extract text for each clause and determine parent relationships
    current_parent_clause = ""
    for i, (pos, cid, ctype, cnum) in enumerate(clause_positions):
        # Text runs from this position to the next clause position or end
        text_start = pos
        text_end = clause_positions[i + 1][0] if i + 1 < len(clause_positions) else len(body)
        text = body[text_start:text_end].strip()

        parent_id = ""
        if ctype == "clause":
            current_parent_clause = cid
        elif ctype == "sub-clause":
            parent_id = current_parent_clause
            cid = f"{current_parent_clause}{cid}"  # e.g. "(2)(a)"
        elif ctype == "proviso":
            parent_id = current_parent_clause

        clauses.append(
            Clause(
                id=cid,
                clause_type=ctype,
                number=cnum,
                text=text,
                parent_id=parent_id,
            )
        )

    return clauses


def detect_schedules(full_text: str) -> list[Schedule]:
    """Extract Schedule sections."""
    schedules: list[Schedule] = []
    seen: set[int] = set()

    # Find schedule content from the latter portion of the document
    # Schedules are typically at the end
    for m in SCHEDULE_RE.finditer(full_text):
        num = int(m.group(2))
        if num in seen:
            continue
        seen.add(num)

        body = m.group(3).strip()
        body = _clean_body(body)

        # Try to extract a meaningful title from the first line
        first_line = body.split("\n")[0].strip() if body else ""
        title = first_line[:120] if first_line else f"Schedule {num}"

        schedules.append(Schedule(number=num, title=title, body=body))

    return schedules


def extract_preamble(full_text: str) -> str:
    """Extract the Preamble text."""
    m = re.search(
        r"Preamble:\s*\n(.*?)(?=\n\s*Part-1)", full_text, re.DOTALL | re.IGNORECASE
    )
    if m:
        return _clean_body(m.group(1).strip())
    return ""


def _clean_body(text: str) -> str:
    """Clean extracted text: normalise whitespace, fix line breaks."""
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove trailing spaces per line
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


# ── OKF Markdown Rendering ────────────────────────────────────────────────────
def yaml_frontmatter(**fields: object) -> str:
    """Render YAML front-matter block."""
    lines = ["---"]
    for key, value in fields.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {item}")
        elif isinstance(value, str) and "\n" in value:
            lines.append(f"{key}: |")
            for l in value.split("\n"):
                lines.append(f"  {l}")
        else:
            # Quote strings that contain special YAML chars
            v = str(value)
            if any(c in v for c in ":{}[]#&*!|>',\"@`"):
                v = f'"{v}"'
            lines.append(f"{key}: {v}")
    lines.append("---")
    return "\n".join(lines)


def render_article_md(article: Article) -> str:
    """Render an Article as an OKF Markdown file."""
    # Build clause metadata for YAML
    clause_summary = []
    for c in article.clauses:
        entry = {"id": c.id, "type": c.clause_type}
        if c.parent_id:
            entry["parent"] = c.parent_id
        clause_summary.append(entry)

    fm_fields = dict(
        type="legal-provision",
        title=f"Article {article.number} -- {article.title}",
        description=f"Article {article.number} of {SOURCE_TITLE}",
        source=SOURCE_URL,
        timestamp=TIMESTAMP,
        tags=[
            "constitution",
            "nepal",
            "legal",
            f"part-{article.part_number}",
            _slugify(article.part_title),
        ],
        resource=f"const.pdf#article-{article.number}",
        part_number=article.part_number,
        part_title=article.part_title,
        article_number=article.number,
        page=article.page,
        clause_count=len([c for c in article.clauses if c.clause_type == "clause"]),
        sub_clause_count=len([c for c in article.clauses if c.clause_type == "sub-clause"]),
        proviso_count=len([c for c in article.clauses if c.clause_type == "proviso"]),
    )
    fm = yaml_frontmatter(**fm_fields)

    # Build structured body
    body = f"# Article {article.number} -- {article.title}\n\n"
    body += f"> **Part {article.part_number}:** {article.part_title}  \n"
    body += f"> **Page:** {article.page}\n\n"

    if article.clauses:
        for c in article.clauses:
            if c.clause_type == "clause":
                body += f"### Clause {c.id}\n\n"
                body += c.text + "\n\n"
            elif c.clause_type == "sub-clause":
                body += f"#### Sub-clause {c.id}\n\n"
                body += c.text + "\n\n"
            elif c.clause_type == "proviso":
                body += f"#### {c.id}\n\n"
                body += f"> {c.text}\n\n"
    else:
        # No clauses detected — output raw body
        body += article.body

    # Append clause index table
    if article.clauses:
        body += "---\n\n"
        body += "## Clause Index\n\n"
        body += "| ID | Type | Parent |\n"
        body += "|----|------|--------|\n"
        for c in article.clauses:
            body += f"| {c.id} | {c.clause_type} | {c.parent_id or '-'} |\n"

    return fm + "\n\n" + body + "\n"


def render_part_md(part: Part) -> str:
    """Render a Part as an OKF Markdown file (roll-up with article listing)."""
    article_refs = "\n".join(
        f"- [Article {a.number} — {a.title}](../articles/article-{a.number:03d}.md)"
        for a in part.articles
    )

    fm = yaml_frontmatter(
        type="legal-section",
        title=f"Part {part.number} — {part.title}",
        description=f"Part {part.number} of {SOURCE_TITLE}: {part.title}",
        source=SOURCE_URL,
        timestamp=TIMESTAMP,
        tags=["constitution", "nepal", "legal", f"part-{part.number}"],
        resource=f"const.pdf#part-{part.number}",
        part_number=part.number,
        article_count=len(part.articles),
    )

    body = f"# Part {part.number} — {part.title}\n\n"
    if part.articles:
        body += f"This part contains **{len(part.articles)} articles** "
        art_range = f"(Articles {part.articles[0].number}–{part.articles[-1].number})"
        body += art_range + ".\n\n"
        body += "## Articles\n\n"
        body += article_refs + "\n\n"
        body += "---\n\n"

    # Include the full part text for completeness
    body += "## Full Text\n\n"
    body += part.raw_text

    return fm + "\n\n" + body + "\n"


def render_schedule_md(schedule: Schedule) -> str:
    """Render a Schedule as an OKF Markdown file."""
    fm = yaml_frontmatter(
        type="legal-appendix",
        title=f"Schedule {schedule.number}",
        description=f"Schedule {schedule.number} of {SOURCE_TITLE}",
        source=SOURCE_URL,
        timestamp=TIMESTAMP,
        tags=["constitution", "nepal", "legal", "schedule"],
        resource=f"const.pdf#schedule-{schedule.number}",
        schedule_number=schedule.number,
    )

    body = f"# Schedule {schedule.number}\n\n"
    body += schedule.body

    return fm + "\n\n" + body + "\n"


def render_preamble_md(preamble_text: str) -> str:
    """Render the Preamble as an OKF Markdown file."""
    fm = yaml_frontmatter(
        type="legal-provision",
        title="Preamble",
        description=f"Preamble of {SOURCE_TITLE}",
        source=SOURCE_URL,
        timestamp=TIMESTAMP,
        tags=["constitution", "nepal", "legal", "preamble"],
        resource="const.pdf#preamble",
    )

    body = "# Preamble\n\n" + preamble_text
    return fm + "\n\n" + body + "\n"


def render_index_md(
    parts: list[Part],
    schedules: list[Schedule],
    total_articles: int,
) -> str:
    """Render the master index.md (manifest / TOC)."""
    fm = yaml_frontmatter(
        type="index",
        title=SOURCE_TITLE,
        description=f"Open Knowledge Format (OKF) bundle index for {SOURCE_TITLE}",
        source=SOURCE_URL,
        timestamp=TIMESTAMP,
        tags=["constitution", "nepal", "legal", "index", "okf"],
        resource="const.pdf",
        source_date=SOURCE_DATE,
        total_parts=len(parts),
        total_articles=total_articles,
        total_schedules=len(schedules),
    )

    body = f"# {SOURCE_TITLE}\n\n"
    body += textwrap.dedent(f"""\
        > **Source:** {SOURCE_URL}
        > **Original Date:** {SOURCE_DATE}
        > **Format:** Google Open Knowledge Format (OKF)
        > **Generated:** {TIMESTAMP}

        This bundle contains the full text of {SOURCE_TITLE}, structured as an
        OKF-compliant knowledge base for AI agent consumption.

        ---

        ## Summary

        | Metric         | Count |
        |----------------|-------|
        | Parts          | {len(parts)} |
        | Articles       | {total_articles} |
        | Schedules      | {len(schedules)} |

        ---

        ## Preamble

        - [Preamble](preamble.md)

        ---

        ## Parts

    """)

    for p in parts:
        slug = _slugify(p.title)
        body += f"### Part {p.number} — {p.title}\n\n"
        body += f"- **Overview:** [part-{p.number:02d}-{slug}.md](parts/part-{p.number:02d}-{slug}.md)\n"
        if p.articles:
            body += f"- **Articles:** {p.articles[0].number}–{p.articles[-1].number} ({len(p.articles)} articles)\n"
            for a in p.articles:
                body += f"  - [Article {a.number} — {a.title}](articles/article-{a.number:03d}.md)\n"
        body += "\n"

    body += "---\n\n## Schedules\n\n"
    for s in schedules:
        body += f"- [Schedule {s.number}](schedules/schedule-{s.number:02d}.md)\n"

    return fm + "\n\n" + body + "\n"


# ── Utilities ──────────────────────────────────────────────────────────────────
def _slugify(text: str) -> str:
    """Convert text to a URL-friendly slug."""
    slug = text.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ── Main Pipeline ─────────────────────────────────────────────────────────────
def main() -> None:
    print(f"{'=' * 60}")
    print(f"  Constitution of Nepal -> Google OKF Converter")
    print(f"{'=' * 60}")
    print()

    if not PDF_PATH.exists():
        sys.exit(f"ERROR: PDF not found at {PDF_PATH}")

    # ── Step 1: Extract text ──
    print("[1/6] Extracting text from PDF...")
    pages = extract_full_text(PDF_PATH)
    full_text = merge_text(pages)
    print(f"       -> {len(pages)} pages, {len(full_text):,} characters")

    # ── Step 2: Extract preamble ──
    print("[2/6] Extracting Preamble...")
    preamble = extract_preamble(full_text)
    print(f"       -> {len(preamble):,} characters")

    # ── Step 2b: Build page offset map ──
    page_offsets = build_page_offset_map(pages)

    # ── Step 3: Detect parts & articles ──
    print("[3/6] Detecting Parts and Articles...")
    parts = detect_parts(full_text)
    total_articles = 0
    total_clauses = 0
    for part in parts:
        # Assign page number to the Part
        part_match = re.search(rf"Part-{part.number}\s*\n", full_text)
        if part_match:
            part.start_page = offset_to_page(page_offsets, part_match.start())

        part.articles = detect_articles(part)

        # Assign page numbers to articles by finding their position in full_text
        for article in part.articles:
            # Search for the article's distinctive pattern in full_text
            # Pattern: "N. \n Title:" with possible whitespace variations
            art_match = re.search(
                rf"(?:^|\n){article.number}\.\s*\n\s*{re.escape(article.title[:30])}",
                full_text,
            )
            if art_match:
                article.page = offset_to_page(page_offsets, art_match.start())
            else:
                # Fallback: search for just the title text
                title_pos = full_text.find(f"{article.title[:50]}:")
                if title_pos >= 0:
                    article.page = offset_to_page(page_offsets, title_pos)
                else:
                    # Last resort: use part's start page
                    article.page = part.start_page

            total_clauses += len(article.clauses)

        total_articles += len(part.articles)
    print(f"       -> {len(parts)} parts, {total_articles} articles, {total_clauses} clauses")

    # ── Step 4: Detect schedules ──
    print("[4/6] Detecting Schedules...")
    schedules = detect_schedules(full_text)
    print(f"       -> {len(schedules)} schedules")

    # ── Step 5: Write OKF output ──
    print("[5/6] Writing OKF bundle...")
    ensure_dir(OUTPUT_DIR / "parts")
    ensure_dir(OUTPUT_DIR / "articles")
    ensure_dir(OUTPUT_DIR / "schedules")

    files_written = 0

    # Preamble
    preamble_path = OUTPUT_DIR / "preamble.md"
    preamble_path.write_text(render_preamble_md(preamble), encoding="utf-8")
    files_written += 1

    # Parts
    for part in parts:
        slug = _slugify(part.title)
        fname = f"part-{part.number:02d}-{slug}.md"
        (OUTPUT_DIR / "parts" / fname).write_text(
            render_part_md(part), encoding="utf-8"
        )
        files_written += 1

    # Articles — deduplicate by article number (first occurrence wins,
    # since constitutional articles are numbered sequentially across Parts;
    # later collisions are false positives from numbered lists inside Parts)
    seen_articles: set[int] = set()
    for part in parts:
        for article in part.articles:
            if article.number in seen_articles:
                continue  # skip duplicate
            seen_articles.add(article.number)
            fname = f"article-{article.number:03d}.md"
            (OUTPUT_DIR / "articles" / fname).write_text(
                render_article_md(article), encoding="utf-8"
            )
            files_written += 1

    # Schedules
    for schedule in schedules:
        fname = f"schedule-{schedule.number:02d}.md"
        (OUTPUT_DIR / "schedules" / fname).write_text(
            render_schedule_md(schedule), encoding="utf-8"
        )
        files_written += 1

    # ── Step 6: Write index ──
    print("[6/6] Writing index.md (manifest)...")
    index_path = OUTPUT_DIR / "index.md"
    index_path.write_text(
        render_index_md(parts, schedules, total_articles), encoding="utf-8"
    )
    files_written += 1

    # ── Summary ──
    print()
    print(f"{'=' * 60}")
    print(f"  [OK]  OKF bundle written to: {OUTPUT_DIR}")
    print(f"      Total files: {files_written}")
    print(f"{'=' * 60}")
    print()
    print("  Bundle structure:")
    print(f"    {OUTPUT_DIR}/")
    print(f"    |-- index.md")
    print(f"    |-- preamble.md")
    print(f"    |-- parts/          ({len(parts)} files)")
    print(f"    |-- articles/       ({total_articles} files)")
    print(f"    +-- schedules/      ({len(schedules)} files)")
    print()


if __name__ == "__main__":
    main()
