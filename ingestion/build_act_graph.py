"""
build_act_graph.py -- Universal Act Cross-Reference Graph Builder for Nepal Acts

Parses all OKF section markdown files for any Act of Nepal (e.g. Criminal Procedure,
Penal Code, Civil Code) and extracts multi-dimensional legal connections:
  1. Other statutory Sections (e.g., Section 16, Sections 20 and 21)
  2. Legal Schedules (e.g., Schedule-1, Schedule-2, Schedule-12, Schedule-13)
  3. Internal Sub-sections (e.g., Sub-section (1), Sub-section (3) of Section 10)
  4. Constitutional Articles (e.g., Section 96 citing Articles 134 and 145)

Produces a clean, structured act_graph.json inside that Act's directory.

Usage:
    python ingestion/build_act_graph.py                          # Defaults to criminal_procedure
    python ingestion/build_act_graph.py --act criminal_procedure # Explicit act name
    python ingestion/build_act_graph.py --all                    # Process all acts in data/okf

Output:
    data/okf/<act_name>/act_graph.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

BASE_PROJECT_DIR = Path(__file__).resolve().parent.parent
OKF_ROOT = BASE_PROJECT_DIR / "data" / "okf"

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
TITLE_RE = re.compile(r"^\s*#\s*Section\s*\d+.*?\n", re.IGNORECASE)


def extract_section_number(filepath: Path) -> int | None:
    """Extract the section number from filename (e.g., section-018.md -> 18)."""
    m = re.search(r"section-(\d+)", filepath.stem, re.IGNORECASE)
    return int(m.group(1)) if m else None


def extract_body(text: str) -> str:
    """Strip YAML frontmatter, top title, and the Act/Chapter/Page metadata header."""
    m = FRONTMATTER_RE.match(text)
    body = text[m.end():] if m else text
    body = TITLE_RE.sub("", body)
    body = re.sub(r"^\s*\*\*Act:\*\*.*?$", "", body, flags=re.MULTILINE | re.IGNORECASE)
    body = re.sub(r"^\s*\*\*Chapter\s*\d+:\*\*.*?$", "", body, flags=re.MULTILINE | re.IGNORECASE)
    body = re.sub(r"^\s*\*Page\s*\d+\*\s*$", "", body, flags=re.MULTILINE | re.IGNORECASE)
    # Structural list labels like "- **Clause (a)**:" / "- **Sub-clause (1)**:" are layout, not citations
    body = re.sub(r"^\s*-\s*\*\*\s*(?:Sub-)?Clause\s*\([^)]*\)\s*\*\*\s*:?", "", body, flags=re.MULTILINE | re.IGNORECASE)
    return body


def extract_provision_references(
    body: str,
    self_section: int,
    max_section: int,
    max_schedule: int = 50,
) -> dict[str, list[int]]:
    """Extract sections, schedules, internal sub-sections, and constitutional articles."""
    # De-hyphenate line breaks (e.g. sub-\nsection -> sub-section, Schedule-\n12 -> Schedule-12)
    text = re.sub(r"sub-\s*\n\s*section", "sub-section", body, flags=re.IGNORECASE)
    text = re.sub(r"Schedule-\s*\n\s*(\d+)", r"Schedule-\1", text, flags=re.IGNORECASE)

    # Strip structural sub-section markdown headings (### Sub-section (1), etc.)
    clean_body = re.sub(r"###\s*Sub-section\s*\(\d+\)", "", text, flags=re.IGNORECASE)

    # 1. External Section references (e.g., "Section 16", "Sections 14 and 15")
    sec_matches = re.findall(
        r"(?:Section|Sections|Sec\.|Secs\.)\s*(\d{1,4}(?:\s*(?:,|or|and|/)\s*\d{1,4})*)",
        clean_body,
        re.IGNORECASE,
    )
    secs = sorted({
        int(n) for m in sec_matches for n in re.findall(r"\d{1,4}", m)
        if int(n) != self_section and 1 <= int(n) <= max_section
    })

    # 2. Schedule references (e.g., "Schedule-1", "Schedule-2", "Schedule-12")
    sched_matches = re.findall(
        r"(?:Schedule|Schedules|Sched\.)\s*-?\s*(\d{1,2}(?:\s*(?:,|or|and|/)\s*\d{1,2})*)",
        clean_body,
        re.IGNORECASE,
    )
    scheds = sorted({
        int(n) for m in sched_matches for n in re.findall(r"\d{1,2}", m)
        if 1 <= int(n) <= max_schedule
    })

    # 3. Pinpoint citations "sub-section (5) of Section 16" -> "16(5)"
    #    Captured first and removed so (5) isn't mistaken for this Section's own sub-section.
    pinpoint_re = re.compile(
        r"sub-sections?\s*((?:\(\d{1,2}\)\s*(?:,|or|and)?\s*)+)of\s+Section\s+(\d{1,4})",
        re.IGNORECASE,
    )
    section_subsections: set[str] = set()
    for m in pinpoint_re.finditer(clean_body):
        tgt = int(m.group(2))
        if tgt == self_section or not (1 <= tgt <= max_section):
            continue
        for s in re.findall(r"\d{1,2}", m.group(1)):
            section_subsections.add(f"{tgt}({int(s)})")
    body_no_ext = pinpoint_re.sub(" ", clean_body)

    # 4. Internal Sub-sections referenced (e.g., "pursuant to sub-section (1)")
    subsecs = sorted({
        int(n)
        for grp in re.findall(r"sub-sections?\s*((?:\(\d{1,2}\)\s*(?:,|or|and)?\s*)+)", body_no_ext, re.IGNORECASE)
        for n in re.findall(r"\d{1,2}", grp)
    })

    # 5. Internal clauses (e.g., "clause (a) of sub-section (1)") -- excludes constitutional
    #    "clause (1) of Article N" which uses numeric clauses
    clauses = sorted({
        c.lower() for c in re.findall(r"(?<!sub-)clauses?\s*\(([a-z])\)", body_no_ext, re.IGNORECASE)
    })

    # 6. Chapter references (e.g., "Chapter 2")
    chapters = sorted({
        int(n) for n in re.findall(r"\bChapters?\s*-?\s*(\d{1,2})\b", clean_body, re.IGNORECASE)
    })

    # 7. Constitutional Articles (e.g., Section 96 citing "Article 134", "Article 145")
    art_matches = re.findall(
        r"(?:Article|Articles|Art\.)\s*(\d{1,3}(?:\s*(?:,|or|and|/)\s*\d{1,3})*)",
        clean_body,
        re.IGNORECASE,
    )
    arts = sorted({
        int(n) for m in art_matches for n in re.findall(r"\d{1,3}", m)
        if 1 <= int(n) <= 308
    })
    # Pinpoint constitutional citations "clause (1) of Article 134" -> "134(1)"
    const_clauses = sorted({
        f"{int(a)}({int(c)})"
        for cs, a in re.findall(
            r"clauses?\s*((?:\(\d{1,2}\)\s*(?:,|or|and)?\s*)+)of\s+Article\s+(\d{1,3})",
            clean_body,
            re.IGNORECASE,
        )
        for c in re.findall(r"\d{1,2}", cs)
    })

    result: dict[str, list] = {}
    if secs:
        result["sections"] = secs
    if section_subsections:
        result["section_subsections"] = sorted(
            section_subsections, key=lambda s: (int(s.split("(")[0]), int(s.split("(")[1][:-1]))
        )
    if scheds:
        result["schedules"] = scheds
    if subsecs:
        result["sub_sections"] = subsecs
    if clauses:
        result["clauses"] = clauses
    if chapters:
        result["chapters"] = chapters
    if arts:
        result["constitution_articles"] = arts
    if const_clauses:
        result["constitution_article_clauses"] = const_clauses

    return result


def build_graph_for_act(act_dir: Path) -> dict[str, dict[str, list[int]]]:
    """Build the multi-dimensional cross-reference graph for a given Act directory."""
    sections_dir = act_dir / "sections"
    if not sections_dir.exists():
        raise FileNotFoundError(f"Sections directory not found at: {sections_dir}")

    section_files = sorted(sections_dir.glob("section-*.md"))
    if not section_files:
        raise ValueError(f"No section-*.md files found in {sections_dir}")

    # Determine maximum section and schedule numbers
    existing_nums = set()
    for f in section_files:
        num = extract_section_number(f)
        if num is not None:
            existing_nums.add(num)
    max_section = max(existing_nums) if existing_nums else 500

    schedules_dir = act_dir / "schedules"
    max_schedule = 50
    if schedules_dir.exists():
        sched_nums = [
            int(m.group(1)) for f in schedules_dir.glob("schedule-*.md")
            if (m := re.search(r"schedule-(\d+)", f.stem))
        ]
        if sched_nums:
            max_schedule = max(sched_nums)

    print(f"\n[ACT GRAPH] Processing Act: '{act_dir.name}'")
    print(f"            Sections directory: {sections_dir}")
    print(f"            Found {len(section_files)} section files (Max Section: {max_section})")
    if schedules_dir.exists():
        print(f"            Schedules directory: {schedules_dir} ({len(list(schedules_dir.glob('schedule-*.md')))} schedules)")

    graph: dict[str, dict[str, list]] = {}
    totals: dict[str, int] = {}

    for filepath in section_files:
        sec_num = extract_section_number(filepath)
        if sec_num is None:
            continue

        try:
            text = filepath.read_text(encoding="utf-8")
        except Exception as e:
            print(f"  WARNING: Could not read {filepath.name}: {e}")
            continue

        body = extract_body(text)
        refs = extract_provision_references(body, sec_num, max_section, max_schedule)

        if refs:
            graph[str(sec_num)] = refs
            for k, v in refs.items():
                totals[k] = totals.get(k, 0) + len(v)

    output_path = act_dir / "act_graph.json"
    output_path.write_text(
        json.dumps(graph, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"\n[ACT GRAPH] Extraction Summary for '{act_dir.name}':")
    print(f"   * Total sections with references: {len(graph)} / {len(section_files)}")
    for k, v in totals.items():
        print(f"   * {k:<30} {v}")
    print(f"[ACT GRAPH] Saved graph to: {output_path}")

    # Show sample output
    print(f"\n--- Key Verification Samples ---")
    for sample_id in ["10", "18", "96"]:
        if sample_id in graph:
            print(f"  Section {sample_id}: {json.dumps(graph[sample_id])}")

    return graph


def discover_acts(root_dir: Path) -> list[Path]:
    """Find all Act directories that contain a 'sections' folder, excluding constitution."""
    acts = []
    if not root_dir.exists():
        return acts

    for item in root_dir.iterdir():
        if item.is_dir() and item.name.lower() != "constitution":
            if (item / "sections").exists():
                acts.append(item)
            elif item.name.lower() == "acts":
                for sub in item.iterdir():
                    if sub.is_dir() and (sub / "sections").exists():
                        acts.append(sub)
    return acts


def main():
    parser = argparse.ArgumentParser(
        description="Universal Act Cross-Reference Graph Builder for Nepal Acts"
    )
    parser.add_argument(
        "--act",
        type=str,
        default=None,
        help="Name of the specific Act to build graph for (e.g. criminal_procedure)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all available statutory Acts discovered in data/okf/",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Universal Act Cross-Reference Graph Builder")
    print("=" * 60)

    target_acts: list[Path] = []

    if args.act:
        direct_path = OKF_ROOT / args.act
        acts_subpath = OKF_ROOT / "acts" / args.act
        if direct_path.exists() and (direct_path / "sections").exists():
            target_acts.append(direct_path)
        elif acts_subpath.exists() and (acts_subpath / "sections").exists():
            target_acts.append(acts_subpath)
        else:
            sys.exit(f"[ERROR] Could not find Act directory with 'sections/' for '{args.act}'")
    else:
        discovered = discover_acts(OKF_ROOT)
        if not discovered:
            sys.exit("[ERROR] No statutory Acts found with 'sections/' in data/okf/.")
        if args.all:
            target_acts = discovered
        else:
            cp = [a for a in discovered if a.name == "criminal_procedure"]
            target_acts = cp if cp else [discovered[0]]

    for act_dir in target_acts:
        build_graph_for_act(act_dir)

    print("\n[DONE] Act graph generation completed successfully.")


if __name__ == "__main__":
    main()
