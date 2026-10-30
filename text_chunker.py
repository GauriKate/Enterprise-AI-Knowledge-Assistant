
import re


def is_section_heading(line: str) -> bool:
    """Detect common heading formats without fixed heading keywords."""

    line = re.sub(r"\s+", " ", line).strip()

    if not line or len(line) > 120:
        return False

    # Examples:
    # I. INTRODUCTION
    # III. METHODOLOGY
    # 1. Introduction
    # 2.1 Dataset Description
    # A. Data Collection
    numbered_pattern = (
        r"^(?:"
        r"[IVXLCDM]+"
        r"|\d+(?:\.\d+)*"
        r"|[A-Z]"
        r")[.)]\s+\S.{0,100}$"
    )

    if re.match(numbered_pattern, line, flags=re.IGNORECASE):
        return True

    # Recognize headings written entirely in uppercase.
    # Avoid treating table and figure captions as headings.
    if (
        len(line.split()) >= 2
        and line.isupper()
        and re.search(r"[A-Z]", line)
        and not re.match(
            r"^(TABLE|FIG\.|FIGURE)\s*\d*",
            line,
        )
    ):
        return True

    return False


def split_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 80,
) -> list[str]:
    """
    Split extracted document text into overlapping chunks.

    Uses general heading patterns and paragraph boundaries.
    Chunk size and overlap are measured in words.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    if not text or not text.strip():
        return []

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.splitlines()

    sections = []
    current_heading = ""
    current_lines = []

    def save_section():
        content = "\n".join(current_lines).strip()

        if current_heading or content:
            sections.append((current_heading, content))

        current_lines.clear()

    for raw_line in lines:
        line = re.sub(r"[ \t]+", " ", raw_line).strip()

        if not line:
            if current_lines and current_lines[-1] != "":
                current_lines.append("")
            continue

        if is_section_heading(line):
            save_section()
            current_heading = line
        else:
            current_lines.append(line)

    save_section()

    chunks = []

    for heading, content in sections:
        if not content.strip():
            continue

        heading_prefix = f"{heading}\n" if heading else ""
        heading_words = len(heading.split())
        body_size = max(1, chunk_size - heading_words)
        body_overlap = min(overlap, body_size - 1)
        step_size = body_size - body_overlap

        # Blank lines separate paragraphs where extraction preserves them.
        paragraphs = [
            re.sub(r"\s+", " ", paragraph).strip()
            for paragraph in re.split(r"\n\s*\n", content)
            if paragraph.strip()
        ]

        # Split long paragraphs into overlapping word segments.
        units = []

        for paragraph in paragraphs:
            words = paragraph.split()

            if len(words) <= body_size:
                units.append(words)
            else:
                for start in range(0, len(words), step_size):
                    units.append(words[start:start + body_size])

        current_words = []

        for unit in units:
            if not unit:
                continue

            if len(current_words) + len(unit) > body_size:
                if current_words:
                    chunks.append(
                        (
                            heading_prefix
                            + " ".join(current_words)
                        ).strip()
                    )

                current_words = (
                    current_words[-body_overlap:]
                    if body_overlap
                    else []
                )

            # Handle a unit that cannot fit after overlap.
            if len(current_words) + len(unit) > body_size:
                current_words = []

            current_words.extend(unit)

        if current_words:
            chunks.append(
                (heading_prefix + " ".join(current_words)).strip()
            )

    return [chunk for chunk in chunks if chunk.strip()]
