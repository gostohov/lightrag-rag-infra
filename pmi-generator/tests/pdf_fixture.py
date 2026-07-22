from __future__ import annotations

from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject


def write_text_pdf(
    path: Path,
    page_texts: tuple[str, ...],
    *,
    outline: tuple[tuple[str, int, int | None], ...] = (),
    page_labels: tuple[str, ...] = (),
) -> None:
    writer = PdfWriter()
    for text in page_texts:
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        font_reference = writer._add_object(font)
        page[NameObject("/Resources")] = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject(
                    {NameObject("/F1"): font_reference}
                )
            }
        )
        commands = ["BT", "/F1 12 Tf", "72 720 Td"]
        for index, line in enumerate(text.splitlines()):
            if index:
                commands.extend(("0 -18 Td",))
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            commands.append(f"({escaped}) Tj")
        commands.append("ET")
        content = DecodedStreamObject()
        content.set_data("\n".join(commands).encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(content)
    for page_number, label in enumerate(page_labels):
        writer.set_page_label(page_number, page_number, prefix=label)
    references: list[object] = []
    for title, page_number, parent_index in outline:
        parent = references[parent_index] if parent_index is not None else None
        references.append(
            writer.add_outline_item(
                title,
                page_number,
                parent=parent,  # type: ignore[arg-type]
            )
        )
    writer.write(path)
