"""EPUB 3 export.

Written directly against the specification rather than through a library, because the
details that matter for an Amharic edition are exactly the ones generic exporters get
wrong: an embedded subsetted Ethiopic font (without it the book is empty boxes on most
readers), correct ``xml:lang="am"`` so reading systems pick Ethiopic fonts and
hyphenation rules, generous line spacing for fidel, and a ``page-list`` mapping the
digital text back to the printed pagination so the edition stays citable.
"""

from __future__ import annotations

import html
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .. import fonts
from ..fonts import FontInfo
from .document import Block, BookDocument

CONTAINER_XML = """<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""

STYLESHEET = """@charset "utf-8";

@font-face {{
  font-family: "EthiopicBook";
  src: url("../fonts/{font_file}") format("truetype");
  font-weight: normal;
  font-style: normal;
}}

html {{ font-size: 100%; }}

body {{
  font-family: "EthiopicBook", "Abyssinica SIL", "Noto Serif Ethiopic", "Nyala", serif;
  /* Fidel carry detail above and below the notional x-height, so they need more leading
     than a Latin face of the same size before the lines start to collide. */
  line-height: 1.75;
  margin: 0 5%;
  text-align: {alignment};
  hyphens: none;
  -webkit-hyphens: none;
  widows: 2;
  orphans: 2;
}}

h1, h2, h3 {{
  font-family: "EthiopicBook", serif;
  line-height: 1.45;
  text-align: center;
  page-break-after: avoid;
  margin: 1.4em 0 0.9em;
}}
h1 {{ font-size: 1.5em; }}
h2 {{ font-size: 1.28em; }}
h3 {{ font-size: 1.12em; }}

p {{ margin: 0; text-indent: 1.4em; }}
p.first, h1 + p, h2 + p, h3 + p {{ text-indent: 0; }}

p.verse {{
  text-indent: 0;
  margin: 0.25em 0 0.25em 1.6em;
  text-align: left;
}}

p.quote {{ margin: 0.8em 2em; text-indent: 0; font-size: 0.96em; }}
p.caption {{ text-align: center; font-size: 0.9em; margin: 0.6em 0; text-indent: 0; }}

aside.footnote {{ font-size: 0.88em; border-top: 1px solid #999; margin-top: 1.2em; padding-top: 0.4em; }}

span.pagebreak {{
  /* Invisible anchor; readers that display print pagination pick it up from page-list. */
  display: none;
}}
"""


@dataclass
class EpubOptions:
    identifier: str = ""
    publisher: str = ""
    description: str = ""
    rights: str = ""
    justify: bool = False
    embed_font: bool = True
    subset_font: bool = True
    include_page_list: bool = True
    chapters_per_file: int = 1
    font: FontInfo | None = None


def export_epub(
    document: BookDocument, output_path: str | Path, options: EpubOptions | None = None
) -> Path:
    """Write an EPUB 3 file and return its path."""
    options = options or EpubOptions()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    identifier = options.identifier or f"urn:uuid:{uuid.uuid4()}"
    modified = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    font = options.font or fonts.default_font()
    font_bytes: bytes | None = None
    font_file = ""
    if options.embed_font and font is not None:
        font_file = "ethiopic.ttf"
        font_bytes = _font_payload(font, document, options.subset_font)

    chapter_files = _render_chapters(document, options)

    with zipfile.ZipFile(output, "w") as archive:
        # The mimetype entry must be first and stored uncompressed.
        archive.writestr(
            zipfile.ZipInfo("mimetype"), "application/epub+zip", compress_type=zipfile.ZIP_STORED
        )
        write = lambda name, data: archive.writestr(  # noqa: E731
            name, data, compress_type=zipfile.ZIP_DEFLATED
        )

        write("META-INF/container.xml", CONTAINER_XML)
        write(
            "OEBPS/styles/main.css",
            STYLESHEET.format(
                font_file=font_file or "ethiopic.ttf",
                alignment="justify" if options.justify else "left",
            ),
        )
        if font_bytes:
            write(f"OEBPS/fonts/{font_file}", font_bytes)

        for filename, content in chapter_files:
            write(f"OEBPS/text/{filename}", content)

        write("OEBPS/nav.xhtml", _render_nav(document, chapter_files, options))
        write(
            "OEBPS/content.opf",
            _render_opf(document, chapter_files, identifier, modified, font_file, options),
        )

    return output


# --------------------------------------------------------------------------------------
# Content
# --------------------------------------------------------------------------------------


def _font_payload(font: FontInfo, document: BookDocument, subset: bool) -> bytes:
    if not subset:
        return font.path.read_bytes()
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "subset.ttf"
        try:
            fonts.subset_font(font.path, document.character_set, target)
            return target.read_bytes()
        except Exception:  # noqa: BLE001 - subsetting is an optimization, not a requirement
            return font.path.read_bytes()


def _render_chapters(document: BookDocument, options: EpubOptions) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for index, chapter in enumerate(document.chapters, start=1):
        filename = f"chapter_{index:03d}.xhtml"
        body: list[str] = []
        first_paragraph = True
        seen_pages: set[int] = set()

        for block in chapter.blocks:
            if options.include_page_list and block.page_idx not in seen_pages:
                seen_pages.add(block.page_idx)
                label = _page_label(document, block.page_idx)
                body.append(
                    f'<span class="pagebreak" epub:type="pagebreak" role="doc-pagebreak" '
                    f'id="page-{block.page_idx}" aria-label="{html.escape(label)}"></span>'
                )
            body.append(_render_block(block, first_paragraph))
            if block.kind == "paragraph":
                first_paragraph = False
            elif block.is_heading:
                first_paragraph = True

        out.append((filename, _xhtml_page(chapter.title, "\n".join(body), document.language)))
    return out


def _render_block(block: Block, first: bool) -> str:
    text = html.escape(block.text).replace("\n", "<br/>")
    if block.kind == "chapter":
        return f"<h1>{text}</h1>"
    if block.kind == "heading":
        return f"<h2>{text}</h2>"
    if block.kind == "subheading":
        return f"<h3>{text}</h3>"
    if block.kind == "verse":
        return f'<p class="verse">{text}</p>'
    if block.kind == "quote":
        return f'<p class="quote">{text}</p>'
    if block.kind == "caption":
        return f'<p class="caption">{text}</p>'
    if block.kind == "footnote":
        return f'<aside class="footnote" epub:type="footnote">{text}</aside>'
    return f'<p class="first">{text}</p>' if first else f"<p>{text}</p>"


def _xhtml_page(title: str, body: str, language: str) -> str:
    return f"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops"
      lang="{language}" xml:lang="{language}">
<head>
  <meta charset="utf-8"/>
  <title>{html.escape(title)}</title>
  <link rel="stylesheet" type="text/css" href="../styles/main.css"/>
</head>
<body>
{body}
</body>
</html>
"""


def _page_label(document: BookDocument, page_idx: int) -> str:
    for idx, label in document.page_breaks:
        if idx == page_idx:
            return label
    return str(page_idx + 1)


def _render_nav(
    document: BookDocument, chapter_files: list[tuple[str, str]], options: EpubOptions
) -> str:
    toc_items = "\n".join(
        f'      <li><a href="text/{filename}">{html.escape(chapter.title or f"Section {i}")}</a></li>'
        for i, (chapter, (filename, _)) in enumerate(
            zip(document.chapters, chapter_files, strict=True), start=1
        )
    )

    page_list = ""
    if options.include_page_list and document.page_breaks:
        file_for_page = _page_to_file(document, chapter_files)
        entries = "\n".join(
            f'      <li><a href="text/{file_for_page.get(idx, chapter_files[0][0])}#page-{idx}">'
            f"{html.escape(label)}</a></li>"
            for idx, label in document.page_breaks
        )
        page_list = f"""
  <nav epub:type="page-list" role="doc-pagelist" hidden="hidden">
    <h2>Printed pages</h2>
    <ol>
{entries}
    </ol>
  </nav>
"""

    return f"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops"
      lang="{document.language}" xml:lang="{document.language}">
<head>
  <meta charset="utf-8"/>
  <title>{html.escape(document.title)}</title>
  <link rel="stylesheet" type="text/css" href="styles/main.css"/>
</head>
<body>
  <nav epub:type="toc" role="doc-toc" id="toc">
    <h1>{html.escape(document.title)}</h1>
    <ol>
{toc_items}
    </ol>
  </nav>
{page_list}
</body>
</html>
"""


def _page_to_file(
    document: BookDocument, chapter_files: list[tuple[str, str]]
) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for chapter, (filename, _) in zip(document.chapters, chapter_files, strict=True):
        for block in chapter.blocks:
            mapping.setdefault(block.page_idx, filename)
    return mapping


def _render_opf(
    document: BookDocument,
    chapter_files: list[tuple[str, str]],
    identifier: str,
    modified: str,
    font_file: str,
    options: EpubOptions,
) -> str:
    manifest = [
        '    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>',
        '    <item id="css" href="styles/main.css" media-type="text/css"/>',
    ]
    if font_file:
        manifest.append(
            f'    <item id="font" href="fonts/{font_file}" media-type="font/ttf"/>'
        )
    spine: list[str] = []
    for index, (filename, _) in enumerate(chapter_files, start=1):
        item_id = f"ch{index:03d}"
        manifest.append(
            f'    <item id="{item_id}" href="text/{filename}" media-type="application/xhtml+xml"/>'
        )
        spine.append(f'    <itemref idref="{item_id}"/>')

    optional = []
    if options.publisher:
        optional.append(f"    <dc:publisher>{html.escape(options.publisher)}</dc:publisher>")
    if options.description:
        optional.append(f"    <dc:description>{html.escape(options.description)}</dc:description>")
    if options.rights:
        optional.append(f"    <dc:rights>{html.escape(options.rights)}</dc:rights>")
    if document.author:
        optional.append(
            f'    <dc:creator id="creator">{html.escape(document.author)}</dc:creator>'
        )

    page_source = ""
    if options.include_page_list and document.page_breaks:
        page_source = (
            '    <meta property="dcterms:source" id="src">print edition</meta>\n'
            '    <meta refines="#src" property="source-of">pagination</meta>'
        )

    return f"""<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="pub-id"
         xml:lang="{document.language}">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="pub-id">{html.escape(identifier)}</dc:identifier>
    <dc:title>{html.escape(document.title)}</dc:title>
    <dc:language>{document.language}</dc:language>
{chr(10).join(optional)}
    <meta property="dcterms:modified">{modified}</meta>
{page_source}
  </metadata>
  <manifest>
{chr(10).join(manifest)}
  </manifest>
  <spine>
{chr(10).join(spine)}
  </spine>
</package>
"""
