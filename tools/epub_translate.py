#!/usr/bin/env python3
"""Translate epub content via the OpenAI Chat Completions API."""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import time
import urllib.request
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from xml.etree import ElementTree


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._current: list[str] = []
        self._block_tags = {
            "p",
            "div",
            "section",
            "article",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "li",
        }
        self._skip_tags = {"script", "style", "svg"}
        self._skipping = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._skip_tags:
            self._skipping = True
        if tag in self._block_tags and self._current:
            self._flush_current()

    def handle_endtag(self, tag: str) -> None:
        if tag in self._skip_tags:
            self._skipping = False
        if tag in self._block_tags:
            self._flush_current()

    def handle_data(self, data: str) -> None:
        if self._skipping:
            return
        text = data.strip()
        if text:
            self._current.append(text)

    def _flush_current(self) -> None:
        if self._current:
            joined = " ".join(self._current).strip()
            if joined:
                self._chunks.append(joined)
        self._current = []

    def get_chunks(self) -> list[str]:
        self._flush_current()
        return self._chunks


class OpenAIClient:
    def __init__(self, api_key: str, model: str, delay: float) -> None:
        self.api_key = api_key
        self.model = model
        self.delay = delay

    def translate(self, text: str, source: str, target: str) -> str:
        system_prompt = (
            "You are a translation engine. "
            "Translate the user text from {source} to {target}. "
            "Preserve paragraph breaks. Return only the translation."
        ).format(source=source, target=target)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            "temperature": 0.2,
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                response_body = response.read().decode("utf-8")
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(
                f"HTTP {exc.code} from OpenAI API: {error_body or exc.reason}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(f"Network error contacting OpenAI API: {exc}") from exc
        parsed = json.loads(response_body)
        try:
            translated = parsed["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected response: {parsed}") from exc
        if self.delay:
            time.sleep(self.delay)
        return translated.strip()


def _read_xml(archive: zipfile.ZipFile, name: str) -> ElementTree.Element:
    data = archive.read(name)
    return ElementTree.fromstring(data)


def _resolve_href(base: Path, href: str) -> str:
    cleaned = href.split("#", 1)[0]
    resolved = (base / cleaned).as_posix()
    return resolved.lstrip("./")


def _extract_spine_order(archive: zipfile.ZipFile) -> list[str]:
    try:
        container = _read_xml(archive, "META-INF/container.xml")
    except KeyError:
        return []
    rootfile = container.find(".//{*}rootfile")
    if rootfile is None:
        return []
    opf_path = rootfile.attrib.get("full-path")
    if not opf_path:
        return []
    try:
        opf = _read_xml(archive, opf_path)
    except KeyError:
        return []
    opf_dir = Path(opf_path).parent
    manifest: dict[str, str] = {}
    for item in opf.findall(".//{*}manifest/{*}item"):
        item_id = item.attrib.get("id")
        href = item.attrib.get("href")
        if item_id and href:
            manifest[item_id] = _resolve_href(opf_dir, href)
    spine_hrefs: list[str] = []
    for itemref in opf.findall(".//{*}spine/{*}itemref"):
        idref = itemref.attrib.get("idref")
        if not idref:
            continue
        href = manifest.get(idref)
        if href:
            spine_hrefs.append(href)
    return spine_hrefs


def extract_epub_text(epub_path: Path, max_sections: int) -> list[tuple[str, list[str]]]:
    results: list[tuple[str, list[str]]] = []
    with zipfile.ZipFile(epub_path) as archive:
        spine = _extract_spine_order(archive)
        if not spine:
            spine = [
                name
                for name in archive.namelist()
                if name.lower().endswith((".xhtml", ".html", ".htm"))
            ]
        for name in spine[: max_sections or None]:
            try:
                data = archive.read(name).decode("utf-8", errors="ignore")
            except KeyError:
                continue
            parser = TextExtractor()
            parser.feed(data)
            chunks = parser.get_chunks()
            if chunks:
                results.append((name, chunks))
    return results


def chunk_text(chunks: list[str], max_chars: int) -> list[str]:
    grouped: list[str] = []
    current: list[str] = []
    current_len = 0
    for paragraph in chunks:
        if current_len + len(paragraph) + 1 > max_chars and current:
            grouped.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(paragraph)
        current_len += len(paragraph) + 1
    if current:
        grouped.append("\n\n".join(current))
    return grouped


def build_output(translated: list[tuple[str, list[str]]]) -> str:
    lines: list[str] = []
    for name, segments in translated:
        lines.append(f"# {name}")
        lines.append("")
        for segment in segments:
            wrapped = textwrap.fill(segment, width=80)
            lines.append(wrapped)
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate epub content using the OpenAI Chat Completions API.",
    )
    parser.add_argument("epub", type=Path, help="Path to the epub file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("translated.md"),
        help="Output markdown file.",
    )
    parser.add_argument(
        "--source",
        default="en",
        help="Source language code (default: en).",
    )
    parser.add_argument(
        "--target",
        default="zh",
        help="Target language code (default: zh).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model name (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1500,
        help="Maximum characters per translation request.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="Delay (seconds) between API requests.",
    )
    parser.add_argument(
        "--max-sections",
        type=int,
        default=0,
        help="Limit number of spine sections to translate (0 = all).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.epub.exists():
        print(f"Epub not found: {args.epub}", file=sys.stderr)
        return 1
    print(f"Reading: {args.epub}")
    sections = extract_epub_text(args.epub, args.max_sections)
    if not sections:
        print("No readable text found in epub.", file=sys.stderr)
        return 1
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Missing OpenAI API key. Use --api-key or set OPENAI_API_KEY.", file=sys.stderr)
        return 1
    client = OpenAIClient(api_key, args.model, args.delay)
    translated_sections: list[tuple[str, list[str]]] = []
    for name, chunks in sections:
        grouped = chunk_text(chunks, args.max_chars)
        translated_chunks: list[str] = []
        for block in grouped:
            try:
                translated_chunks.append(
                    client.translate(block, args.source, args.target)
                )
            except RuntimeError as exc:
                print(f"Translation failed: {exc}", file=sys.stderr)
                return 1
        translated_sections.append((name, translated_chunks))
        print(f"Translated {name} ({len(grouped)} blocks)")
    output_text = build_output(translated_sections)
    args.output.write_text(output_text, encoding="utf-8")
    print(f"Saved translation to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
