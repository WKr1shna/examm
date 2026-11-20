import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
import pypdf

@dataclass
class DocumentChunk:
    chunk_id: str
    page_number: int
    text: str
    chapter: str = ""
    module: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

def parse_pdf_manual(pdf_path: str) -> List[DocumentChunk]:
    """
    Parses the 50-page Grand Bistro Culinary Recipe Manual.
    Extracts text per page, preserving exact page numbers, numbers, units, and structural metadata.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    reader = pypdf.PdfReader(str(path))
    total_pages = len(reader.pages)
    chunks: List[DocumentChunk] = []

    for idx, page in enumerate(reader.pages):
        page_num = idx + 1
        raw_text = page.extract_text() or ""
        
        # Clean leading/trailing whitespaces while preserving structural lines
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        cleaned_text = "\n".join(lines)

        # Extract Chapter & Module from headers
        chapter = ""
        module = ""
        chap_match = re.search(r"CHAPTER\s+(\d+)", cleaned_text, re.IGNORECASE)
        if chap_match:
            chapter = f"Chapter {chap_match.group(1)}"

        mod_match = re.search(r"MODULE\s+([\d\.]+:\s*[^\n=]+)", cleaned_text, re.IGNORECASE)
        if mod_match:
            module = f"Module {mod_match.group(1).strip()}"

        has_recipe = "RECIPE CARD" in cleaned_text
        has_allergen = "ALLERGEN MATRIX" in cleaned_text
        has_haccp = "HACCP" in cleaned_text or "PASTEURIZATION" in cleaned_text

        chunk = DocumentChunk(
            chunk_id=f"page_{page_num}",
            page_number=page_num,
            text=cleaned_text,
            chapter=chapter,
            module=module,
            metadata={
                "source": path.name,
                "page_number": page_num,
                "total_pages": total_pages,
                "chapter": chapter,
                "module": module,
                "has_recipe": has_recipe,
                "has_allergen": has_allergen,
                "has_haccp": has_haccp,
                "char_count": len(cleaned_text),
                "word_count": len(cleaned_text.split())
            }
        )
        chunks.append(chunk)

    return chunks

if __name__ == "__main__":
    from app.config import settings
    chunks = parse_pdf_manual(settings.PDF_PATH)
    print(f"Extracted {len(chunks)} chunks from {settings.PDF_PATH}")
    for c in chunks:
        if c.metadata.get("has_recipe") or c.metadata.get("has_allergen") or c.metadata.get("has_haccp"):
            print(f"Special page {c.page_number}: {c.module} (Recipe: {c.metadata['has_recipe']}, Allergen: {c.metadata['has_allergen']}, HACCP: {c.metadata['has_haccp']})")
