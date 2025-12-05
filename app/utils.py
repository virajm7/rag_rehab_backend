# app/utils.py
from pathlib import Path
from typing import Optional
from pypdf import PdfReader

def load_text_or_pdf(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} not found")
    if p.suffix.lower() in [".txt", ".md"]:
        return p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".pdf":
        text = []
        reader = PdfReader(str(p))
        for page in reader.pages:
            text.append(page.extract_text() or "")
        return "\n".join(text)
    # Add other loaders if needed
    return p.read_text(encoding="utf-8")
