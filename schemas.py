from typing import List, Optional
from pydantic import BaseModel


# =========================
# Pydantic Schemas
# =========================
class Source(BaseModel):
    doc: str
    page: Optional[int] = None
    quote: Optional[str] = None


class Hit(BaseModel):
    span: str
    label: str
    replacement: Optional[str] = None
    confidence: float
    source: Source
    start: int
    end: int
    delete_with_particle: bool = False


class AnalyzeRequest(BaseModel):
    text: str
    policy_version: str = "2024-03"


class AnalyzeResponse(BaseModel):
    hits: List[Hit]
    latency_ms: int


# =========================
# Byte Counter Schemas (v2.0)
# =========================
class SuspiciousChar(BaseModel):
    index: int
    char_repr: str
    codepoint: str
    name: str
    unicode_category: str


class ByteCountRequest(BaseModel):
    text: str
    normalize: bool = False
    newline_mode: str = "LF"


class ByteCountResponse(BaseModel):
    utf8_bytes: int
    char_count_including_spaces: int
    char_count_excluding_spaces: int
    newline_lf: int
    newline_cr: int
    tab: int
    suspicious: List[SuspiciousChar]
    normalized_text: Optional[str] = None
    normalized_utf8_bytes: Optional[int] = None
