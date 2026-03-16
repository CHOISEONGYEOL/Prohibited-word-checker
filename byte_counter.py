import unicodedata
from typing import List
from schemas import SuspiciousChar


# =========================
# Byte Counter Logic (v2.0)
# =========================
SUSPICIOUS_CODEPOINTS = {
    0x0009: "TAB (\\t)",
    0x000A: "LF (\\n)",
    0x000D: "CR (\\r)",
    0x00A0: "NBSP (no-break space)",
    0x2000: "EN QUAD",
    0x2001: "EM QUAD",
    0x2002: "EN SPACE",
    0x2003: "EM SPACE",
    0x2004: "THREE-PER-EM SPACE",
    0x2005: "FOUR-PER-EM SPACE",
    0x2006: "SIX-PER-EM SPACE",
    0x2007: "FIGURE SPACE",
    0x2008: "PUNCTUATION SPACE",
    0x2009: "THIN SPACE",
    0x200A: "HAIR SPACE",
    0x200B: "ZWSP (zero width space)",
    0x200C: "ZWNJ (zero width non-joiner)",
    0x200D: "ZWJ (zero width joiner)",
    0x202F: "NNBSP (narrow no-break space)",
    0x205F: "MMSP (medium mathematical space)",
    0x3000: "IDEOGRAPHIC SPACE",
    0xFEFF: "BOM/ZWNBSP",
}


def utf8_byte_len(text: str) -> int:
    return len(text.encode("utf-8"))


def analyze_bytes(text: str) -> dict:
    chars_including = len(text)
    chars_excluding_spaces = sum(1 for ch in text if not ch.isspace())
    lf = text.count("\n")
    cr = text.count("\r")
    tab = text.count("\t")

    suspicious: List[SuspiciousChar] = []
    for i, ch in enumerate(text):
        cp = ord(ch)
        if cp in SUSPICIOUS_CODEPOINTS:
            name = SUSPICIOUS_CODEPOINTS[cp]
            category = unicodedata.category(ch)
            suspicious.append(SuspiciousChar(
                index=i,
                char_repr=repr(ch),
                codepoint=f"U+{cp:04X}",
                name=name,
                unicode_category=category,
            ))

    return {
        "utf8_bytes": utf8_byte_len(text),
        "char_count_including_spaces": chars_including,
        "char_count_excluding_spaces": chars_excluding_spaces,
        "newline_lf": lf,
        "newline_cr": cr,
        "tab": tab,
        "suspicious": suspicious,
    }


def normalize_for_neis(
    text: str,
    newline_mode: str = "LF",
    replace_nbsp: bool = True,
    remove_zero_width: bool = True,
) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if newline_mode.upper() == "CRLF":
        text = text.replace("\n", "\r\n")

    if replace_nbsp:
        text = text.replace("\u00A0", " ")
        text = text.replace("\u202F", " ")
        text = text.replace("\u3000", " ")

    if remove_zero_width:
        for zw in ("\u200B", "\u200C", "\u200D", "\uFEFF"):
            text = text.replace(zw, "")

    return text
