import os
import time
import re
import unicodedata
from typing import List, Optional, Dict
import numpy as np
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# --- AI Model Loading ---
EMB_MODEL = os.getenv("EMB_MODEL", "intfloat/multilingual-e5-small")
_embedder = None
try:
    from sentence_transformers import SentenceTransformer
    print(f"Loading embedding model: {EMB_MODEL}...")
    _embedder = SentenceTransformer(EMB_MODEL)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Warning: Failed to load sentence-transformer model: {e}")
    print("Running in regex-only mode.")
    _embedder = None
# --- End of AI Model Loading ---

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
    delete_with_particle: bool = False  # v2.0.2: 삭제 시 조사도 함께 삭제


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
    newline_mode: str = "LF"  # "LF" or "CRLF"


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
    """생활기록부 카운팅에 맞춘 핵심: UTF-8 인코딩 바이트 길이."""
    return len(text.encode("utf-8"))


def analyze_bytes(text: str) -> dict:
    """텍스트의 바이트 및 문자 정보 분석."""
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
    """
    NEIS 입력에 맞춘 정규화.
    - newline_mode: 줄바꿈을 LF 또는 CRLF로 통일
    - replace_nbsp: NBSP류를 일반 스페이스로 치환
    - remove_zero_width: ZWSP/ZWJ/ZWNJ/FEFF 제거
    """
    # 1) 줄바꿈 통일
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if newline_mode.upper() == "CRLF":
        text = text.replace("\n", "\r\n")

    # 2) 특수 공백 치환
    if replace_nbsp:
        text = text.replace("\u00A0", " ")   # NBSP
        text = text.replace("\u202F", " ")   # NNBSP
        text = text.replace("\u3000", " ")   # IDEOGRAPHIC SPACE

    # 3) 제로폭/보이지 않는 문자 제거
    if remove_zero_width:
        for zw in ("\u200B", "\u200C", "\u200D", "\uFEFF"):
            text = text.replace(zw, "")

    return text


# =========================
# FastAPI App Setup
# =========================
app = FastAPI(title="LifeRec Checker", version="2.0.7")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)

# =========================
# Policy Rules Database
# =========================
RULES = [
    # --- 상호명, 플랫폼명 ---
    {"pattern": r"(?:NAVER|Naver|네이버|Daum|다음|\bGoogle\b(?!\s?(Docs|Classroom|TV)))", "label": "상호명", "replacement": "포털사이트", "confidence": 0.95, "source": {"doc": "대체표현", "page": 1, "quote": "Google(구글), NAVER(네이버), Daum(다음) 등 → 포털사이트"}, "aliases": ["googel", "gooogle", "구글검색", "네이버검색", "다음검색"]},
    {"pattern": r"(?:Google\s?Classroom|구글\s?클래스룸|EBS\s?온라인클래스|classting|클래스팅)", "label": "상호명", "replacement": "학습 플랫폼", "confidence": 0.95, "source": {"doc": "대체표현", "page": 1, "quote": "Google Classroom(구글 클래스룸), EBS 온라인클래스 등 → 학습 플랫폼"}, "aliases": ["gclassroom", "구클", "클래스팅앱", "이비에스 온라인클래스"]},
    {"pattern": r"(?:TikTok|틱톡)", "label": "상호명", "replacement": "엔터테인먼트 플랫폼", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "TikTok(틱톡) 등 → 엔터테인먼트 플랫폼"}, "aliases": ["틱톡영상", "tiktoc", "틱톡스"]},
    {"pattern": r"(?:YouTube|유튜브|TVING|티빙|watcha|왓챠|netflix|넷플릭스|wavve|웨이브|disney\s?plus|디즈니\+?|디즈니플러스|OTT)", "label": "상호명", "replacement": "동영상 플랫폼", "confidence": 0.95, "source": {"doc": "대체표현", "page": 1, "quote": "YouTube(유튜브), TVING(티빙) ... OTT 등 → 동영상 플랫폼"}, "aliases": ["yutube", "you tube", "유튭", "유툽", "넷플", "왓챠플레이"]},
    {"pattern": r"(?:YouTuber|유튜버)", "label": "직업명", "replacement": "동영상 크리에이터", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "YouTuber(유튜버) 등 → 동영상 크리에이터, 동영상 제공자"}, "aliases": ["유튜브러", "youtuber"]},
    {"pattern": r"(?:KakaoTalk|카카오톡|카톡|\bLINE\b|(?<![가-힣])라인(?![가-힣])|Instagram|인스타그램|Twitter|트위터|Meta|메타|Facebook|페이스북)", "label": "상호명", "replacement": "소셜 네트워크 서비스", "confidence": 0.95, "source": {"doc": "대체표현", "page": 1, "quote": "KakaoTalk, Instagram, Facebook 등 → 메신저, 소셜네트워크서비스"}, "aliases": ["kakaotalk", "kkt", "카톡방", "인스타", "insta", "페북", "x(트위터)"]},
    {"pattern": r"(?:Chat\s?GPT|챗\s?GPT|챗지피티|wrtn|뤼튼|bing\s?Chat|빙챗|Bard|바드|하이퍼클로바X|HyperClova\s?X)", "label": "상호명", "replacement": "생성형 인공지능", "confidence": 0.95, "source": {"doc": "대체표현", "page": 1, "quote": "Chat GPT(챗지피티), wrtn(뤼튼) ... 등 → 대화형 인공지능, 생성형 인공지능"}, "aliases": ["chatgpt", "챗쥐피티", "gpt챗", "빙챗봇", "하클x", "뤼튼ai"]},
    {"pattern": r"(?:Canva|캔바|miricanvas|미리캔버스|mangoboard|망고보드)", "label": "상호명", "replacement": "디자인 제작 플랫폼", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "miricanvas(미리캔버스), mangoboard(망고보드), Canva(캔바) 등"}, "aliases": ["캔바앱", "미캔"]},
    {"pattern": r"(?:KineMaster|키네마스터|Premiere\s?Pro|프리미어\s?프로)", "label": "프로그램명", "replacement": "영상 편집 프로그램", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "영상 제작 프로그램, 영상 편집 프로그램"}, "aliases": ["키네", "프리미어"]},
    # --- 개발 언어 / 개발 도구 ---
    {"pattern": r"(?:Python|파이썬|Java\b|자바|C\+\+|C언어|자바\s?스크립트|Java\s*Script|JavaScript|Javascript|JS\b)", "label": "프로그램명", "replacement": "프로그래밍 언어", "confidence": 0.92, "source": {"doc": "대체표현", "page": 2, "quote": "프로그램명 (파이썬, C언어 등) 기재 불가"}, "aliases": ["파이선", "자바스크립", "씨언어", "javascript", "java script", "js"]},
    {"pattern": r"(?:Jupyter|주피터|Colab|코랩|PyCharm|파이참|VS\s?Code|Visual\s?Studio\s?Code|비주얼\s?스튜디오\s?코드|Anaconda|Spyder)", "label": "프로그램명", "replacement": "개발 도구", "confidence": 0.92, "source": {"doc": "대체표현", "page": 2, "quote": "특정 소프트웨어/개발환경 기재 지양, 일반화 표현 사용"}, "aliases": ["주피터", "코랩", "파이참", "vscode", "vs코드"]},
    # --- 오피스 / 문서 ---
    {"pattern": r"(?:MS\s?워드|MS\s?Word|Microsoft\s?Word|워드)", "label": "프로그램명", "replacement": "문서작성 프로그램", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "hwp, MS워드 → 문서작성 프로그램"}, "aliases": ["msword", "워드파일"]},
    {"pattern": r"(?:Google\s?Docs|구글\s?문서|구글\s?독스)", "label": "프로그램명", "replacement": "온라인 문서 편집기", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "Google Docs(구글문서) 등 → 온라인 문서 편집기"}, "aliases": ["gdocs"]},
    # --- 강연 / 이동수단 / 화상회의 ---
    {"pattern": r"(?:TED|테드)", "label": "강연명", "replacement": "온라인 강연회", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "TED(테드) 등 → 온라인 강연회"}, "aliases": ["ted 강연", "테드톡"]},
    {"pattern": r"(?:KTX|케이티엑스|SRT|에스알티)", "label": "상호명", "replacement": "고속 열차", "confidence": 0.95, "source": {"doc": "대체표현", "page": 1, "quote": "KTX, SRT → 고속 열차"}, "aliases": ["케텍", "에스알티"]},
    {"pattern": r"(?:\bZoom\b|(?<![가-힣])줌(?![가-힣])|웨일온|Whale\s?ON)", "label": "상호명", "replacement": "화상 회의", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "Zoom(줌) 등 → 화상 회의"}, "aliases": ["줌미팅", "웨일온회의"]},
    # --- 기관명 / 논문 / 외국어 / 특수문자 ---
    {"pattern": r"(?:UN|EU|ASEAN|APEC|G7|G20|WHO|WTO|OECD|IMF|IAEA|NATO|UNESCO|UNICEF|UNEP|UNDP|UNHCR|유엔|유럽연합)", "label": "기관명", "replacement": "국제기구", "confidence": 0.98, "source": {"doc": "단체명 기재", "page": 1, "quote": "교육관련기관 제외 특정 기관명 기재 불가"}, "aliases": ["유엔기구", "오이시디", "나토", "유네스코한국위원회"]},
    {"pattern": r"소논문|연구보고서", "label": "논문 실적", "replacement": "탐구 활동", "confidence": 0.99, "source": {"doc": "논문 기재", "page": 1, "quote": "자율탐구활동 산출물 실적 기재 불가"}, "aliases": ["소논문 작성", "연구보고서를 제출"]},
    {"pattern": r"[一-龥]", "label": "외국어", "replacement": None, "confidence": 0.99, "source": {"doc": "외국어 기재", "page": 1, "quote": "한글 사용 원칙. 영문 제외 외국어 입력 불가."}, "aliases": []},
    {"pattern": r"·", "label": "특수문자", "replacement": ", ", "confidence": 0.99, "source": {"doc": "특수문자", "page": 1, "quote": "서술형 특수문자 입력 지양"}, "aliases": []},
    {"pattern": r"[※▷▶]", "label": "특수문자", "replacement": " ", "confidence": 0.99, "source": {"doc": "특수문자", "page": 1, "quote": "서술형 특수문자 입력 지양"}, "aliases": []},
    # --- 국내 연구기관 ---
    {"pattern": r"\bKIOST\b", "label": "기관명", "replacement": "해양과학기술원", "confidence": 0.95, "source": {"doc": "영문 약어", "page": 1, "quote": "영문 약어는 한글 풀이로 대체"}, "aliases": []},
    {"pattern": r"\bKIGAM\b", "label": "기관명", "replacement": "지질자원연구원", "confidence": 0.95, "source": {"doc": "영문 약어", "page": 1, "quote": "영문 약어는 한글 풀이로 대체"}, "aliases": []},
    {"pattern": r"\bNOAA\b", "label": "기관명", "replacement": "미국해양대기청", "confidence": 0.95, "source": {"doc": "영문 약어", "page": 1, "quote": "영문 약어는 한글 풀이로 대체"}, "aliases": []},
    {"pattern": r"\bIAU\b", "label": "기관명", "replacement": "국제천문연맹", "confidence": 0.95, "source": {"doc": "영문 약어", "page": 1, "quote": "영문 약어는 한글 풀이로 대체"}, "aliases": []},
    # --- 과학 전문 용어 ---
    {"pattern": r"\bGIC\b", "label": "전문 약어", "replacement": "지자기유도전류", "confidence": 0.95, "source": {"doc": "전문 약어", "page": 1, "quote": "전문 약어는 한글 풀이로 대체"}, "aliases": []},
    {"pattern": r"\bEEZ\b", "label": "전문 약어", "replacement": "배타적 경제수역", "confidence": 0.95, "source": {"doc": "전문 약어", "page": 1, "quote": "전문 약어는 한글 풀이로 대체"}, "aliases": []},
    {"pattern": r"\bK-VENT\b", "label": "전문 약어", "replacement": "호흡기 감염병 위험도 평가툴", "confidence": 0.95, "source": {"doc": "전문 약어", "page": 1, "quote": "전문 약어는 한글 풀이로 대체"}, "aliases": []},
    # --- 우주/항공 기관명 ---
    {"pattern": r"\bNASA\b", "label": "기관명", "replacement": "미국항공우주국", "confidence": 0.95, "source": {"doc": "영문 약어", "page": 1, "quote": "영문 약어는 한글 풀이로 대체"}, "aliases": ["나사"]},
    {"pattern": r"\bESA\b", "label": "기관명", "replacement": "유럽우주국", "confidence": 0.95, "source": {"doc": "영문 약어", "page": 1, "quote": "영문 약어는 한글 풀이로 대체"}, "aliases": []},
    {"pattern": r"\bJAXA\b", "label": "기관명", "replacement": "일본우주항공연구개발기구", "confidence": 0.95, "source": {"doc": "영문 약어", "page": 1, "quote": "영문 약어는 한글 풀이로 대체"}, "aliases": []},
    {"pattern": r"\bSpaceX\b", "label": "기관명", "replacement": "민간 우주개발 기업", "confidence": 0.95, "source": {"doc": "영문 약어", "page": 1, "quote": "특정 기업명은 일반화 표현 사용"}, "aliases": ["스페이스엑스"]},
    # --- 해양/과학 전문 약어 ---
    {"pattern": r"\bAUV\b", "label": "전문 약어", "replacement": "자율무인잠수정", "confidence": 0.95, "source": {"doc": "전문 약어", "page": 1, "quote": "전문 약어는 한글 풀이로 대체"}, "aliases": []},
    {"pattern": r"\bROV\b", "label": "전문 약어", "replacement": "원격조종무인잠수정", "confidence": 0.95, "source": {"doc": "전문 약어", "page": 1, "quote": "전문 약어는 한글 풀이로 대체"}, "aliases": []},
    {"pattern": r"\bGPS\b", "label": "전문 약어", "replacement": "위성항법장치", "confidence": 0.95, "source": {"doc": "전문 약어", "page": 1, "quote": "전문 약어는 한글 풀이로 대체"}, "aliases": []},
    {"pattern": r"\bLiDAR\b", "label": "전문 약어", "replacement": "레이저 거리측정장치", "confidence": 0.95, "source": {"doc": "전문 약어", "page": 1, "quote": "전문 약어는 한글 풀이로 대체"}, "aliases": ["라이다"]},
    {"pattern": r"\bSODAR\b", "label": "전문 약어", "replacement": "음파 탐지장치", "confidence": 0.95, "source": {"doc": "전문 약어", "page": 1, "quote": "전문 약어는 한글 풀이로 대체"}, "aliases": []},
    {"pattern": r"\bSONAR\b", "label": "전문 약어", "replacement": "수중 음파 탐지기", "confidence": 0.95, "source": {"doc": "전문 약어", "page": 1, "quote": "전문 약어는 한글 풀이로 대체"}, "aliases": ["소나"]},
    {"pattern": r"\bRADAR\b", "label": "전문 약어", "replacement": "전파 탐지기", "confidence": 0.95, "source": {"doc": "전문 약어", "page": 1, "quote": "전문 약어는 한글 풀이로 대체"}, "aliases": ["레이더"]},
    # --- 환경/에너지 전문 약어 ---
    {"pattern": r"\bCO2\b", "label": "전문 약어", "replacement": "이산화탄소", "confidence": 0.95, "source": {"doc": "전문 약어", "page": 1, "quote": "화학식은 한글명으로 대체"}, "aliases": []},
    {"pattern": r"\bPM2\.5\b", "label": "전문 약어", "replacement": "초미세먼지", "confidence": 0.95, "source": {"doc": "전문 약어", "page": 1, "quote": "전문 약어는 한글 풀이로 대체"}, "aliases": []},
    {"pattern": r"\bPM10\b", "label": "전문 약어", "replacement": "미세먼지", "confidence": 0.95, "source": {"doc": "전문 약어", "page": 1, "quote": "전문 약어는 한글 풀이로 대체"}, "aliases": []},
    {"pattern": r"\bLED\b", "label": "전문 약어", "replacement": "발광다이오드", "confidence": 0.95, "source": {"doc": "전문 약어", "page": 1, "quote": "전문 약어는 한글 풀이로 대체"}, "aliases": []},
    {"pattern": r"\bIoT\b", "label": "전문 약어", "replacement": "사물인터넷", "confidence": 0.95, "source": {"doc": "전문 약어", "page": 1, "quote": "전문 약어는 한글 풀이로 대체"}, "aliases": []},
    {"pattern": r"\bAI\b", "label": "전문 약어", "replacement": "인공지능", "confidence": 0.95, "source": {"doc": "전문 약어", "page": 1, "quote": "전문 약어는 한글 풀이로 대체"}, "aliases": []},
    {"pattern": r"\bVR\b", "label": "전문 약어", "replacement": "가상현실", "confidence": 0.95, "source": {"doc": "전문 약어", "page": 1, "quote": "전문 약어는 한글 풀이로 대체"}, "aliases": []},
    {"pattern": r"\bAR\b", "label": "전문 약어", "replacement": "증강현실", "confidence": 0.95, "source": {"doc": "전문 약어", "page": 1, "quote": "전문 약어는 한글 풀이로 대체"}, "aliases": []},
    {"pattern": r"\bDNA\b", "label": "전문 약어", "replacement": "디옥시리보핵산", "confidence": 0.95, "source": {"doc": "전문 약어", "page": 1, "quote": "전문 약어는 한글 풀이로 대체"}, "aliases": []},
    {"pattern": r"\bRNA\b", "label": "전문 약어", "replacement": "리보핵산", "confidence": 0.95, "source": {"doc": "전문 약어", "page": 1, "quote": "전문 약어는 한글 풀이로 대체"}, "aliases": []},
    {"pattern": r"\bPCR\b", "label": "전문 약어", "replacement": "중합효소연쇄반응", "confidence": 0.95, "source": {"doc": "전문 약어", "page": 1, "quote": "전문 약어는 한글 풀이로 대체"}, "aliases": []},
    # --- 학술 용어(일반화) ---
    {"pattern": r"(?:CRISPR-?Cas9|크리스퍼-?카스9?)", "label": "전문 용어", "replacement": "유전자 가위 기술", "confidence": 0.93, "source": {"doc": "학술 용어 일반화", "page": 1, "quote": "과도한 전문용어는 일반화/설명적 표현 사용 권장"}, "aliases": ["crispr", "cas9", "크리스퍼"]},
    # --- 기재불가 공인어학시험 (v2.0.2) ---
    {"pattern": r"(?:TOEIC|TOEFL|TEPS|토익|토플|탭스|토익시험|토플시험|탭스시험)", "label": "공인어학시험", "replacement": "", "confidence": 0.99, "source": {"doc": "어학시험 기재불가", "page": 1, "quote": "공인어학시험 성적 기재 불가"}, "aliases": ["토익점수", "토플점수", "탭스점수"], "delete_with_particle": True},
    {"pattern": r"(?:HSK|에이치에스케이)", "label": "공인어학시험", "replacement": "", "confidence": 0.99, "source": {"doc": "어학시험 기재불가", "page": 1, "quote": "공인어학시험 성적 기재 불가"}, "aliases": [], "delete_with_particle": True},
    {"pattern": r"(?:JPT|JLPT|제이피티|제이엘피티)", "label": "공인어학시험", "replacement": "", "confidence": 0.99, "source": {"doc": "어학시험 기재불가", "page": 1, "quote": "공인어학시험 성적 기재 불가"}, "aliases": [], "delete_with_particle": True},
    {"pattern": r"(?:DELF|DALF|델프|달프)", "label": "공인어학시험", "replacement": "", "confidence": 0.99, "source": {"doc": "어학시험 기재불가", "page": 1, "quote": "공인어학시험 성적 기재 불가"}, "aliases": [], "delete_with_particle": True},
    {"pattern": r"(?:ZD|TESTDAF|DSH|DSD|테스트다프)", "label": "공인어학시험", "replacement": "", "confidence": 0.99, "source": {"doc": "어학시험 기재불가", "page": 1, "quote": "공인어학시험 성적 기재 불가"}, "aliases": [], "delete_with_particle": True},
    {"pattern": r"(?:TORFL|토르플)", "label": "공인어학시험", "replacement": "", "confidence": 0.99, "source": {"doc": "어학시험 기재불가", "page": 1, "quote": "공인어학시험 성적 기재 불가"}, "aliases": [], "delete_with_particle": True},
    {"pattern": r"(?:DELE|델레)", "label": "공인어학시험", "replacement": "", "confidence": 0.99, "source": {"doc": "어학시험 기재불가", "page": 1, "quote": "공인어학시험 성적 기재 불가"}, "aliases": [], "delete_with_particle": True},
    {"pattern": r"(?:상공회의소\s?한자시험|한자능력검정|한능검|실용한자|한자급수자격검정|YBM\s?상무한검|한자급수인증시험|한자자격검정)", "label": "공인어학시험", "replacement": "", "confidence": 0.99, "source": {"doc": "어학시험 기재불가", "page": 1, "quote": "공인어학시험 성적 기재 불가"}, "aliases": ["한자시험", "한검", "한능검시험"], "delete_with_particle": True},
    # v2.0.6: 추가 어학시험 약어
    {"pattern": r"(?:OPIC|오픽|OPIc)", "label": "공인어학시험", "replacement": "", "confidence": 0.99, "source": {"doc": "어학시험 기재불가", "page": 1, "quote": "공인어학시험 성적 기재 불가"}, "aliases": ["오픽시험", "오픽점수"], "delete_with_particle": True},
    {"pattern": r"(?:FLEX|플렉스)", "label": "공인어학시험", "replacement": "", "confidence": 0.99, "source": {"doc": "어학시험 기재불가", "page": 1, "quote": "공인어학시험 성적 기재 불가"}, "aliases": ["플렉스시험"], "delete_with_particle": True},
    {"pattern": r"(?:SNULT|스널트)", "label": "공인어학시험", "replacement": "", "confidence": 0.99, "source": {"doc": "어학시험 기재불가", "page": 1, "quote": "공인어학시험 성적 기재 불가"}, "aliases": [], "delete_with_particle": True},
    {"pattern": r"(?:IELTS|아이엘츠)", "label": "공인어학시험", "replacement": "", "confidence": 0.99, "source": {"doc": "어학시험 기재불가", "page": 1, "quote": "공인어학시험 성적 기재 불가"}, "aliases": ["아엘츠"], "delete_with_particle": True},
    {"pattern": r"(?:TOPIK|토픽|한국어능력시험)", "label": "공인어학시험", "replacement": "", "confidence": 0.99, "source": {"doc": "어학시험 기재불가", "page": 1, "quote": "공인어학시험 성적 기재 불가"}, "aliases": ["토픽시험"], "delete_with_particle": True},
    {"pattern": r"(?:GTQ|지티큐|ITQ|아이티큐)", "label": "자격시험", "replacement": "", "confidence": 0.99, "source": {"doc": "자격시험 기재불가", "page": 1, "quote": "민간자격시험 성적 기재 불가"}, "aliases": [], "delete_with_particle": True},
    {"pattern": r"(?:MOS|모스|컴활|컴퓨터활용능력|워드프로세서|워프)", "label": "자격시험", "replacement": "", "confidence": 0.99, "source": {"doc": "자격시험 기재불가", "page": 1, "quote": "민간자격시험 성적 기재 불가"}, "aliases": ["컴활시험", "모스자격증"], "delete_with_particle": True},
    # --- v2.0.4: 누락된 기재 유의어 전체 추가 ---
    # 메타버스 플랫폼
    {"pattern": r"(?:Gather\s?Town|개더타운|ZEPETO|제페토|ifland|이프랜드)", "label": "상호명", "replacement": "메타버스 플랫폼", "confidence": 0.95, "source": {"doc": "대체표현", "page": 1, "quote": "Gather Town(개더타운), ZEPETO(제페토) 등 → 메타버스 플랫폼"}, "aliases": ["게더타운", "제페토앱"]},
    # Google TV 추가
    {"pattern": r"(?:Google\s?TV|구글\s?티비)", "label": "상호명", "replacement": "동영상 플랫폼", "confidence": 0.95, "source": {"doc": "대체표현", "page": 1, "quote": "Google TV(구글 티비) 등 → 동영상 플랫폼"}, "aliases": []},
    # 영상 편집 프로그램 추가 (Vllo, Final Cut Pro)
    {"pattern": r"(?:Vllo|블로|Final\s?Cut\s?Pro|파이널\s?컷\s?프로)", "label": "프로그램명", "replacement": "영상 편집 프로그램", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "Vllo(블로), Final Cut Pro(파이널 컷 프로) 등 → 영상 편집 프로그램"}, "aliases": ["파컷프로", "fcpx"]},
    # 협업 플랫폼
    {"pattern": r"(?:Padlet|패들렛|ThinkerBell|띵커벨|Allo|알로)", "label": "상호명", "replacement": "온라인 협업 플랫폼", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "Padlet(패들렛), ThinkerBell(띵커벨), Allo(알로) 등 → 협업 플랫폼"}, "aliases": ["패들릿"]},
    # 진로정보망
    {"pattern": r"(?:careernet|커리어넷|majormap|메이저맵)", "label": "상호명", "replacement": "진로 정보 사이트", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "careernet(커리어넷), majormap(메이저맵) 등 → 진로정보망, 진로 정보 사이트"}, "aliases": ["커리어넷검사", "메이저맵검사"]},
    # 직업선호도 검사
    {"pattern": r"(?:Holland|홀랜드)\s?검사", "label": "검사명", "replacement": "직업선호도 검사", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "Holland(홀랜드) 검사 등 → 직업선호도 검사"}, "aliases": ["홀란드검사", "홀랜드직업검사"]},
    # MBTI
    {"pattern": r"\b(?:MBTI|엠비티아이)\b", "label": "검사명", "replacement": "성격유형 검사", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "MBTI(엠비티아이) 등 → 성격유형 검사"}, "aliases": ["mbti검사", "엠비티아이검사"]},
    # HTML
    {"pattern": r"\b(?:HTML|에이치티엠엘)\b", "label": "프로그램명", "replacement": "웹 페이지 제작 언어", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "HTML(에이치티엠엘) 등 → 하이퍼텍스트 마크업 언어, 웹 페이지 제작 언어"}, "aliases": ["html5", "에치티엠엘"]},
    # CSS
    {"pattern": r"\b(?:CSS|씨에스에스)\b", "label": "프로그램명", "replacement": "스타일 시트 언어", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "CSS(씨에스에스) 등 → 스타일 시트 언어"}, "aliases": ["css3", "씨에쎄스"]},
    # 태블릿PC
    {"pattern": r"(?:iPad|아이패드|Galaxy\s?Tab|갤럭시\s?탭)", "label": "상호명", "replacement": "태블릿PC", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "iPad(아이패드), Galaxy Tab(갤럭시탭) 등 → 태블릿PC"}, "aliases": ["갤탭", "아이패드프로"]},
    # 크롬북
    {"pattern": r"(?:chrome\s?book|크롬북)", "label": "상호명", "replacement": "휴대용 컴퓨터", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "chrome book(크롬북) 등 → 휴대용 컴퓨터"}, "aliases": ["chromebook"]},
    # 가상화폐
    {"pattern": r"(?:Altcoin|알트코인|Bitcoin|비트코인|이더리움|Ethereum|리플|Ripple|도지코인|Dogecoin)", "label": "상호명", "replacement": "가상화폐", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "Altcoin(알트코인), Bitcoin(비트코인) 등 → 가상화폐"}, "aliases": ["비코", "잡코인", "암호화폐"]},
]

# =========================
# Core AI Engine Logic + Heuristics
# =========================

# --- Preview/Apply thresholds ---
MIN_PREVIEW_CONF = float(os.getenv("MIN_PREVIEW_CONF", "0.90"))

# --- Korean token heuristics (to tame semantic) ---
_JOSA_RE = re.compile(r"(으로|라서|라며|라고|이라|라|을|를|은|는|이|가|에|에서|에게|께서|로|와|과|도|만|까지|부터|처럼|보다|께|한테|에게서|이다|함)$")
STOPWORDS_KO = {
    "프로그램", "개발", "진행", "통해", "작성", "정리", "제출", "보고서",
    "발표", "이동", "제작", "편집", "마무리", "활용", "참조", "옮겼다",
    "초안", "정리하고", "옮김", "검토", "결과", "내용", "학습", "활동"
}


def _normalize_ko_token(tok: str) -> str:
    if re.fullmatch(r"[가-힣]+", tok):
        return _JOSA_RE.sub("", tok)
    return tok


def _should_consider_token(tok: str) -> bool:
    base = _normalize_ko_token(tok)
    if base in STOPWORDS_KO:
        return False
    if re.fullmatch(r"[가-힣]+", tok):
        if not re.search(r"(톡|그램|북|넷플|왓챠|웨일온|유튭|유튜브|티빙|캔바|키네|틱톡|페북|인스타|클래스룸|코랩|주피터|파이참)$", base):
            return False
    return True


# --- Build embedding alias index (for fuzzy) ---
def _build_alias_index():
    if _embedder is None:
        return None, None
    alias_rules, alias_texts = [], []
    for rule in RULES:
        for alias in rule.get("aliases", []):
            if text := alias.strip():
                alias_texts.append(text)
                alias_rules.append(rule)
    if not alias_texts:
        return None, None
    print(f"Building embedding index for {len(alias_texts)} aliases...")
    alias_embeddings = _embedder.encode(alias_texts, normalize_embeddings=True, show_progress_bar=False)
    print("Embedding index built successfully.")
    return np.array(alias_embeddings, dtype=np.float32), alias_rules


_ALIAS_EMB_INDEX, _ALIAS_RULES = _build_alias_index()

# --- Build exact alias map (for safe auto-replace of common typos like 'yutube') ---
_ALIAS_EXACT_MAP = {}
for rule in RULES:
    for a in rule.get("aliases", []):
        if not a:
            continue
        _ALIAS_EXACT_MAP[a.lower()] = rule


def regex_match(text: str) -> List[Hit]:
    hits: List[Hit] = []
    for rule in RULES:
        for match in re.finditer(rule["pattern"], text, flags=re.IGNORECASE):
            src = rule["source"]
            hits.append(Hit(
                span=match.group(0), label=rule["label"], replacement=rule.get("replacement"),
                confidence=float(rule.get("confidence", 0.9)),
                source=Source(doc=src.get("doc", ""), page=src.get("page"), quote=src.get("quote", "")),
                start=match.start(), end=match.end(),
                delete_with_particle=rule.get("delete_with_particle", False)
            ))
    return hits


# --- 알려지지 않은 영문 약어 감지 ---
_KNOWN_ABBREVS = set()
for rule in RULES:
    pattern = rule.get("pattern", "")
    # \bXXX\b 형태에서 추출
    matches = re.findall(r'\\b([A-Z][A-Z0-9]{1,10})\\b', pattern)
    _KNOWN_ABBREVS.update(matches)
    # (?:XXX|YYY) 형태에서 추출
    matches = re.findall(r'(?:^|[|(?:])([A-Z][A-Z0-9]{1,10})(?:[|)]|$)', pattern)
    _KNOWN_ABBREVS.update(matches)

# 일반 영어 단어 (약어 아님)
_COMMON_ENGLISH = {"THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN", "HAD", "HER", "WAS", "ONE", "OUR", "OUT", "HAS", "HIS", "HOW", "ITS", "MAY", "NEW", "NOW", "OLD", "SEE", "WAY", "BOY", "DID", "GET", "HIM", "LET", "PUT", "SAY", "SHE", "TOO", "USE", "TOP", "END", "SET", "ADD"}


def detect_unknown_abbreviations(text: str, existing_hits: List[Hit]) -> List[Hit]:
    """2글자 이상 대문자 영문 약어 중 규칙에 없는 것 감지 (빨간줄 표시용)"""
    hits: List[Hit] = []
    covered = set()
    for h in existing_hits:
        for i in range(h.start, h.end):
            covered.add(i)

    for match in re.finditer(r'(?<![A-Za-z])([A-Z]{2,10})(?![A-Za-z])', text):
        abbrev = match.group(1)
        start, end = match.start(), match.end()
        if any(i in covered for i in range(start, end)):
            continue
        if abbrev in _KNOWN_ABBREVS or abbrev in _COMMON_ENGLISH:
            continue
        hits.append(Hit(
            span=abbrev,
            label="미확인 영문 약어",
            replacement=None,
            confidence=0.85,
            source=Source(doc="자동 감지", page=None, quote="영문 약어가 감지됨. 한글 표기 필요 여부 검토 필요."),
            start=start, end=end
        ))
    return hits


def alias_exact_match(text: str) -> List[Hit]:
    """Exact alias hits (auto-replace ok): e.g., 'yutube' -> YouTube category."""
    hits: List[Hit] = []
    if not _ALIAS_EXACT_MAP:
        return hits
    aliases = sorted(_ALIAS_EXACT_MAP.keys(), key=len, reverse=True)
    alt = "|".join(re.escape(a) for a in aliases)
    pattern = re.compile(rf"(?<![A-Za-z0-9가-힣])({alt})(?![A-Za-z0-9가-힣])", re.IGNORECASE)
    for m in pattern.finditer(text):
        key = m.group(1).lower()
        rule = _ALIAS_EXACT_MAP.get(key)
        if not rule:
            continue
        src = rule["source"]
        hits.append(Hit(
            span=m.group(0), label=rule["label"], replacement=rule.get("replacement"),
            confidence=0.94,
            source=Source(doc=src.get("doc", ""), page=src.get("page"), quote=src.get("quote", "")),
            start=m.start(), end=m.end()
        ))
    return hits


def _get_semantic_candidates(text: str, min_len=2, max_len=30):
    for match in re.finditer(r"\b[A-Za-z가-힣0-9][A-Za-z가-힣0-9\.\-_/()]*\b", text):
        token = match.group(0).strip()
        if not (min_len <= len(token) <= max_len):
            continue
        if not _should_consider_token(token):
            continue
        yield token, match.start(), match.end()


def semantic_match(text: str, threshold: float = 0.86, max_hits: int = 10) -> List[Hit]:
    if not (_embedder and _ALIAS_EMB_INDEX is not None):
        return []
    candidates = list(_get_semantic_candidates(text))
    if not candidates:
        return []
    cand_tokens = [c[0] for c in candidates]
    cand_embeddings = _embedder.encode(cand_tokens, normalize_embeddings=True, show_progress_bar=False)
    sim_matrix = np.matmul(np.array(cand_embeddings, dtype=np.float32), _ALIAS_EMB_INDEX.T)
    hits: List[Hit] = []
    used_spans = set()
    for i, row in enumerate(sim_matrix):
        if len(hits) >= max_hits:
            break
        best_idx = int(np.argmax(row))
        score = float(row[best_idx])
        second = float(np.partition(row, -2)[-2]) if row.size > 1 else 0.0
        if not (score >= threshold and (score - second) >= 0.06):
            continue
        span_text, start, end = candidates[i]
        if (start, end) in used_spans:
            continue
        matched_rule = _ALIAS_RULES[best_idx]
        conf = min(0.88, max(0.6, score * 0.85))  # < 0.90 → preview highlight only
        hits.append(Hit(
            span=span_text, label=matched_rule["label"], replacement=matched_rule.get("replacement"),
            confidence=float(conf), source=Source(**matched_rule["source"]),
            start=start, end=end
        ))
        used_spans.add((start, end))
    return hits


def collapse_parenthetical_duplicates(text: str, hits: List[Hit]) -> List[Hit]:
    """
    Collapse patterns like '유엔(UN)' → one replacement ('국제기구'),
    instead of replacing both '유엔' and 'UN' → '국제기구(국제기구)'.
    Works when both inner and outer are hits with the same label & replacement.
    """
    hits = sorted(hits, key=lambda h: (h.start, h.end))
    n = len(hits)
    used = [False] * n
    result: List[Hit] = []
    i = 0
    while i < n:
        if used[i]:
            i += 1
            continue
        hi = hits[i]
        # look for "( ... )" right after hi
        k = hi.end
        # skip spaces
        while k < len(text) and text[k].isspace():
            k += 1
        if k < len(text) and text[k] == "(":
            # find a next hit wholly inside the parentheses
            j = i + 1
            while j < n and hits[j].start < k + 1:
                j += 1
            if j < n:
                hj = hits[j]
                # skip spaces after '('
                inner_start = k + 1
                while inner_start < len(text) and text[inner_start].isspace():
                    inner_start += 1
                # find closing ')'
                # allow spaces before ')'
                inner_end = hj.end
                tmp = inner_end
                while tmp < len(text) and text[tmp].isspace():
                    tmp += 1
                if tmp < len(text) and text[tmp] == ")":
                    close_pos = tmp
                    # verify both hits share same label+replacement and hj is inside the parens
                    if (hj.start >= inner_start) and (hj.end <= close_pos) and \
                       (hi.label == hj.label) and (hi.replacement == hj.replacement) and \
                       (hi.replacement is not None):
                        # create combined hit
                        combined = Hit(
                            span=text[hi.start:close_pos + 1],
                            label=hi.label,
                            replacement=hi.replacement,
                            confidence=max(hi.confidence, hj.confidence),
                            source=hi.source,
                            start=hi.start,
                            end=close_pos + 1
                        )
                        result.append(combined)
                        used[i] = True
                        used[j] = True
                        i += 1
                        continue
        # default path: keep hi
        result.append(hi)
        used[i] = True
        i += 1
    # keep any not-yet-added hits
    for idx in range(n):
        if not used[idx]:
            result.append(hits[idx])
    # de-duplicate and sort
    seen = set()
    dedup: List[Hit] = []
    for h in sorted(result, key=lambda h: (h.start, h.end)):
        key = (h.start, h.end, h.label, h.replacement)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(h)
    return dedup


def merge_hits(*hit_groups: List[Hit]) -> List[Hit]:
    all_hits: List[Hit] = []
    for g in hit_groups:
        all_hits.extend(g)
    all_hits = sorted(all_hits, key=lambda h: (h.start, -(h.end - h.start), -h.confidence))
    merged: List[Hit] = []
    if not all_hits:
        return []
    last_hit_end = -1
    for hit in all_hits:
        if hit.start >= last_hit_end:
            merged.append(hit)
            last_hit_end = hit.end
    return merged


# =========================
# UI (HTML/CSS/JS)
# =========================
HTML_PAGE = r"""
<!doctype html><html lang="ko"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>생기부 금칙어 검사기 – v2.0.7</title>
<style>:root{--bg:#0b1020;--card:#111830;--ink:#e6edff;--muted:#9db1ff;--accent:#4f7cff;--hit:#ff4455;--ok:#25d366;--warn:#ffaa00}*{box-sizing:border-box}body{margin:0;font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Apple SD Gothic Neo,Noto Sans KR,sans-serif;background:var(--bg);color:var(--ink)}.wrap{max-width:1100px;margin:36px auto;padding:0 16px}.card{background:var(--card);border-radius:20px;padding:20px;box-shadow:0 10px 30px rgba(0,0,0,.35)}h1{margin:0 0 8px}.muted{color:var(--muted);font-size:12px}textarea{width:100%;min-height:160px;padding:14px;border-radius:14px;border:1px solid #263257;background:#0e1430;color:var(--ink);font-size:16px;resize:vertical}button{background:var(--accent);color:white;border:0;padding:12px 16px;border-radius:12px;font-weight:700;cursor:pointer}button:disabled{opacity:.6;cursor:not-allowed}.row{display:flex;gap:12px;flex-wrap:wrap;align-items:center}.grid{margin-top:16px;display:grid;grid-template-columns:1fr 1fr 320px;gap:16px}@media (max-width: 900px) {.grid{grid-template-columns: 1fr;}}.panel{background:#0e1430;border:1px solid #263257;border-radius:14px;padding:14px}mark{background:transparent;color:var(--hit);font-weight:800;text-decoration:underline;text-underline-offset:3px}ins.rep{background:#0f2a1f;color:#b2ffd8;text-decoration:none;border-bottom:2px solid var(--ok);padding:0 2px}.hit{display:flex;justify-content:space-between;gap:8px;border-bottom:1px dashed #263257;padding:8px 0}.pill{font-size:12px;padding:3px 8px;border-radius:999px;background:#1b2342;color:#c7d3ff}.byte-box{background:linear-gradient(135deg,#1a2744 0%,#0e1430 100%);border:1px solid #263257;border-radius:14px;padding:16px;margin-top:12px}.byte-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px}.byte-item{text-align:center;padding:12px;background:#0b1020;border-radius:10px}.byte-value{font-size:28px;font-weight:800;color:var(--accent)}.byte-label{font-size:11px;color:var(--muted);margin-top:4px}.byte-warn{color:var(--warn)}.suspicious-list{margin-top:12px;font-size:12px;color:var(--warn)}.suspicious-item{padding:4px 0;border-bottom:1px dashed #263257}

.panel-head{display:flex;align-items:center;gap:8px;margin-bottom:8px}
.btn-mini{padding:6px 10px;border-radius:10px;font-size:12px;font-weight:700}

</style></head><body>
<div class="wrap">
<h1>생기부 금칙어 검사기 <span style="font-size:14px;color:var(--accent)">(v2.0.7)</span></h1>
<div class="card">
<div class="muted">본문을 붙여넣고 "검사"를 누르세요 · <b>바이트 수</b>와 <b>금칙어</b>를 동시에 검사합니다</div>
<textarea id="txt"></textarea>
<div class="row" style="margin-top:8px">
<button id="btn">검사</button>
<button id="btnApplyAll" disabled>모두 적용</button>
<button id="btnSample">샘플 텍스트</button>
<span id="lat" class="muted">–</span>
<span id="chg" class="muted">변경 0건</span>
</div>
<!-- 바이트 계산 결과 영역 (v2.0) -->
<div class="byte-box" id="byteBox" style="display:none">
<div style="margin-bottom:12px;font-weight:700">바이트 계산 결과 <span class="muted">(생활기록부 UTF-8 기준)</span></div>
<div class="byte-grid">
<div class="byte-item"><div class="byte-value" id="byteUtf8">-</div><div class="byte-label">UTF-8 바이트</div></div>
</div>
<div id="suspiciousArea" style="display:none">
<div class="suspicious-list">
<div style="margin-bottom:8px;font-weight:600;color:var(--warn)">⚠️ 의심 문자 감지됨 (보이지 않는 특수문자)</div>
<div id="suspiciousList"></div>
</div>
</div>
</div>
<div class="grid">
<div class="panel"><div class="muted" style="margin-bottom:8px">하이라이트 결과(원문)</div><div id="view" style="line-height:1.8; white-space:pre-wrap;"></div></div>

<div class="panel">
  <div class="panel-head">
    <div class="muted">수정본 미리보기(대체어 적용)</div>
    <button id="btnCopyPreview" class="btn-mini" type="button">복사하기</button>
  </div>
  <div id="preview" style="line-height:1.8; white-space:pre-wrap;"></div>
</div>


<div class="panel"><div class="muted" style="margin-bottom:8px">근거 / 대체표현</div><div id="hits"></div></div>
</div>
</div>
</div>
<script>
const POLICY="2024-03";
const MIN_PREVIEW_CONF = 0.90; // 자동 치환 기준


let currentHits = [];
const txtEl = document.getElementById("txt");

// --- Copy preview to clipboard ---
const previewEl = document.getElementById("preview");
const btnCopyPreview = document.getElementById("btnCopyPreview");

btnCopyPreview.onclick = async function () {
  const text = (previewEl && (previewEl.innerText || previewEl.textContent) || "").trim();
  if (!text) {
    alert("복사할 내용이 없습니다. 먼저 '검사'를 실행하세요.");
    return;
  }

  // 1) Modern Clipboard API (works on https or localhost)
  try {
    await navigator.clipboard.writeText(text);
    const old = this.textContent;
    this.textContent = "복사됨";
    setTimeout(() => (this.textContent = old), 1200);
    return;
  } catch (e) {
    // 2) Fallback (older browsers / permission issues)
    try {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.left = "-9999px";
      ta.style.top = "0";
      document.body.appendChild(ta);
      ta.focus();
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);

      const old = this.textContent;
      this.textContent = "복사됨";
      setTimeout(() => (this.textContent = old), 1200);
      return;
    } catch (e2) {
      alert("복사에 실패했습니다. 브라우저 권한 설정을 확인하세요.");
      console.error(e2);
    }
  }
};

function esc(s){return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");}



async function analyze(text){const r=await fetch("/analyze",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({text:text,policy_version:POLICY})});if(!r.ok)throw new Error(`API Error: ${r.statusText}`);return await r.json();}
// --- Byte Counter API (v2.0) ---
async function countBytes(text){const r=await fetch("/byte-count",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({text:text,normalize:false})});if(!r.ok)throw new Error(`API Error: ${r.statusText}`);return await r.json();}
function renderByteResults(data){
document.getElementById("byteBox").style.display="block";
document.getElementById("byteUtf8").textContent=(data.utf8_bytes).toLocaleString();

const suspArea=document.getElementById("suspiciousArea");
const suspList=document.getElementById("suspiciousList");
if(data.suspicious && data.suspicious.length>0){
suspArea.style.display="block";
suspList.innerHTML=data.suspicious.map(s=>`<div class="suspicious-item">위치 ${s.index}: <b>${s.name}</b> (${s.codepoint})</div>`).join("");
}else{
suspArea.style.display="none";
suspList.innerHTML="";
}
}
// --- Hangul helpers for postposition (조사) auto-fix ---
function _lastHangulHasBatchim(word){
word = (word||"").trim();
if(!word) return false;
const ch = word[word.length-1].charCodeAt(0);
if (ch < 0xAC00 || ch > 0xD7A3) return false;
const jong = (ch - 0xAC00) % 28;
return jong !== 0;
}
function _lastHangulIsRieul(word){
word = (word||"").trim();
if(!word) return false;
const ch = word[word.length-1].charCodeAt(0);
if (ch < 0xAC00 || ch > 0xD7A3) return false;
const jong = (ch - 0xAC00) % 28;
return jong === 8; // ㄹ
}
function chooseParticle(baseWord, particle){
const hasBatchim = _lastHangulHasBatchim(baseWord);
switch(particle){
case "로": case "으로":
if(!hasBatchim || _lastHangulIsRieul(baseWord)) return "로";
return "으로";
case "와": case "과":
return hasBatchim ? "과" : "와";
case "는": case "은":
return hasBatchim ? "은" : "는";
case "가": case "이":
return hasBatchim ? "이" : "가";
case "를": case "을":
return hasBatchim ? "을" : "를";
default:
return particle;
}
}
// when auto-replacing, adjust the following particle if present
function spliceWithParticle(text, start, end, replacement){
const look = text.slice(end, end+2);
const m = look.match(/^(으로|로|을|를|은|는|이|가|과|와)/);
if(m){
const fixed = chooseParticle(replacement, m[0]);
return {
newText: text.slice(0, start) + replacement + fixed + text.slice(end + m[0].length),
skip: m[0].length,
appended: fixed
};
}
return {
newText: text.slice(0, start) + replacement + text.slice(end),
skip: 0,
appended: ""
};
}
// v2.0.2: 조사 패턴 (삭제 시 함께 삭제)
const PARTICLE_PATTERN = /^(으로|에서|에게|라서|라며|라고|이라|로|을|를|은|는|이|가|과|와|에|도|만|까지|부터|처럼|보다|께|한테|라)/;

// v2.0.6: 금지어 삭제 후 어색한 문장 정리 (강화)
function cleanupAfterDeletion(text) {
// 1. 시험 관련 문장 패턴 완전 삭제
// "시험봐서 좋은 점수를 받았고" 같은 패턴
text = text.replace(/시험봐서[^.]*?(받았고|받았다|받음)[,.]?\s*/g, '');
text = text.replace(/준비했다[,.]?\s*/g, '');
text = text.replace(/도\s*준비했다[,.]?\s*/g, '');

// 2. "N급을 취득했다", "N점을 받았다" 등 앞에 시험명이 없으면 삭제
text = text.replace(/\d+급을?\s*(취득했다|취득함|땄다|딸 수 있었다)[,.]?\s*/g, '');
text = text.replace(/\d+점을?\s*(받았다|취득했다|획득했다)[,.]?\s*/g, '');
text = text.replace(/에서\s*\d+급/g, '');
text = text.replace(/에서\s*\d+점/g, '');

// 3. 빈 절 제거: ", 준비했다" -> ""
text = text.replace(/,\s*(시험봐서|준비했다|취득했다|응시했다|합격했다|불합격했다|통과했다)[^,.\n]*/g, '');

// 4. 문장 시작이 어색한 경우 정리
text = text.replace(/^\s*(시험봐서|준비했다|취득했다|응시했다)[^,.\n]*[,.]\s*/gm, '');

// 5. 의미없는 조각 문장 제거
text = text.replace(/,\s*,/g, ',');
text = text.replace(/\.\s*\./g, '.');
text = text.replace(/,\s*\./g, '.');
text = text.replace(/^\s*,\s*/gm, '');
text = text.replace(/,\s*$/gm, '.');

// 6. "좋은 점수를 받았고," 처럼 앞 문맥이 삭제된 경우
text = text.replace(/^\s*좋은 점수를 받았고[,.]?\s*/gm, '');

// 7. 연속 공백 정리
text = text.replace(/\s{2,}/g, ' ');

// 8. 문장 시작 공백 제거
text = text.replace(/^\s+/gm, '');

// 9. 빈 줄 제거
text = text.replace(/\n\s*\n/g, '\n');

return text.trim();
}

// v2.0.7: 문맥상 어색한 조사 자동 보정
function fixAwkwardParticles(text) {
// "시험봐서" 등 앞에 주어가 없는 경우 삭제
text = text.replace(/^\s*시험봐서/gm, '');
// 문장 시작이 조사로 시작하는 경우 삭제
text = text.replace(/^\s*(을|를|은|는|이|가|에서|에게|로|으로)\s+/gm, '');
// 연속 조사 정리
text = text.replace(/(을|를)\s+(을|를)/g, '$1');
text = text.replace(/(은|는)\s+(은|는)/g, '$1');
// 빈 괄호 제거
text = text.replace(/\(\s*\)/g, '');
// 연속 공백
text = text.replace(/\s{2,}/g, ' ');
return text.trim();
}

// v2.0.7: 중복 대체어 병합 - "프로그래밍 언어 및 프로그래밍 언어" -> "프로그래밍 언어"
function removeDuplicateReplacements(text) {
// 먼저 "및" 앞뒤에 공백이 없는 경우 공백 추가 (예: "언어및JS" -> "언어 및 JS")
text = text.replace(/(\S)및(\S)/g, '$1 및 $2');
text = text.replace(/(\S)및\s/g, '$1 및 ');
text = text.replace(/\s및(\S)/g, ' 및 $1');

// v2.0.7: "프로그래밍 언어 및 프로그래밍 언어도" -> "프로그래밍 언어도" (조사 포함)
text = text.replace(/(프로그래밍 언어)\s*및\s*\1(도|를|을|은|는|이|가|로|으로)?/g, '$1$2');
text = text.replace(/(프로그래밍 언어)(도|를|을|은|는|이|가|로|으로)?\s*및\s*\1/g, '$1');

// 패턴: 2~3단어로 이루어진 대체어 중복 제거 (예: "프로그래밍 언어 및 프로그래밍 언어")
const patterns = [
// 2단어 이상 대체어 + 조사 포함
/(\S+\s+\S+)\s*및\s*\1(도|를|을|은|는|이|가)?/g,
/(\S+\s+\S+)(도|를|을|은|는|이|가)?\s*및\s*\1/g,
// 2단어 이상 대체어: "프로그래밍 언어 및 프로그래밍 언어"
/(\S+\s+\S+)\s*및\s*\1/g,
/(\S+\s+\S+)\s*,\s*\1/g,
/(\S+\s+\S+)\s*그리고\s*\1/g,
/(\S+\s+\S+)\s*와\s*\1/g,
/(\S+\s+\S+)\s*과\s*\1/g,
// 1단어 대체어
/(\S+)\s*및\s*\1/g,
/(\S+)\s*,\s*\1/g,
/(\S+)\s*그리고\s*\1/g,
/(\S+)\s*와\s*\1/g,
/(\S+)\s*과\s*\1/g,
];
let result = text;
// 여러 번 반복 적용 (중첩된 경우 대비)
for (let i = 0; i < 3; i++) {
for (const p of patterns) {
result = result.replace(p, '$1$2');
}
}
// 연속 동일 2단어 제거 (예: "프로그래밍 언어 프로그래밍 언어")
result = result.replace(/(\S+\s+\S+)\s+\1/g, '$1');
// 연속 동일 1단어 제거
result = result.replace(/(\S+)\s+\1/g, '$1');
// "X도 사용했다" 에서 X가 대체어와 같으면 "X도 사용했다"로 정리
result = result.replace(/(\S+\s+\S+)\s*도\s+\1/g, '$1');
// undefined 제거 (캡처 그룹이 없는 경우)
result = result.replace(/undefined/g, '');
return result;
}

function renderResults(text, hits) {
const sortedHits = [...hits].sort((a, b) => a.start - b.start);
let viewLastIndex = 0;  // v2.0.7: 원문용 별도 인덱스
let previewLastIndex = 0;  // v2.0.7: 미리보기용 별도 인덱스
const viewParts = [];
const previewParts = [];
for (const hit of sortedHits) {
if (hit.start < viewLastIndex) continue; // overlapped (safety)

// v2.0.7: 원문은 항상 원본 텍스트 그대로 (조사 포함)
if (hit.start > viewLastIndex) {
viewParts.push(esc(text.slice(viewLastIndex, hit.start)));
}
viewParts.push(`<mark title="${esc(hit.label)}">${esc(hit.span)}</mark>`);
viewLastIndex = hit.end;

// v2.0.7: 미리보기는 별도 처리
if (hit.start > previewLastIndex) {
previewParts.push(esc(text.slice(previewLastIndex, hit.start)));
}

// v2.0.2: 삭제 처리 (replacement가 빈 문자열인 경우)
const isDelete = hit.replacement === "" && hit.delete_with_particle;
const canReplace = (hit.replacement !== null && hit.replacement !== undefined) && (hit.confidence >= MIN_PREVIEW_CONF);

if (isDelete && hit.confidence >= MIN_PREVIEW_CONF) {
// 삭제 시 뒤따르는 조사도 함께 삭제 (미리보기에서만)
const look = text.slice(hit.end, hit.end+3);
const m = look.match(PARTICLE_PATTERN);
if(m){
previewLastIndex = hit.end + m[0].length;
// 조사 삭제 후 남은 공백 처리
const nextChar = text[previewLastIndex];
if(nextChar === ' ') {
const prevPart = previewParts[previewParts.length - 1] || '';
if(prevPart.endsWith(' ') || prevPart.endsWith('&gt;')) {
previewLastIndex++;
}
}
}else{
previewLastIndex = hit.end;
}
// 삭제된 내용은 preview에 아무것도 추가하지 않음 (완전 삭제)
} else if (canReplace && hit.replacement !== "") {
const look = text.slice(hit.end, hit.end+2);
const m = look.match(/^(으로|로|을|를|은|는|이|가|과|와)/);
if(m){
const appended = chooseParticle(hit.replacement, m[0]);
previewParts.push(`<ins class="rep" title="${esc(hit.label)}">${esc(hit.replacement + appended)}</ins>`);
previewLastIndex = hit.end + m[0].length;
}else{
previewParts.push(`<ins class="rep" title="${esc(hit.label)}">${esc(hit.replacement)}</ins>`);
previewLastIndex = hit.end;
}
} else {
previewParts.push(`<mark title="${esc(hit.label)}">${esc(hit.span)}</mark>`);
previewLastIndex = hit.end;
}
}
// v2.0.7: 원문과 미리보기 각각 나머지 텍스트 추가
if (viewLastIndex < text.length) {
viewParts.push(esc(text.slice(viewLastIndex)));
}
if (previewLastIndex < text.length) {
previewParts.push(esc(text.slice(previewLastIndex)));
}
document.getElementById("view").innerHTML = viewParts.join("").replace(/\n/g, "<br>");
// v2.0.7: 미리보기에서 중복 대체어 제거 (전체 텍스트 기준)
let previewHtml = previewParts.join("").replace(/\n/g, "<br>");
const previewEl = document.getElementById("preview");
previewEl.innerHTML = previewHtml;
// v2.0.7: 전체 텍스트를 추출해서 중복 제거 후 다시 적용
let fullText = previewEl.innerText || previewEl.textContent;
fullText = removeDuplicateReplacements(fullText);
fullText = cleanupAfterDeletion(fullText);
fullText = fixAwkwardParticles(fullText);  // v2.0.7: 조사 보정
// HTML 없이 순수 텍스트로 표시 (대체어 스타일은 제거되지만 정확한 결과 우선)
previewEl.innerHTML = esc(fullText).replace(/\n/g, "<br>");
const hitsEl = document.getElementById("hits");
hitsEl.innerHTML = "";
if (!hits.length) {
hitsEl.innerHTML = '<div class="muted">규정 위반 항목을 찾지 못했습니다.</div>';
return;
}
for (const h of hits) {
const row = document.createElement("div");
row.className = "hit";
const isDelete = h.replacement === "" && h.delete_with_particle;
const auto = ((h.replacement || isDelete) && h.confidence >= MIN_PREVIEW_CONF);
const conf = Math.round(h.confidence * 100);
let actionText = "";
if (isDelete) {
actionText = `<div class="pill" style="background:#3a1a1a;color:#ff6b6b">${auto ? '자동삭제' : '검토필요'}: [삭제+조사제거]</div>`;
} else if (h.replacement) {
actionText = `<div class="pill">${auto ? '자동적용' : '검토필요'}: ${esc(h.replacement)}</div>`;
}
row.innerHTML =
`<div>
<b>${esc(h.span)}</b> <span class="pill">${h.label}</span> <span class="pill">${conf}%</span><br/>
<span class="muted">${h.source.doc} p.${h.source.page||'?'}: ${esc(h.source.quote||'')}</span>
</div>` + actionText;
hitsEl.appendChild(row);
}
}
function applyAllReplacements() {
let text = txtEl.value;
const replacableHits = currentHits
.filter(h => (h.replacement !== null && h.replacement !== undefined) && h.confidence >= MIN_PREVIEW_CONF)
.sort((a,b) => b.start - a.start);
for (const hit of replacableHits) {
const isDelete = hit.replacement === "" && hit.delete_with_particle;
if (isDelete) {
// v2.0.2: 삭제 + 조사 제거
const look = text.slice(hit.end, hit.end+3);
const m = look.match(PARTICLE_PATTERN);
const endPos = m ? hit.end + m[0].length : hit.end;
text = text.slice(0, hit.start) + text.slice(endPos);
} else {
const sp = spliceWithParticle(text, hit.start, hit.end, hit.replacement);
text = sp.newText;
}
}
// v2.0.2: 중복 대체어 제거
text = removeDuplicateReplacements(text);
// v2.0.3: 어색한 문장 정리
text = cleanupAfterDeletion(text);
// 연속 공백 정리
text = text.replace(/  +/g, ' ').trim();
txtEl.value = text;
document.getElementById("btn").click();
}
document.getElementById("btnSample").onclick = function() {
txtEl.value = "유엔(UN) 보고서를 참조하여 챗GPT 초안 작성 후 MS워드 정리하고 Google Docs에 옮겼다.\nZoom(웨일온)으로 발표하고 yutube·Instagram에 홍보했다.\n이동은 KTX, 표지는 Canva 제작, 편집은 키네마스터 마무리했으며 소논문도 제출했다. 또한 Jupyter 통해 실험을 정리했고 CRISPR-Cas9 관련 내용을 참고했다. Java Script 및 JS도 사용했다.\n토익을 시험봐서 좋은 점수를 받았고, TOEFL도 준비했다. 한능검에서 2급을 취득했다.";
};
document.getElementById("btn").onclick = async function() {
const text = txtEl.value || "";
this.textContent = "검사 중...";
this.disabled = true;
try {
// 금칙어 검사 + 바이트 계산 동시 실행 (v2.0)
const [res, byteRes] = await Promise.all([analyze(text), countBytes(text)]);
currentHits = res.hits;
document.getElementById("lat").textContent = `처리시간: ${res.latency_ms} ms`;
renderResults(text, res.hits);
renderByteResults(byteRes);
const changedCount = currentHits.filter(h => (h.replacement !== null && h.replacement !== undefined) && h.confidence >= MIN_PREVIEW_CONF).length;
document.getElementById("chg").textContent = `변경 ${changedCount}건`;
const btnApply = document.getElementById("btnApplyAll");
btnApply.disabled = changedCount === 0;
btnApply.onclick = applyAllReplacements;
} catch (e) {
alert("오류가 발생했습니다. 잠시 후 다시 시도해주세요. (서버 기상 중일 수 있음)");
console.error(e);
} finally {
this.textContent = "검사";
this.disabled = false;
}
};
document.getElementById("btnSample").click();
</script></body></html>
"""

# =========================
# API Routes
# =========================
@app.get("/", response_class=HTMLResponse, summary="Main UI Page")
def home():
    return HTML_PAGE


@app.post("/analyze", response_model=AnalyzeResponse, summary="Analyze student record text")
def analyze(payload: AnalyzeRequest = Body(...)):
    t0 = time.perf_counter()
    # 1) Regex Analysis Unit
    hits_rule = regex_match(payload.text)
    # 2) Exact Alias Unit (safe auto-fix for common typos)
    hits_alias = alias_exact_match(payload.text)
    # 3) Collapse parenthetical duplicates like '유엔(UN)'
    primary_hits = collapse_parenthetical_duplicates(payload.text, hits_rule + hits_alias)
    # 4) Embedding Analysis Unit (conservative)
    hits_semantic = semantic_match(payload.text)
    # 5) Merge known hits
    known_hits = merge_hits(primary_hits, hits_semantic)
    # 6) Detect unknown abbreviations (v1.8.3)
    hits_unknown = detect_unknown_abbreviations(payload.text, known_hits)
    # 7) Final merge & respond
    final_hits = merge_hits(known_hits, hits_unknown)
    latency_ms = int((time.perf_counter() - t0) * 1000)
    return AnalyzeResponse(hits=final_hits, latency_ms=latency_ms)


@app.post("/byte-count", response_model=ByteCountResponse, summary="Count bytes for student record (v2.0)")
def byte_count(payload: ByteCountRequest = Body(...)):
    """생활기록부 전용 바이트 계산기 - UTF-8 기준"""
    result = analyze_bytes(payload.text)

    normalized_text = None
    normalized_bytes = None
    if payload.normalize:
        normalized_text = normalize_for_neis(payload.text, newline_mode=payload.newline_mode)
        normalized_bytes = utf8_byte_len(normalized_text)

    return ByteCountResponse(
        utf8_bytes=result["utf8_bytes"],
        char_count_including_spaces=result["char_count_including_spaces"],
        char_count_excluding_spaces=result["char_count_excluding_spaces"],
        newline_lf=result["newline_lf"],
        newline_cr=result["newline_cr"],
        tab=result["tab"],
        suspicious=result["suspicious"],
        normalized_text=normalized_text,
        normalized_utf8_bytes=normalized_bytes,
    )
