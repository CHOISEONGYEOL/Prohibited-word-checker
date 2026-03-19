import os
import re
from typing import List, Optional, Dict

import numpy as np

from schemas import Hit, Source

# =========================
# AI Model Loading
# =========================
# 경량 모델로 교체: bge-m3(2.2GB) → multilingual-e5-small(470MB)
# E5 prefix 자동 처리 지원
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


# =========================
# Encoding wrapper (E5 model prefix support)
# =========================
def _encode_texts(texts: list, is_query: bool = False):
    """Encode texts, adding query/passage prefix for E5 family models."""
    if "e5" in EMB_MODEL.lower():
        prefix = "query: " if is_query else "passage: "
        texts = [prefix + t for t in texts]
    return _embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)


# =========================
# Preview/Apply thresholds
# =========================
MIN_PREVIEW_CONF = float(os.getenv("MIN_PREVIEW_CONF", "0.90"))


# =========================
# Korean token heuristics (불용어 대폭 축소)
# =========================
_JOSA_RE = re.compile(
    r"(으로|라서|라며|라고|이라|라|을|를|은|는|이|가|에|에서|에게|께서|로|와|과|도|만|까지|부터|처럼|보다|께|한테|에게서|이다|함)$"
)

# 축소된 불용어: 절대로 금칙어/브랜드명이 될 수 없는 단어만 포함
# 기존 72개 → 30개로 축소 ("프로그램", "개발", "제작" 등 제거)
STOPWORDS_KO = {
    "학생", "선생님", "수업", "교실", "시간", "방법", "교육",
    "태도", "노력", "참여", "협력", "소통", "성장", "변화",
    "뛰어난", "탁월한", "다양한", "적극적", "자발적", "능동적",
    "과정", "단계", "주제", "목표", "친구", "선배", "후배",
    "배움", "성찰", "의견",
}


def _normalize_ko_token(tok: str) -> str:
    if re.fullmatch(r"[가-힣]+", tok):
        return _JOSA_RE.sub("", tok)
    return tok


def _should_consider_token(tok: str) -> bool:
    """블랙리스트 방식: 불용어만 제외, 나머지는 허용."""
    base = _normalize_ko_token(tok)
    if base in STOPWORDS_KO:
        return False
    # 한국어 단독 토큰: 2음절 이상만
    if re.fullmatch(r"[가-힣]+", tok):
        if len(base) < 2:
            return False
    # 순수 숫자 제외
    if re.fullmatch(r"[0-9]+", tok):
        return False
    return True


# =========================
# Module-level state (initialized by init_engine)
# =========================
_ALIAS_EMB_INDEX = None
_ALIAS_RULES = None
_ALIAS_EXACT_MAP: Dict[str, dict] = {}
_KNOWN_ABBREVS = set()
_COMMON_ENGLISH = {
    "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN", "HAD",
    "HER", "WAS", "ONE", "OUR", "OUT", "HAS", "HIS", "HOW", "ITS", "MAY",
    "NEW", "NOW", "OLD", "SEE", "WAY", "BOY", "DID", "GET", "HIM", "LET",
    "PUT", "SAY", "SHE", "TOO", "USE", "TOP", "END", "SET", "ADD",
}
_RULES = []


def init_engine(rules: list):
    """Initialize engine with rules. Call once at startup."""
    global _ALIAS_EMB_INDEX, _ALIAS_RULES, _ALIAS_EXACT_MAP, _KNOWN_ABBREVS, _RULES
    _RULES = rules

    # Build embedding alias index
    _ALIAS_EMB_INDEX, _ALIAS_RULES = _build_alias_index(rules)

    # Build exact alias map
    _ALIAS_EXACT_MAP.clear()
    for rule in rules:
        for a in rule.get("aliases", []):
            if not a:
                continue
            _ALIAS_EXACT_MAP[a.lower()] = rule

    # Build known abbreviations set
    _KNOWN_ABBREVS.clear()
    for rule in rules:
        pattern = rule.get("pattern", "")
        matches = re.findall(r'\\b([A-Z][A-Z0-9]{1,10})\\b', pattern)
        _KNOWN_ABBREVS.update(matches)
        matches = re.findall(r'(?:^|[|(?:])([A-Z][A-Z0-9]{1,10})(?:[|)]|$)', pattern)
        _KNOWN_ABBREVS.update(matches)


def _build_alias_index(rules: list):
    if _embedder is None:
        return None, None
    alias_rules, alias_texts = [], []
    for rule in rules:
        for alias in rule.get("aliases", []):
            if text := alias.strip():
                alias_texts.append(text)
                alias_rules.append(rule)
    if not alias_texts:
        return None, None
    print(f"Building embedding index for {len(alias_texts)} aliases...")
    alias_embeddings = _encode_texts(alias_texts, is_query=False)
    print("Embedding index built successfully.")
    return np.array(alias_embeddings, dtype=np.float32), alias_rules


# =========================
# Regex Matching
# =========================
def regex_match(text: str) -> List[Hit]:
    hits: List[Hit] = []
    for rule in _RULES:
        for match in re.finditer(rule["pattern"], text, flags=re.IGNORECASE):
            src = rule["source"]
            hits.append(Hit(
                span=match.group(0), label=rule["label"],
                replacement=rule.get("replacement"),
                confidence=float(rule.get("confidence", 0.9)),
                source=Source(
                    doc=src.get("doc", ""), page=src.get("page"),
                    quote=src.get("quote", ""),
                ),
                start=match.start(), end=match.end(),
                delete_with_particle=rule.get("delete_with_particle", False),
            ))
    return hits


# =========================
# Exact Alias Matching (조사 결합형 지원)
# =========================
_JOSA_SUFFIX_PATTERN = r"(?:은|는|이|가|을|를|에|에서|으로|로|와|과|도|만|까지|부터)?"


def alias_exact_match(text: str) -> List[Hit]:
    hits: List[Hit] = []
    if not _ALIAS_EXACT_MAP:
        return hits
    aliases = sorted(_ALIAS_EXACT_MAP.keys(), key=len, reverse=True)
    alt = "|".join(re.escape(a) for a in aliases)
    # 조사 결합형 지원: 별칭 뒤에 한국어 조사가 올 수 있음
    pattern = re.compile(
        rf"(?<![A-Za-z0-9가-힣])({alt}){_JOSA_SUFFIX_PATTERN}(?![A-Za-z0-9가-힣])",
        re.IGNORECASE,
    )
    for m in pattern.finditer(text):
        key = m.group(1).lower()
        rule = _ALIAS_EXACT_MAP.get(key)
        if not rule:
            continue
        src = rule["source"]
        # 매칭 범위는 별칭 부분만 (조사 제외)
        alias_end = m.start() + len(m.group(1))
        hits.append(Hit(
            span=m.group(1), label=rule["label"],
            replacement=rule.get("replacement"),
            confidence=0.94,
            source=Source(
                doc=src.get("doc", ""), page=src.get("page"),
                quote=src.get("quote", ""),
            ),
            start=m.start(), end=alias_end,
        ))
    return hits


# =========================
# Semantic Matching (n-gram 후보 생성으로 개선)
# =========================
_TOKEN_RE = re.compile(r"[A-Za-z가-힣0-9][A-Za-z가-힣0-9.\-_/()]*")


def _get_semantic_candidates(text: str, min_len=2, max_len=40):
    """1-gram, 2-gram, 3-gram 후보를 생성하여 복합 표현도 매칭."""
    tokens = list(_TOKEN_RE.finditer(text))
    candidates = []
    seen = set()

    for i, match in enumerate(tokens):
        token = match.group(0).strip()

        # 1-gram
        if min_len <= len(token) <= max_len and _should_consider_token(token):
            key = (match.start(), match.end())
            if key not in seen:
                candidates.append((token, match.start(), match.end()))
                seen.add(key)

        # 2-gram: 인접 토큰이 가까우면 결합
        if i + 1 < len(tokens):
            next_m = tokens[i + 1]
            gap = next_m.start() - match.end()
            if gap <= 3:  # 공백/조사 하나 정도
                combined = text[match.start():next_m.end()]
                if min_len <= len(combined) <= max_len:
                    key = (match.start(), next_m.end())
                    if key not in seen:
                        candidates.append((combined, match.start(), next_m.end()))
                        seen.add(key)

        # 3-gram: 3개 토큰 결합 (Final Cut Pro, Chat GPT 등)
        if i + 2 < len(tokens):
            next_m = tokens[i + 1]
            next_next_m = tokens[i + 2]
            gap1 = next_m.start() - match.end()
            gap2 = next_next_m.start() - next_m.end()
            if gap1 <= 3 and gap2 <= 3:
                combined = text[match.start():next_next_m.end()]
                if min_len <= len(combined) <= max_len * 2:
                    key = (match.start(), next_next_m.end())
                    if key not in seen:
                        candidates.append((combined, match.start(), next_next_m.end()))
                        seen.add(key)

    return candidates


def semantic_match(text: str, threshold: float = 0.78, max_hits: int = 20) -> List[Hit]:
    """임베딩 기반 시맨틱 매칭. threshold 0.80→0.78로 완화."""
    if not (_embedder and _ALIAS_EMB_INDEX is not None):
        return []
    candidates = _get_semantic_candidates(text)
    if not candidates:
        return []
    cand_tokens = [c[0] for c in candidates]
    cand_embeddings = _encode_texts(cand_tokens, is_query=True)
    sim_matrix = np.matmul(
        np.array(cand_embeddings, dtype=np.float32), _ALIAS_EMB_INDEX.T
    )
    hits: List[Hit] = []
    used_spans = set()
    for i, row in enumerate(sim_matrix):
        if len(hits) >= max_hits:
            break
        best_idx = int(np.argmax(row))
        score = float(row[best_idx])
        second = float(np.partition(row, -2)[-2]) if row.size > 1 else 0.0
        # 임계값 완화: 0.80→0.78, gap 0.04→0.03
        if not (score >= threshold and (score - second) >= 0.03):
            continue
        span_text, start, end = candidates[i]
        if (start, end) in used_spans:
            continue
        # 이미 잡힌 범위와 겹치면 스킵
        if any(s < end and start < e for s, e in used_spans):
            continue
        matched_rule = _ALIAS_RULES[best_idx]
        # 단계별 신뢰도
        if score >= 0.95:
            conf = 0.94
        elif score >= 0.90:
            conf = 0.91
        else:
            conf = max(0.60, score * 0.90)
        hits.append(Hit(
            span=span_text, label=matched_rule["label"],
            replacement=matched_rule.get("replacement"),
            confidence=float(conf),
            source=Source(**matched_rule["source"]),
            start=start, end=end,
        ))
        used_spans.add((start, end))
    return hits


# =========================
# Unknown Abbreviation Detection (개선: 혼합 대소문자도 감지)
# =========================
def detect_unknown_abbreviations(text: str, existing_hits: List[Hit]) -> List[Hit]:
    hits: List[Hit] = []
    covered = set()
    for h in existing_hits:
        for i in range(h.start, h.end):
            covered.add(i)

    # 기존: ALL-CAPS 2~10자
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
            source=Source(
                doc="자동 감지", page=None,
                quote="영문 약어가 감지됨. 한글 표기 필요 여부 검토 필요.",
            ),
            start=start, end=end,
        ))

    # 신규: CamelCase 패턴 (예: ChatGpt, PyTorch 등 - 이미 규칙에 없는 것만)
    for match in re.finditer(r'(?<![A-Za-z])([A-Z][a-z]+(?:[A-Z][a-z]+)+)(?![A-Za-z])', text):
        term = match.group(1)
        start, end = match.start(), match.end()
        if any(i in covered for i in range(start, end)):
            continue
        hits.append(Hit(
            span=term,
            label="미확인 영문 용어",
            replacement=None,
            confidence=0.80,
            source=Source(
                doc="자동 감지", page=None,
                quote="영문 복합어가 감지됨. 한글 표기 필요 여부 검토 필요.",
            ),
            start=start, end=end,
        ))

    return hits


# =========================
# Collapse Parenthetical Duplicates
# =========================
def collapse_parenthetical_duplicates(text: str, hits: List[Hit]) -> List[Hit]:
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
        k = hi.end
        while k < len(text) and text[k].isspace():
            k += 1
        if k < len(text) and text[k] == "(":
            j = i + 1
            while j < n and hits[j].start < k + 1:
                j += 1
            if j < n:
                hj = hits[j]
                inner_start = k + 1
                while inner_start < len(text) and text[inner_start].isspace():
                    inner_start += 1
                inner_end = hj.end
                tmp = inner_end
                while tmp < len(text) and text[tmp].isspace():
                    tmp += 1
                if tmp < len(text) and text[tmp] == ")":
                    close_pos = tmp
                    if (
                        (hj.start >= inner_start)
                        and (hj.end <= close_pos)
                        and (hi.label == hj.label)
                        and (hi.replacement == hj.replacement)
                        and (hi.replacement is not None)
                    ):
                        combined = Hit(
                            span=text[hi.start : close_pos + 1],
                            label=hi.label,
                            replacement=hi.replacement,
                            confidence=max(hi.confidence, hj.confidence),
                            source=hi.source,
                            start=hi.start,
                            end=close_pos + 1,
                        )
                        result.append(combined)
                        used[i] = True
                        used[j] = True
                        i += 1
                        continue
        result.append(hi)
        used[i] = True
        i += 1
    for idx in range(n):
        if not used[idx]:
            result.append(hits[idx])
    seen = set()
    dedup: List[Hit] = []
    for h in sorted(result, key=lambda h: (h.start, h.end)):
        key = (h.start, h.end, h.label, h.replacement)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(h)
    return dedup


# =========================
# Merge Hits (신뢰도+길이 기반 우선순위로 전면 개선)
# =========================
def merge_hits(*hit_groups: List[Hit]) -> List[Hit]:
    """겹치는 히트 중 가장 좋은 것을 선택.

    기존: 왼쪽부터 탐욕적으로 선택 → 짧은 오탐이 긴 정탐을 삭제하는 문제
    개선: 신뢰도 높고 긴 히트를 우선 선택, 겹치는 것은 제거
    """
    all_hits: List[Hit] = []
    for g in hit_groups:
        all_hits.extend(g)
    if not all_hits:
        return []

    # 중복 제거 (같은 위치, 같은 라벨)
    seen = set()
    deduped: List[Hit] = []
    for h in all_hits:
        key = (h.start, h.end, h.label)
        if key not in seen:
            seen.add(key)
            deduped.append(h)

    # 신뢰도 내림차순, 길이 내림차순으로 정렬 → 좋은 히트부터 선택
    deduped.sort(key=lambda h: (-h.confidence, -(h.end - h.start)))

    # 탐욕적 선택: 겹치지 않는 히트만 추가
    merged: List[Hit] = []
    for hit in deduped:
        if not any(h.start < hit.end and hit.start < h.end for h in merged):
            merged.append(hit)

    # 위치순 정렬
    merged.sort(key=lambda h: h.start)
    return merged
