import os
import time
import re
from typing import List, Optional, Tuple
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


class AnalyzeRequest(BaseModel):
    text: str
    policy_version: str = "2024-03"


class AnalyzeResponse(BaseModel):
    hits: List[Hit]
    latency_ms: int


# =========================
# FastAPI App Setup
# =========================
app = FastAPI(title="Seongnam LifeRec Checker", version="1.6.0")
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
    {"pattern": r"(?:KakaoTalk|카카오톡|카톡|LINE|라인|Instagram|인스타그램|Twitter|트위터|Meta|메타|Facebook|페이스북)", "label": "상호명", "replacement": "소셜 네트워크 서비스", "confidence": 0.95, "source": {"doc": "대체표현", "page": 1, "quote": "KakaoTalk, Instagram, Facebook 등 → 메신저, 소셜네트워크서비스"}, "aliases": ["kakaotalk", "kkt", "카톡방", "인스타", "insta", "페북", "x(트위터)"]},
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
    {"pattern": r"(?:Zoom|줌|웨일온|Whale\s?ON)", "label": "상호명", "replacement": "화상 회의", "confidence": 0.92, "source": {"doc": "대체표현", "page": 1, "quote": "Zoom(줌) 등 → 화상 회의"}, "aliases": ["줌미팅", "웨일온회의"]},
    # --- 기관명 / 논문 / 외국어 / 특수문자 ---
    {"pattern": r"(?:UN|EU|WHO|WTO|OECD|IMF|UNESCO|IAEA|NATO|유엔|유럽연합|세계\s?보건\s?기구|세계\s?무역\s?기구|경제협력개발기구|국제통화기금|유네스코|국제원자력기구|북대서양조약기구)", "label": "기관명", "replacement": "국제기구", "confidence": 0.98, "source": {"doc": "단체명 기재", "page": 1, "quote": "교육관련기관 제외 특정 기관명 기재 불가"}, "aliases": ["유엔기구", "오이시디", "나토", "유네스코한국위원회"]},
    {"pattern": r"소논문|연구보고서", "label": "논문 실적", "replacement": "탐구 활동", "confidence": 0.99, "source": {"doc": "논문 기재", "page": 1, "quote": "자율탐구활동 산출물 실적 기재 불가"}, "aliases": ["소논문 작성", "연구보고서를 제출"]},
    {"pattern": r"[一-龥]", "label": "외국어", "replacement": None, "confidence": 0.99, "source": {"doc": "외국어 기재", "page": 1, "quote": "한글 사용 원칙. 영문 제외 외국어 입력 불가."}, "aliases": []},
    {"pattern": r"[·※▷▶]", "label": "특수문자", "replacement": " ", "confidence": 0.99, "source": {"doc": "특수문자", "page": 1, "quote": "서술형 특수문자 입력 지양"}, "aliases": []},
    # --- 학술 용어(일반화) ---
    {"pattern": r"(?:CRISPR-?Cas9|크리스퍼-?카스9?)", "label": "전문 용어", "replacement": "유전자 가위 기술", "confidence": 0.93, "source": {"doc": "학술 용어 일반화", "page": 1, "quote": "과도한 전문용어는 일반화/설명적 표현 사용 권장"}, "aliases": ["crispr", "cas9", "크리스퍼"]},
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
                start=match.start(), end=match.end()
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
HTML_PAGE = """
<!doctype html><html lang="ko"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>생활기록부 자동 점검 – 데모 v1.6</title>
<style>:root{--bg:#0b1020;--card:#111830;--ink:#e6edff;--muted:#9db1ff;--accent:#4f7cff;--hit:#ff4455;--ok:#25d366}*{box-sizing:border-box}body{margin:0;font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Apple SD Gothic Neo,Noto Sans KR,sans-serif;background:var(--bg);color:var(--ink)}.wrap{max-width:1100px;margin:36px auto;padding:0 16px}.card{background:var(--card);border-radius:20px;padding:20px;box-shadow:0 10px 30px rgba(0,0,0,.35)}h1{margin:0 0 8px}.muted{color:var(--muted);font-size:12px}textarea{width:100%;min-height:160px;padding:14px;border-radius:14px;border:1px solid #263257;background:#0e1430;color:var(--ink);font-size:16px;resize:vertical}button{background:var(--accent);color:white;border:0;padding:12px 16px;border-radius:12px;font-weight:700;cursor:pointer}button:disabled{opacity:.6;cursor:not-allowed}.row{display:flex;gap:12px;flex-wrap:wrap;align-items:center}.grid{margin-top:16px;display:grid;grid-template-columns:1fr 1fr 320px;gap:16px}@media (max-width: 900px) {.grid{grid-template-columns: 1fr;}}.panel{background:#0e1430;border:1px solid #263257;border-radius:14px;padding:14px}mark{background:transparent;color:var(--hit);font-weight:800;text-decoration:underline;text-underline-offset:3px}ins.rep{background:#0f2a1f;color:#b2ffd8;text-decoration:none;border-bottom:2px solid var(--ok);padding:0 2px}.hit{display:flex;justify-content:space-between;gap:8px;border-bottom:1px dashed #263257;padding:8px 0}.pill{font-size:12px;padding:3px 8px;border-radius:999px;background:#1b2342;color:#c7d3ff}</style></head><body>
<div class="wrap">
<h1>🧭 성남 생활기록부 자동 점검 (데모 v1.6)</h1>
<div class="card">
<div class="muted">본문을 붙여넣고 "검사"를 누르세요 · 오른쪽에 <b>수정본 미리보기</b>와 <b>모두 적용</b>이 있어요</div>
<textarea id="txt"></textarea>
<div class="row" style="margin-top:8px">
<button id="btn">검사</button>
<button id="btnApplyAll" disabled>모두 적용</button>
<button id="btnSample">샘플 텍스트</button>
<span id="lat" class="muted">–</span>
<span id="chg" class="muted">변경 0건</span>
</div>
<div class="grid">
<div class="panel"><div class="muted" style="margin-bottom:8px">하이라이트 결과(원문)</div><div id="view" style="line-height:1.8; white-space:pre-wrap;"></div></div>
<div class="panel"><div class="muted" style="margin-bottom:8px">수정본 미리보기(대체어 적용)</div><div id="preview" style="line-height:1.8; white-space:pre-wrap;"></div></div>
<div class="panel"><div class="muted" style="margin-bottom:8px">근거 / 대체표현</div><div id="hits"></div></div>
</div>
</div>
</div>
<script>
const POLICY="2024-03";
const MIN_PREVIEW_CONF = 0.90; // 자동 치환 기준
let currentHits = [];
const txtEl = document.getElementById("txt");
function esc(s){return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");}
async function analyze(text){const r=await fetch("/analyze",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({text:text,policy_version:POLICY})});if(!r.ok)throw new Error(`API Error: ${r.statusText}`);return await r.json();}
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
function renderResults(text, hits) {
const sortedHits = [...hits].sort((a, b) => a.start - b.start);
let lastIndex = 0;
const viewParts = [];
const previewParts = [];
for (const hit of sortedHits) {
if (hit.start < lastIndex) continue; // overlapped (safety)
if (hit.start > lastIndex) {
const segment = esc(text.slice(lastIndex, hit.start));
viewParts.push(segment);
previewParts.push(segment);
}
// always show original highlight in "원문"
viewParts.push(`<mark title="${esc(hit.label)}">${esc(hit.span)}</mark>`);
const canReplace = !!hit.replacement && (hit.confidence >= MIN_PREVIEW_CONF);
if (canReplace) {
const look = text.slice(hit.end, hit.end+2);
const m = look.match(/^(으로|로|을|를|은|는|이|가|과|와)/);
if(m){
const appended = chooseParticle(hit.replacement, m[0]);
previewParts.push(`<ins class="rep" title="${esc(hit.label)}">${esc(hit.replacement + appended)}</ins>`);
lastIndex = hit.end + m[0].length;
}else{
previewParts.push(`<ins class="rep" title="${esc(hit.label)}">${esc(hit.replacement)}</ins>`);
lastIndex = hit.end;
}
} else {
previewParts.push(`<mark title="${esc(hit.label)}">${esc(hit.span)}</mark>`);
lastIndex = hit.end;
}
}
if (lastIndex < text.length) {
const segment = esc(text.slice(lastIndex));
viewParts.push(segment);
previewParts.push(segment);
}
document.getElementById("view").innerHTML = viewParts.join("").replace(/\\n/g, "<br>");
document.getElementById("preview").innerHTML = previewParts.join("").replace(/\\n/g, "<br>");
const hitsEl = document.getElementById("hits");
hitsEl.innerHTML = "";
if (!hits.length) {
hitsEl.innerHTML = '<div class="muted">규정 위반 항목을 찾지 못했습니다. 👌</div>';
return;
}
for (const h of hits) {
const row = document.createElement("div");
row.className = "hit";
const auto = (h.replacement && h.confidence >= MIN_PREVIEW_CONF);
const conf = Math.round(h.confidence * 100);
row.innerHTML =
`<div>
<b>${esc(h.span)}</b> <span class="pill">${h.label}</span> <span class="pill">${conf}%</span><br/>
<span class="muted">${h.source.doc} p.${h.source.page||'?'}: ${esc(h.source.quote||'')}</span>
</div>` +
(h.replacement ? `<div class="pill">${auto ? '자동적용' : '검토필요'}: ${esc(h.replacement)}</div>` : "");
hitsEl.appendChild(row);
}
}
function applyAllReplacements() {
let text = txtEl.value;
const replacableHits = currentHits
.filter(h => h.replacement && h.confidence >= MIN_PREVIEW_CONF)
.sort((a,b) => b.start - a.start);
for (const hit of replacableHits) {
const sp = spliceWithParticle(text, hit.start, hit.end, hit.replacement);
text = sp.newText;
}
txtEl.value = text;
document.getElementById("btn").click();
}
document.getElementById("btnSample").onclick = function() {
txtEl.value = '유엔(UN) 보고서를 참조하여 챗GPT 초안 작성 후 MS워드 정리하고 Google Docs에 옮겼다.\\nZoom(웨일온)으로 발표하고 yutube·Instagram에 홍보했다.\\n이동은 KTX, 표지는 Canva 제작, 편집은 키네마스터 마무리했으며 소논문도 제출했다. 또한 Jupyter 통해 실험을 정리했고 CRISPR-Cas9 관련 내용을 참고했다. Java Script 및 JS도 사용했다.';
};
document.getElementById("btn").onclick = async function() {
const text = txtEl.value || "";
this.textContent = "검사 중...";
this.disabled = true;
try {
const res = await analyze(text);
currentHits = res.hits;
document.getElementById("lat").textContent = `처리시간: ${res.latency_ms} ms`;
renderResults(text, res.hits);
const changedCount = currentHits.filter(h => h.replacement && h.confidence >= MIN_PREVIEW_CONF).length;
document.getElementById("chg").textContent = `변경 ${changedCount}건`;
const btnApply = document.getElementById("btnApplyAll");
btnApply.disabled = changedCount === 0;
btnApply.onclick = applyAllReplacements;
} catch (e) {
alert("오류가 발생했습니다. 잠시 후 다시 시도해주세요.\\n(Hugging Face 컨테이너 기상 중일 수 있음)");
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
    # 5) Merge & respond
    final_hits = merge_hits(primary_hits, hits_semantic)
    latency_ms = int((time.perf_counter() - t0) * 1000)
    return AnalyzeResponse(hits=final_hits, latency_ms=latency_ms)


