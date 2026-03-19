import time

from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from schemas import (
    AnalyzeRequest, AnalyzeResponse,
    ByteCountRequest, ByteCountResponse,
)
from rules import RULES
from engine import (
    regex_match, alias_exact_match, semantic_match,
    merge_hits, collapse_parenthetical_duplicates,
    detect_unknown_abbreviations, init_engine,
)
from byte_counter import analyze_bytes, normalize_for_neis, utf8_byte_len

# =========================
# FastAPI App Setup
# =========================
app = FastAPI(title="LifeRec Checker", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)

# =========================
# Initialize Engine (load model + build index)
# =========================
init_engine(RULES)

# =========================
# UI (HTML/CSS/JS) - 기존 프론트엔드 유지
# =========================
HTML_PAGE = r"""
<!doctype html><html lang="ko"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>생기부 금칙어 검사기 – v3.0.0</title>
<style>:root{--bg:#0b1020;--card:#111830;--ink:#e6edff;--muted:#9db1ff;--accent:#4f7cff;--hit:#ff4455;--ok:#25d366;--warn:#ffaa00}*{box-sizing:border-box}body{margin:0;font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Apple SD Gothic Neo,Noto Sans KR,sans-serif;background:var(--bg);color:var(--ink)}.wrap{max-width:1100px;margin:36px auto;padding:0 16px}.card{background:var(--card);border-radius:20px;padding:20px;box-shadow:0 10px 30px rgba(0,0,0,.35)}h1{margin:0 0 8px}.muted{color:var(--muted);font-size:12px}textarea{width:100%;min-height:160px;padding:14px;border-radius:14px;border:1px solid #263257;background:#0e1430;color:var(--ink);font-size:16px;resize:vertical}button{background:var(--accent);color:white;border:0;padding:12px 16px;border-radius:12px;font-weight:700;cursor:pointer}button:disabled{opacity:.6;cursor:not-allowed}.row{display:flex;gap:12px;flex-wrap:wrap;align-items:center}.grid{margin-top:16px;display:grid;grid-template-columns:1fr 1fr 320px;gap:16px}@media (max-width: 900px) {.grid{grid-template-columns: 1fr;}}.panel{background:#0e1430;border:1px solid #263257;border-radius:14px;padding:14px}mark{background:transparent;color:var(--hit);font-weight:800;text-decoration:underline;text-underline-offset:3px}ins.rep{background:#0f2a1f;color:#b2ffd8;text-decoration:none;border-bottom:2px solid var(--ok);padding:0 2px}.hit{display:flex;justify-content:space-between;gap:8px;border-bottom:1px dashed #263257;padding:8px 0}.pill{font-size:12px;padding:3px 8px;border-radius:999px;background:#1b2342;color:#c7d3ff}.byte-box{background:linear-gradient(135deg,#1a2744 0%,#0e1430 100%);border:1px solid #263257;border-radius:14px;padding:16px;margin-top:12px}.byte-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px}.byte-item{text-align:center;padding:12px;background:#0b1020;border-radius:10px}.byte-value{font-size:28px;font-weight:800;color:var(--accent)}.byte-label{font-size:11px;color:var(--muted);margin-top:4px}.byte-warn{color:var(--warn)}.suspicious-list{margin-top:12px;font-size:12px;color:var(--warn)}.suspicious-item{padding:4px 0;border-bottom:1px dashed #263257}

.panel-head{display:flex;align-items:center;gap:8px;margin-bottom:8px}
.btn-mini{padding:6px 10px;border-radius:10px;font-size:12px;font-weight:700}
ins.rep.deleted{background:#1a2a1f;color:#88ccaa;border-bottom:2px solid #ffaa00}
ins.rep.deleted s{text-decoration:line-through;text-decoration-color:#ff6666}

</style></head><body>
<div class="wrap">
<h1>생기부 금칙어 검사기 <span style="font-size:14px;color:var(--accent)">(v3.0.0)</span></h1>
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
let deletedText = hit.span;  // 삭제된 원본 텍스트
if(m){
deletedText += m[0];  // 조사도 포함
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
// v2.1.2: 삭제된 부분을 녹색 취소선으로 표시
previewParts.push(`<ins class="rep deleted" title="삭제됨: ${esc(deletedText)}"><s>${esc(deletedText)}</s></ins>`);
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
// v2.0.8: 미리보기에서 중복 대체어 제거 (HTML 보존)
let previewHtml = previewParts.join("").replace(/\n/g, "<br>");
// v2.0.8: HTML 내에서 연속된 동일 대체어 병합 (녹색 밑줄 유지)
// 예: <ins>프로그래밍 언어</ins> 및 <ins>프로그래밍 언어</ins> -> <ins>프로그래밍 언어</ins>
previewHtml = previewHtml.replace(/(<ins[^>]*>)([^<]+)<\/ins>\s*및\s*\1\2<\/ins>/g, '$1$2</ins>');
previewHtml = previewHtml.replace(/(<ins[^>]*>)([^<]+)<\/ins>\s*및\s*<ins[^>]*>\2<\/ins>/g, '$1$2</ins>');
// "프로그래밍 언어" 및 "프로그래밍 언어도" 패턴
previewHtml = previewHtml.replace(/(<ins[^>]*>)프로그래밍 언어<\/ins>\s*및\s*<ins[^>]*>프로그래밍 언어<\/ins>(도|를|을|은|는)?/g, '$1프로그래밍 언어$2</ins>');
const previewEl = document.getElementById("preview");
previewEl.innerHTML = previewHtml;
// v2.0.8: 텍스트 노드에서 어색한 문장 정리 (HTML 태그는 보존)
const walker = document.createTreeWalker(previewEl, NodeFilter.SHOW_TEXT, null, false);
while(walker.nextNode()) {
let nodeText = walker.currentNode.textContent;
nodeText = removeDuplicateReplacements(nodeText);
nodeText = cleanupAfterDeletion(nodeText);
nodeText = fixAwkwardParticles(nodeText);
walker.currentNode.textContent = nodeText;
}
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
// ============================================================
// [최우선 법칙 - 절대 수정 금지] 가운뎃점(·) → 콤마+공백(, ) 변환
// ============================================================
// ⚠️ 경고: 이 함수는 절대로 수정하거나 삭제하지 마세요!
// ⚠️ WARNING: DO NOT MODIFY OR DELETE THIS FUNCTION!
// 이 변환은 모든 텍스트 처리에서 최우선으로 적용되어야 합니다.
// ============================================================
function convertMiddleDot(text) {
return text.replace(/·/g, ', ');
}

// v2.1.3: 텍스트 전처리 - 특수문자/기호 정리 (대폭 확장)
function preprocessText(text) {
// ★★★ 최우선 법칙: 가운뎃점 → 콤마 변환 (절대 수정 금지!) ★★★
text = convertMiddleDot(text);

// 1-2. 특수 따옴표 → 일반 따옴표 변환
text = text.replace(/['']/g, "'");  // 둥근 작은따옴표 → 일반 작은따옴표
text = text.replace(/[""]/g, '"');  // 둥근 큰따옴표 → 일반 큰따옴표

// 2. 마크다운 기호 제거: **, ##, *, # 등
text = text.replace(/\*\*+/g, '');
text = text.replace(/###+/g, '');
text = text.replace(/(?<![a-zA-Z0-9])#+(?![a-zA-Z0-9])/g, '');
text = text.replace(/(?<![a-zA-Z0-9])\*+(?![a-zA-Z0-9])/g, '');

// 3. 모든 불필요한 특수기호 제거 (문장부호 . , ? ! 및 괄호 () 제외)
text = text.replace(/[「」『』【】〈〉《》〔〕［］｛｝]/g, '');
text = text.replace(/[★☆●○◆◇■□▲△▶▷◀◁◈◉◎]/g, '');
text = text.replace(/[♠♣♥♦♤♧♡♢]/g, '');
text = text.replace(/[※†‡⁂]/g, '');
text = text.replace(/[→←↑↓↔↕⇒⇐⇑⇓⇔]/g, '');
text = text.replace(/[♪♬♩♭♯]/g, '');
text = text.replace(/[☑☐☒✓✔✗✘✕✖]/g, '');
text = text.replace(/[─━│┃┄┅┆┇┈┉┊┋]/g, '');
text = text.replace(/[╭╮╯╰┌┐└┘├┤┬┴┼]/g, '');
text = text.replace(/[°℃℉‰‱]/g, '');
text = text.replace(/[©®™℗]/g, '');
text = text.replace(/[☀☁☂☃☄★☆☇☈]/g, '');
text = text.replace(/[♀♂⚢⚣⚤⚥⚦⚧⚨]/g, '');
text = text.replace(/[⚠⚡⚪⚫⚬⚭⚮⚯]/g, '');
text = text.replace(/[❤❥❦❧❨❩❪❫❬❭❮❯❰❱]/g, '');
text = text.replace(/[❲❳❴❵❶❷❸❹❺❻❼❽❾❿]/g, '');
text = text.replace(/[➀➁➂➃➄➅➆➇➈➉]/g, '');
text = text.replace(/[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]/g, '');
text = text.replace(/[ⓐⓑⓒⓓⓔⓕⓖⓗⓘⓙⓚⓛⓜⓝⓞⓟⓠⓡⓢⓣⓤⓥⓦⓧⓨⓩ]/g, '');
text = text.replace(/[ⒶⒷⒸⒹⒺⒻⒼⒽⒾⒿⓀⓁⓂⓃⓄⓅⓆⓇⓈⓉⓊⓋⓌⓍⓎⓏ]/g, '');
text = text.replace(/[㉠㉡㉢㉣㉤㉥㉦㉧㉨㉩㉪㉫㉬㉭]/g, '');
text = text.replace(/[㈀㈁㈂㈃㈄㈅㈆㈇㈈㈉㈊㈋㈌㈍㈎㈏]/g, '');
text = text.replace(/[㊀㊁㊂㊃㊄㊅㊆㊇㊈㊉]/g, '');
text = text.replace(/[▁▂▃▄▅▆▇█▉▊▋▌▍▎▏]/g, '');
text = text.replace(/[░▒▓]/g, '');
text = text.replace(/[╱╲╳]/g, '');
text = text.replace(/[〃〄々〆〇]/g, '');
text = text.replace(/[＊＃＋－＝＜＞｜～￣＿]/g, '');
text = text.replace(/[`~^\\|@$%&]/g, '');

// 4. 문장 종결 콤마 → 마침표 변환 (v2.1.4)
const sentenceEndings = [
'함', '됨', '있음', '없음', '보임', '드러남', '갖춤', '돋보임',
'기대됨', '가능함', '필요함', '제시함', '강조함', '확보함', '마련함',
'구체화함', '체계화함', '높임', '강화함', '줄임', '늘림', '뒷받침함',
'해소됨', '감소됨', '증가함', '정착됨', '나타남', '이루어짐', '진행됨',
'완료됨', '수행함', '발휘함', '보여줌', '이끌어냄', '키움', '성장함'
];
const endingPattern = new RegExp('(' + sentenceEndings.join('|') + '),(?=\\s)', 'g');
text = text.replace(endingPattern, '$1.');

// 5. 연속 공백 정리
text = text.replace(/\s{2,}/g, ' ');
// 6. 콤마 뒤 공백 정리
text = text.replace(/,\s*,/g, ',');
text = text.replace(/,\s+/g, ', ');
return text.trim();
}
document.getElementById("btn").onclick = async function() {
let text = txtEl.value || "";
// v2.1.0: 텍스트 전처리 적용
text = preprocessText(text);
txtEl.value = text; // 전처리된 텍스트로 textarea 업데이트
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
@app.get("/health", summary="Health check for CloudType")
def health():
    from engine import _model_ready
    return {"status": "ok", "model_ready": _model_ready}


@app.get("/", response_class=HTMLResponse, summary="Main UI Page")
def home():
    return HTML_PAGE


@app.post("/analyze", response_model=AnalyzeResponse, summary="Analyze student record text")
def analyze_endpoint(payload: AnalyzeRequest = Body(...)):
    t0 = time.perf_counter()
    # 1) Regex Analysis Unit
    hits_rule = regex_match(payload.text)
    # 2) Exact Alias Unit (safe auto-fix for common typos)
    hits_alias = alias_exact_match(payload.text)
    # 3) Collapse parenthetical duplicates like '유엔(UN)'
    primary_hits = collapse_parenthetical_duplicates(payload.text, hits_rule + hits_alias)
    # 4) Embedding Analysis Unit (improved with bge-m3)
    hits_semantic = semantic_match(payload.text)
    # 5) Merge known hits
    known_hits = merge_hits(primary_hits, hits_semantic)
    # 6) Detect unknown abbreviations
    hits_unknown = detect_unknown_abbreviations(payload.text, known_hits)
    # 7) Final merge & respond
    final_hits = merge_hits(known_hits, hits_unknown)
    latency_ms = int((time.perf_counter() - t0) * 1000)
    return AnalyzeResponse(hits=final_hits, latency_ms=latency_ms)


@app.post("/byte-count", response_model=ByteCountResponse, summary="Count bytes for student record (v2.0)")
def byte_count(payload: ByteCountRequest = Body(...)):
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
