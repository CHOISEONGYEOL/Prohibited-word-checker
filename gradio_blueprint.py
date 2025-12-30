# gradio_blueprint.py
# =========================================
# 시험 블루프린트 - 4단계 UI (취소선 체크 기능)
# =========================================

import math
from pathlib import Path
import gradio as gr

# ---------- 테마 ----------
def _get_theme():
    try:
        return gr.themes.Soft(primary_hue="violet")
    except Exception:
        return None
theme = _get_theme()

# ---------- 유틸 ----------
def _round1(x: float) -> float:
    return float(f"{x:.1f}")

def _split_counts(total: int, ratios):
    if total <= 0: return [0]*len(ratios)
    s = sum(ratios)
    ratios = [r/s if s>0 else 1/len(ratios) for r in ratios]
    raw = [r*total for r in ratios]
    floors = [int(math.floor(x)) for x in raw]
    rem = total - sum(floors)
    frac = sorted([(raw[i]-floors[i], i) for i in range(len(ratios))], reverse=True)
    counts = floors[:]
    for k in range(rem): counts[frac[k][1]] += 1
    return counts

def _offsets_zero_sum(n: int):
    if n<=0: return []
    if n==1: return [0.0]
    if n==2: return [-0.1,+0.1]
    a = n//3; rem = n%3
    neg, zero, pos = a, a, a
    if rem==1: zero += 1
    elif rem==2: neg += 1; pos += 1
    return ([-0.1]*neg)+([0.0]*zero)+([+0.1]*pos)

def _correct_to_total_0p1(values, target_sum):
    if not values:
        return []
    rounded = [_round1(v) for v in values]
    cur = _round1(sum(rounded))
    diff = _round1(target_sum - cur)
    if abs(diff) < 0.1: return rounded
    steps = int(round(abs(diff)*10))
    sign = +0.1 if diff>0 else -0.1
    i = 0
    n = len(rounded)
    if n == 0:
        return rounded
    for _ in range(steps):
        if rounded[i] + sign >= 0.1:
            rounded[i] = _round1(rounded[i] + sign)
        i = (i+1) % n
    final = _round1(sum(rounded))
    if final != _round1(target_sum) and len(rounded)>0:
        delta = _round1(target_sum - final)
        step = 0.1 if delta>0 else -0.1
        if abs(delta) >= 0.1 and rounded[-1] + step >= 0.1:
            rounded[-1] = _round1(rounded[-1] + step)
    return rounded

# ---------- 프리셋(요청 비율) ----------
RATIOS_3 = {"상":0.20,"중":0.50,"하":0.30}
RATIOS_5 = {"상":0.10,"중상":0.20,"중":0.40,"중하":0.20,"하":0.10}

P_3 = {"하":0.90,"중":0.65,"상":0.40}
P_5 = {"하":0.90,"중하":0.75,"중":0.65,"중상":0.55,"상":0.40}

CURVE_3 = {"하":0.92,"중":1.00,"상":1.10}
CURVE_5 = {"하":0.92,"중하":0.97,"중":1.00,"중상":1.05,"상":1.10}

BIAS_3 = {"하":-0.05,"중":0.00,"상":+0.05}
BIAS_5 = {"하":-0.05,"중하":-0.02,"중":0.00,"중상":+0.02,"상":+0.05}

def _expected_stats(codes, points, p_map):
    if not codes or not points: return 0.0,0.0
    mean=0.0; var=0.0
    for c,w in zip(codes,points):
        p=p_map.get(c,0.65); mean+=w*p; var+=(w*w)*p*(1-p)
    return _round1(mean), _round1(var**0.5)

def _break_cross_difficulty_ties(codes, points, rank):
    for _ in range(12):
        changed=False
        idxs=sorted(range(len(points)), key=lambda i:(rank.get(codes[i],999), points[i]))
        seen={}
        for i in idxs:
            sc=points[i]; r=rank.get(codes[i],999)
            if sc in seen:
                easier=[j for j in seen[sc]["idxs"] if rank.get(codes[j],999)<r]
                if easier:
                    donor=None
                    for j in easier:
                        if points[j]-0.1>=0.1: donor=j; break
                    if donor is not None:
                        points[i]=_round1(points[i]+0.1)
                        points[donor]=_round1(points[donor]-0.1)
                        changed=True; break
            if sc not in seen: seen[sc]={"min_rank":r,"idxs":[i]}
            else:
                seen[sc]["min_rank"]=min(seen[sc]["min_rank"], r)
                seen[sc]["idxs"].append(i)
        if not changed: break
    return points

def build_section(n_q, section_pts, scheme, ratios, curve, bias, p_map, label):
    if n_q<=0 or section_pts<=0:
        return {"codes":[], "rows":[], "counts":{}, "anchors":{},
                "section_points":_round1(section_pts), "exp":(0.0,0.0)}
    easy_order = ["하","중","상"] if scheme=="3단계" else ["하","중하","중","중상","상"]
    rank={lvl:i for i,lvl in enumerate(easy_order)}
    keys=list(ratios.keys())
    counts=_split_counts(n_q,[ratios[k] for k in keys])
    count_map={keys[i]:counts[i] for i in range(len(keys))}
    codes=[]
    for k in keys: codes.extend([k]*count_map.get(k,0))
    avg=section_pts/n_q
    raw=[avg*curve.get(c,1.0) for c in codes]
    s=sum(raw) if raw else 1.0
    scale=section_pts/s
    raw=[x*scale for x in raw]
    raw=[max(0.1, x+bias.get(c,0.0)) for x,c in zip(raw,codes)]
    by_label={}
    for i,c in enumerate(codes): by_label.setdefault(c,[]).append(i)
    for _,idxs in by_label.items():
        offs=_offsets_zero_sum(len(idxs))
        for i,off in zip(idxs,offs): raw[i]=max(0.1, raw[i]+off)
    points=_correct_to_total_0p1(raw, section_pts)
    points=_break_cross_difficulty_ties(codes, points, rank)
    items=[{"난이도":c,"배점":_round1(p)} for c,p in zip(codes,points)]
    items.sort(key=lambda r:(rank.get(r["난이도"],999), r["배점"]))
    rows=[{"번호":i+1,"유형":label,"난이도":it["난이도"],"배점":it["배점"]} for i,it in enumerate(items)]
    anchors={}
    for lvl in easy_order:
        if lvl in count_map:
            anchors[lvl]=_round1(max(0.1, avg*curve.get(lvl,1.0)*scale + bias.get(lvl,0.0)))
    exp_mean,exp_sd=_expected_stats([it["난이도"] for it in items],[it["배점"] for it in items], p_map)
    ordered_counts={lvl:count_map.get(lvl,0) for lvl in easy_order if lvl in count_map}
    return {"codes":[it["난이도"] for it in items],"rows":rows,"counts":ordered_counts,
            "anchors":anchors,"section_points":_round1(section_pts),"exp":(exp_mean,exp_sd)}

# ---------- 표 렌더링 (HTML, JS로 취소선/합계) ----------
def render_section_html(name, plan, easy_order, section_id: str):
    def counts_line(cm):
        return " / ".join([f"{k}:{cm.get(k,0)}" for k in easy_order if k in cm])

    def anchors_line(am):
        return " / ".join([f"{k}:{am[k]:.1f}" for k in easy_order if k in am])

    rows = plan["rows"]
    html = []

    html.append(f"<div class='bp-section' data-section='{section_id}'>")
    html.append(f"<h2>{name}</h2>")
    html.append("<ul>")
    html.append(f"<li>섹션 합계(설계 기준): <strong>{plan['section_points']:.1f}점</strong></li>")
    html.append(f"<li>난이도별 문항 수: {counts_line(plan['counts'])}</li>")
    html.append(f"<li>난이도 기준 배점(±0.1 중심): {anchors_line(plan['anchors'])}</li>")
    html.append("</ul>")

    if not rows:
        html.append("<p>(문항 없음)</p></div>")
        return "\n".join(html)

    total_all = _round1(sum(r["배점"] for r in rows))

    html.append("<table class='bp-table'>")
    html.append("<thead><tr><th>#</th><th>난이도</th><th>배점</th><th>체크</th></tr></thead><tbody>")
    for r in rows:
        no = r["번호"]
        lvl = r["난이도"]
        score = r["배점"]
        html.append(
            f"<tr data-row='{no}'>"
            f"<td>{no}</td>"
            f"<td class='bp-level'>{lvl}</td>"
            f"<td class='bp-score' data-score='{score:.1f}'>{score:.1f}</td>"
            f"<td><button type='button' class='bp-toggle' data-checked='false'>☐</button></td>"
            f"</tr>"
        )
    html.append("</tbody></table>")

    html.append(
        f"<p class='bp-summary-all'>소계 합계(전체): "
        f"<strong>{total_all:.1f}점</strong> (검산)</p>"
    )
    html.append(
        f"<p class='bp-summary-remain'>사용하지 않은 문항 합계: "
        f"<strong>{total_all:.1f}점</strong> (체크된 문항은 제외됨)</p>"
    )
    html.append("</div>")

    return "\n".join(html)

# ---------- 파서/보조 ----------
def _parse_int(txt, name):
    if txt is None or str(txt).strip()=="": raise ValueError(f"{name}을(를) 입력하세요.")
    try:
        v=int(float(str(txt).strip()))
        if v<=0: raise ValueError
        return v
    except Exception:
        raise ValueError(f"{name}은(는) 양의 정수로 입력하세요.")

def _parse_float(txt, name):
    if txt is None or str(txt).strip()=="": raise ValueError(f"{name}을(를) 입력하세요.")
    try:
        v=float(str(txt).strip())
        if v<=0: raise ValueError
        return _round1(v)
    except Exception:
        raise ValueError(f"{name}은(는) 양의 숫자로 입력하세요.")

def _counts_to_ratios(count_map: dict, total: int):
    if total<=0: return {k:0.0 for k in count_map}
    s=max(1,total)
    return {k: max(0.0, float(count_map.get(k,0))/s) for k in count_map}

def _scheme_badge_text(scheme: str):
    return "· **난이도:** 상 · 중 · 하" if scheme=="3단계" else "· **난이도:** 상 · 중상 · 중 · 중하 · 하"

def _sum_panel(current, target, title):
    ok = (current==target)
    icon = "✅" if ok else "❗"
    color = "green" if ok else "red"
    return f"<span style='color:{color};font-weight:600'>{icon} {title} 합계: {current} / {target}</span>", ok


def create_blueprint_demo():
    """시험 블루프린트 Gradio 앱 생성"""

    # ---------- CSS (시스템 다크모드 자동 감지) ----------
    css = """
    /* 라이트 모드 (기본) */
    .bp-table { width: 100%; border-collapse: collapse; }
    .bp-table th, .bp-table td {
        border: 1px solid #e2e8f0;
        padding: 4px 8px;
        text-align: center;
    }
    .bp-table th { background-color: #f8fafc; color: #475569; }
    .bp-table td { color: #334155; }
    .bp-section .bp-summary-all, .bp-section .bp-summary-remain { margin-top: 4px; }
    .bp-section tr.used-row td {
        background-color: #e2e8f0;
    }
    .bp-section tr.used-row .bp-level,
    .bp-section tr.used-row .bp-score {
        text-decoration: line-through;
        text-decoration-thickness: 2px;
        color: #94a3b8;
    }
    .card { padding: 10px; }

    /* 다크 모드 자동 감지 */
    @media (prefers-color-scheme: dark) {
        .bp-table th, .bp-table td {
            border-color: #475569;
        }
        .bp-table th { background-color: #334155; color: #e2e8f0; }
        .bp-table td { color: #f1f5f9; }
        .bp-section tr.used-row td {
            background-color: #1e293b;
        }
        .bp-section tr.used-row .bp-level,
        .bp-section tr.used-row .bp-score {
            color: #64748b;
        }
        .bp-toggle { background: #334155; color: #e2e8f0; border: 1px solid #475569; }
    }
    """

    with gr.Blocks(title="시험 블루프린트", css=css, theme=theme) as demo:
        gr.Markdown("### 시험 블루프린트\n**STEP1 → STEP2 → STEP3 → STEP4** 순서로 진행하세요.")

        st_basic = gr.State({})
        st_scheme = gr.State("3단계")
        st_counts = gr.State({})

        with gr.Row():
            with gr.Column(scale=1, min_width=380):
                # STEP1
                gr.Markdown("#### STEP1 · 기본 입력")
                subject = gr.Textbox(label="과목명", placeholder="예: 과학 / 수학 / 역사", elem_id="subject")
                mc_q = gr.Textbox(label="객관식 문항 수(정수)", placeholder="예: 23", elem_id="mc_q")
                cr_q = gr.Textbox(label="서술형 문항 수(정수)", placeholder="예: 5", elem_id="cr_q")
                mc_pts = gr.Textbox(label="객관식 만점 점수", placeholder="예: 80", elem_id="mc_pts")
                cr_pts = gr.Textbox(label="서술형 만점 점수", placeholder="예: 20", elem_id="cr_pts")
                btn_step1 = gr.Button("STEP1 입력", variant="primary", elem_id="btn_step1")

                # STEP2
                step2_header = gr.Markdown("#### STEP2 · 난이도 단계 선택  \n(선택 후 'STEP2 입력' 클릭)")
                scheme = gr.Radio(
                    choices=["3단계","5단계"],
                    value="3단계",
                    label="난이도 단계",
                    interactive=False,
                    elem_id="scheme_radio"
                )
                scheme_badge = gr.Markdown("· **난이도:** 상 · 중 · 하", visible=True)
                btn_step2 = gr.Button("STEP2 입력", variant="primary", interactive=False, elem_id="btn_step2")

                # STEP3
                gr.Markdown("#### STEP3 · 난이도별 문항 수 입력  \n(자동 채움 → 수정 가능, 합계가 맞아야 다음 단계)")
                grp_mc_3 = gr.Group(visible=False)
                with grp_mc_3:
                    gr.Markdown("**객관식 (3단계)**")
                    mc_hi3  = gr.Number(label="상", precision=0, interactive=True, elem_id="mc_hi3")
                    mc_mid3 = gr.Number(label="중", precision=0, interactive=True, elem_id="mc_mid3")
                    mc_lo3  = gr.Number(label="하", precision=0, interactive=True, elem_id="mc_lo3")
                grp_mc_5 = gr.Group(visible=False)
                with grp_mc_5:
                    gr.Markdown("**객관식 (5단계)**")
                    mc_hi5  = gr.Number(label="상",   precision=0, interactive=True, elem_id="mc_hi5")
                    mc_mhi5 = gr.Number(label="중상", precision=0, interactive=True, elem_id="mc_mhi5")
                    mc_mid5 = gr.Number(label="중",   precision=0, interactive=True, elem_id="mc_mid5")
                    mc_mlo5 = gr.Number(label="중하", precision=0, interactive=True, elem_id="mc_mlo5")
                    mc_lo5  = gr.Number(label="하",   precision=0, interactive=True, elem_id="mc_lo5")
                mc_total_md = gr.Markdown("", visible=False)

                grp_cr_3 = gr.Group(visible=False)
                with grp_cr_3:
                    gr.Markdown("**서술형 (3단계)**")
                    cr_hi3  = gr.Number(label="상", precision=0, interactive=True, elem_id="cr_hi3")
                    cr_mid3 = gr.Number(label="중", precision=0, interactive=True, elem_id="cr_mid3")
                    cr_lo3  = gr.Number(label="하", precision=0, interactive=True, elem_id="cr_lo3")
                grp_cr_5 = gr.Group(visible=False)
                with grp_cr_5:
                    gr.Markdown("**서술형 (5단계)**")
                    cr_hi5  = gr.Number(label="상",   precision=0, interactive=True, elem_id="cr_hi5")
                    cr_mhi5 = gr.Number(label="중상", precision=0, interactive=True, elem_id="cr_mhi5")
                    cr_mid5 = gr.Number(label="중",   precision=0, interactive=True, elem_id="cr_mid5")
                    cr_mlo5 = gr.Number(label="중하", precision=0, interactive=True, elem_id="cr_mlo5")
                    cr_lo5  = gr.Number(label="하",   precision=0, interactive=True, elem_id="cr_lo5")
                cr_total_md = gr.Markdown("", visible=False)

                btn_step3 = gr.Button(
                    "STEP3 입력 완료(확인 미리보기 생성)",
                    variant="primary",
                    interactive=False,
                    elem_id="btn_step3"
                )

                # STEP4
                gr.Markdown("#### STEP4 · 최종 생성")
                preview_md = gr.Markdown("(STEP3 완료 후 미리보기가 표시됩니다.)", elem_classes=["card"])
                btn_final = gr.Button("최종 생성", variant="primary", interactive=False, elem_id="btn_final")

            with gr.Column(scale=1, min_width=360):
                mc_md = gr.HTML(elem_classes=["card"], elem_id="mc_section_root")
            with gr.Column(scale=1, min_width=360):
                cr_md = gr.HTML(elem_classes=["card"], elem_id="cr_section_root")
                summary_md = gr.Markdown(elem_classes=["card"])

        # ----- STEP1 -----
        def on_step1(subject_v, mc_q_v, cr_q_v, mc_pts_v, cr_pts_v):
            try:
                _ = _parse_int(mc_q_v, "객관식 문항 수")
                _ = _parse_int(cr_q_v, "서술형 문항 수")
                _ = _parse_float(mc_pts_v, "객관식 만점 점수")
                _ = _parse_float(cr_pts_v, "서술형 만점 점수")
            except Exception as e:
                return ({}, gr.update(interactive=False, value="3단계"),
                        "#### STEP2 · 난이도 단계 선택  \n(선택 후 'STEP2 입력' 클릭)",
                        gr.update(value=_scheme_badge_text("3단계")),
                        gr.update(interactive=False),
                        f"❗ STEP1 오류: {e}")
            basic = {"subject": (subject_v or "(과목)"),
                     "mc_q": mc_q_v, "cr_q": cr_q_v,
                     "mc_pts": mc_pts_v, "cr_pts": cr_pts_v}
            return (basic, gr.update(interactive=True, value="3단계"),
                    "#### STEP2 · 난이도 단계 선택  \n(선택 후 'STEP2 입력' 클릭)",
                    gr.update(value=_scheme_badge_text("3단계")),
                    gr.update(interactive=True),
                    "STEP1 완료! STEP2에서 단계 선택 후 'STEP2 입력'을 누르세요.")

        btn_step1.click(
            fn=on_step1,
            inputs=[subject, mc_q, cr_q, mc_pts, cr_pts],
            outputs=[st_basic, scheme, step2_header, scheme_badge, btn_step2, preview_md]
        )

        def on_mc_pts_change(mc_pts_v):
            try:
                v = float(str(mc_pts_v).strip())
            except Exception:
                return gr.update()
            remain = max(0.0, 100.0 - v)
            return gr.update(value=str(_round1(remain)))
        mc_pts.change(fn=on_mc_pts_change, inputs=[mc_pts], outputs=[cr_pts])

        def on_scheme_change(scheme_v):
            return gr.update(value=_scheme_badge_text(scheme_v))
        scheme.change(fn=on_scheme_change, inputs=[scheme], outputs=[scheme_badge])

        # ----- STEP2 -----
        def _auto_counts(total, scheme_v):
            total = int(total)
            if scheme_v=="3단계":
                ratios = RATIOS_3; keys=["상","중","하"]
            else:
                ratios = RATIOS_5; keys=["상","중상","중","중하","하"]
            cnts=_split_counts(total,[ratios[k] for k in keys])
            return {keys[i]:cnts[i] for i in range(len(keys))}

        def on_step2(stb, scheme_v):
            try:
                mc_total = _parse_int(stb.get("mc_q"), "객관식 문항 수")
                cr_total = _parse_int(stb.get("cr_q"), "서술형 문항 수")
            except Exception as e:
                return (
                    scheme_v, gr.update(value=_scheme_badge_text(scheme_v)),
                    gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                    None, None, None, None, None, None, None, None,
                    gr.update(visible=True, value=f"❗ {e}"),
                    gr.update(visible=True, value=f"❗ {e}"),
                    gr.update(interactive=False),
                    "STEP1을 먼저 완료해주세요."
                )

            mc_counts = _auto_counts(mc_total, scheme_v)
            cr_counts = _auto_counts(cr_total, scheme_v)

            if scheme_v=="3단계":
                vis = (gr.update(visible=True), gr.update(visible=False),
                       gr.update(visible=True), gr.update(visible=False))
                mc_vals = (mc_counts["상"], mc_counts["중"], mc_counts["하"],
                           None, None, None, None, None)
                cr_vals = (cr_counts["상"], cr_counts["중"], cr_counts["하"],
                           None, None, None, None, None)
            else:
                vis = (gr.update(visible=False), gr.update(visible=True),
                       gr.update(visible=False), gr.update(visible=True))
                mc_vals = (None, None, None,
                           mc_counts["상"], mc_counts["중상"], mc_counts["중"], mc_counts["중하"], mc_counts["하"])
                cr_vals = (None, None, None,
                           cr_counts["상"], cr_counts["중상"], cr_counts["중"], cr_counts["중하"], cr_counts["하"])

            mc_panel, mc_ok = _sum_panel(sum(mc_counts.values()), mc_total, "객관식")
            cr_panel, cr_ok = _sum_panel(sum(cr_counts.values()), cr_total, "서술형")

            return (
                scheme_v, gr.update(value=_scheme_badge_text(scheme_v)),
                *vis,
                *mc_vals, *cr_vals,
                gr.update(visible=True, value=mc_panel),
                gr.update(visible=True, value=cr_panel),
                gr.update(interactive=(mc_ok and cr_ok)),
                "STEP2 완료! STEP3에서 문항 수를 자유롭게 수정하세요."
            )

        btn_step2.click(
            fn=on_step2,
            inputs=[st_basic, scheme],
            outputs=[
                st_scheme, scheme_badge,
                grp_mc_3, grp_mc_5, grp_cr_3, grp_cr_5,
                mc_hi3, mc_mid3, mc_lo3, mc_hi5, mc_mhi5, mc_mid5, mc_mlo5, mc_lo5,
                cr_hi3, cr_mid3, cr_lo3, cr_hi5, cr_mhi5, cr_mid5, cr_mlo5, cr_lo5,
                mc_total_md, cr_total_md,
                btn_step3,
                preview_md
            ]
        )

        # --- STEP3 ---
        def on_counts_change(stb, scheme_v,
                             mc_h3, mc_m3, mc_l3, cr_h3, cr_m3, cr_l3,
                             mc_h5, mc_mh5, mc_md5, mc_ml5, mc_l5,
                             cr_h5, cr_mh5, cr_md5, cr_ml5, cr_l5):
            mc_total = _parse_int(stb.get("mc_q"), "객관식 문항 수")
            cr_total = _parse_int(stb.get("cr_q"), "서술형 문항 수")
            def _i(x):
                try: return int(x)
                except: return 0
            if scheme_v=="3단계":
                mc_sum=_i(mc_h3)+_i(mc_m3)+_i(mc_l3)
                cr_sum=_i(cr_h3)+_i(cr_m3)+_i(cr_l3)
            else:
                mc_sum=_i(mc_h5)+_i(mc_mh5)+_i(mc_md5)+_i(mc_ml5)+_i(mc_l5)
                cr_sum=_i(cr_h5)+_i(cr_mh5)+_i(cr_md5)+_i(cr_ml5)+_i(cr_l5)
            mc_panel, mc_ok = _sum_panel(mc_sum, mc_total, "객관식")
            cr_panel, cr_ok = _sum_panel(cr_sum, cr_total, "서술형")
            return (gr.update(value=mc_panel, visible=True),
                    gr.update(value=cr_panel, visible=True),
                    gr.update(interactive=(mc_ok and cr_ok)))

        num_inputs = [
            st_basic, st_scheme,
            mc_hi3, mc_mid3, mc_lo3, cr_hi3, cr_mid3, cr_lo3,
            mc_hi5, mc_mhi5, mc_mid5, mc_mlo5, mc_lo5,
            cr_hi5, cr_mhi5, cr_mid5, cr_mlo5, cr_lo5
        ]
        for comp in [mc_hi3, mc_mid3, mc_lo3, cr_hi3, cr_mid3, cr_lo3,
                     mc_hi5, mc_mhi5, mc_mid5, mc_mlo5, mc_lo5,
                     cr_hi5, cr_mhi5, cr_mid5, cr_mlo5, cr_lo5]:
            comp.change(fn=on_counts_change, inputs=num_inputs,
                        outputs=[mc_total_md, cr_total_md, btn_step3])

        def on_step3_done(stb, scheme_v,
                          mc_h3, mc_m3, mc_l3, cr_h3, cr_m3, cr_l3,
                          mc_h5, mc_mh5, mc_md5, mc_ml5, mc_l5,
                          cr_h5, cr_mh5, cr_md5, cr_ml5, cr_l5):
            def _i(x):
                try: return int(x)
                except: return 0
            mc_total = _parse_int(stb.get("mc_q"), "객관식 문항 수")
            cr_total = _parse_int(stb.get("cr_q"), "서술형 문항 수")

            if scheme_v=="3단계":
                stc={"scheme":"3단계",
                     "mc":{"상":_i(mc_h3),"중":_i(mc_m3),"하":_i(mc_l3)},
                     "cr":{"상":_i(cr_h3),"중":_i(cr_m3),"하":_i(cr_l3)}}
                mc_sum=_i(mc_h3)+_i(mc_m3)+_i(mc_l3)
                cr_sum=_i(cr_h3)+_i(cr_m3)+_i(cr_l3)
                order=["하","중","상"]
            else:
                stc={"scheme":"5단계",
                     "mc":{"상":_i(mc_h5),"중상":_i(mc_mh5),"중":_i(mc_md5),"중하":_i(mc_ml5),"하":_i(mc_l5)},
                     "cr":{"상":_i(cr_h5),"중상":_i(cr_mh5),"중":_i(cr_md5),"중하":_i(cr_ml5),"하":_i(cr_l5)}}
                mc_sum=_i(mc_h5)+_i(mc_mh5)+_i(mc_md5)+_i(mc_ml5)+_i(mc_l5)
                cr_sum=_i(cr_h5)+_i(cr_mh5)+_i(cr_md5)+_i(cr_ml5)+_i(cr_l5)
                order=["하","중하","중","중상","상"]

            mc_ok = (mc_sum==mc_total); cr_ok=(cr_sum==cr_total)
            mc_str = " / ".join([f"{k}:{stc['mc'].get(k,0)}" for k in order if k in stc["mc"]])
            cr_str = " / ".join([f"{k}:{stc['cr'].get(k,0)}" for k in order if k in stc["cr"]])
            prev = f"### 확인용 미리보기\n- 객관식: {mc_str}\n- 서술형: {cr_str}"
            return stc, gr.update(value=prev), gr.update(interactive=(mc_ok and cr_ok))

        btn_step3.click(
            fn=on_step3_done,
            inputs=[st_basic, st_scheme,
                    mc_hi3, mc_mid3, mc_lo3, cr_hi3, cr_mid3, cr_lo3,
                    mc_hi5, mc_mhi5, mc_mid5, mc_mlo5, mc_lo5,
                    cr_hi5, cr_mhi5, cr_mid5, cr_mlo5, cr_lo5],
            outputs=[st_counts, preview_md, btn_final]
        )

        # ----- STEP4 -----
        def on_final(stb, stc):
            subject_v = stb.get("subject") or "(과목)"
            mc_q_v, cr_q_v = stb.get("mc_q"), stb.get("cr_q")
            mc_pts_v, cr_pts_v = stb.get("mc_pts"), stb.get("cr_pts")
            scheme_v = stc.get("scheme","3단계")
            mc_total = _parse_int(mc_q_v,"객관식 문항 수")
            cr_total = _parse_int(cr_q_v,"서술형 문항 수")

            mc_ratios = _counts_to_ratios(stc["mc"], mc_total)
            cr_ratios = _counts_to_ratios(stc["cr"], cr_total)

            if scheme_v=="3단계":
                curve,bias,pmap,order = CURVE_3,BIAS_3,P_3,["하","중","상"]
            else:
                curve,bias,pmap,order = CURVE_5,BIAS_5,P_5,["하","중하","중","중상","상"]

            mc_plan = build_section(mc_total, _parse_float(mc_pts_v,"객관식 만점 점수"),
                                    scheme_v, mc_ratios, curve, bias, pmap, "객관식")
            cr_plan = build_section(cr_total, _parse_float(cr_pts_v,"서술형 만점 점수"),
                                    scheme_v, cr_ratios, curve, bias, pmap, "서술형")

            total_points=_round1(float(mc_pts_v)+float(cr_pts_v))
            g_mean=_round1(mc_plan["exp"][0]+cr_plan["exp"][0])
            g_sd=_round1((mc_plan["exp"][1]**2 + cr_plan["exp"][1]**2)**0.5)

            summary=[
                "## 요약",
                f"- 과목: **{subject_v}**",
                f"- 총 문항수: **{int(mc_total)+int(cr_total)}문항**, 총점 **{total_points:.1f}점**",
                f"- 난이도 체계: **{scheme_v}**",
                f"- 기대 평균(근사): **{g_mean}점**, 기대 표준편차(근사): **{g_sd}점**",
                "> 실제 값은 집단/문항 상관/채점에 따라 달라질 수 있습니다.",
            ]

            mc_html = render_section_html("객관식", mc_plan, order, "mc")
            cr_html = render_section_html("서술형", cr_plan, order, "cr")

            return "\n".join(summary), mc_html, cr_html

        btn_final.click(
            fn=on_final,
            inputs=[st_basic, st_counts],
            outputs=[summary_md, mc_md, cr_md]
        )

        # ---------- 전역 JS (체크박스 취소선/합계) ----------
        keyboard_js = r"""
() => {
  function injectBpStyle() {
    if (document.getElementById("bp-style")) return;
    const s = document.createElement("style");
    s.id = "bp-style";
    s.textContent = `
      .bp-table { width: 100%; border-collapse: collapse; }
      .bp-table th, .bp-table td {
        border: 1px solid #ccc;
        padding: 4px 8px;
        text-align: center;
      }
      .bp-table th { background-color: #f5f5f5; }
      .bp-section .bp-summary-all, .bp-section .bp-summary-remain { margin-top: 4px; }
      .bp-section tr.used-row td {
        background-color: #e3e3e3;
      }
      .bp-section tr.used-row .bp-level,
      .bp-section tr.used-row .bp-score {
        text-decoration: line-through;
        text-decoration-thickness: 2px;
        color: #666;
      }
    `;
    document.head.appendChild(s);
  }

  function setupScoreSection(rootId) {
    const root = document.getElementById(rootId);
    if (!root || root.dataset.bpBound) return;
    root.dataset.bpBound = "1";

    function recalc() {
      const rows = Array.from(root.querySelectorAll(".bp-table tbody tr"));
      if (!rows.length) return;
      let remain = 0;
      rows.forEach(tr => {
        const btn = tr.querySelector(".bp-toggle");
        const scoreCell = tr.querySelector(".bp-score");
        const score = parseFloat(
          scoreCell ? (scoreCell.dataset.score || scoreCell.textContent || "0") : "0"
        ) || 0;

        const checked = btn && btn.dataset.checked === "true";

        if (checked) {
          tr.classList.add("used-row");
        } else {
          tr.classList.remove("used-row");
          remain += score;
        }
      });
      const remEl = root.querySelector(".bp-summary-remain");
      if (remEl) {
        remEl.innerHTML = "사용하지 않은 문항 합계: <strong>" +
          remain.toFixed(1) + "점</strong> (체크된 문항은 제외됨)";
      }
    }

    root.addEventListener("click", function(e) {
      const btn = e.target.closest(".bp-toggle");
      if (!btn) return;
      const now = btn.dataset.checked === "true";
      btn.dataset.checked = now ? "false" : "true";
      btn.textContent = now ? "☐" : "☑";
      recalc();
    });

    setInterval(recalc, 800);
  }

  setTimeout(() => {
    injectBpStyle();
    setupScoreSection("mc_section_root");
    setupScoreSection("cr_section_root");
  }, 300);
}
"""

        def _noop():
            return

        demo.load(fn=_noop, inputs=[], outputs=[], js=keyboard_js)

    return demo


# 단독 실행용
if __name__ == "__main__":
    demo = create_blueprint_demo()
    demo.launch()
