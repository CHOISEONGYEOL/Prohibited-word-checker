# gradio_counter.py
# =========================================
# 12345 카운터 - 정답 분포 분석 도구
# =========================================

import os
import json
import gradio as gr


def create_table_html(num_questions, current_values=None, highlight_nums=None, empty_idxs=None, invalid_idxs=None):
    """
    HTML 표를 생성하는 함수
    - highlight_nums: 최다 빈도 (빨강)
    - empty_idxs: 빈칸 (노랑)
    - invalid_idxs: 1~5 이외의 값 (보라)
    """
    try:
        n = int(num_questions)
    except:
        n = 0

    if n < 1:
        return "<div style='padding: 40px; text-align: center; color: #64748b; font-size: 16px;'>👈 왼쪽에서 <b>문항 수</b>를 입력하고<br><b>[새 표 만들기]</b> 버튼을 눌러주세요.</div>"

    if current_values is None:
        current_values = []

    # 세트 변환 (빠른 조회를 위해)
    target_nums = set(highlight_nums) if highlight_nums else set()
    empty_set = set(empty_idxs) if empty_idxs else set()
    invalid_set = set(invalid_idxs) if invalid_idxs else set()

    # CSS 스타일
    style = """
    <style>
        .custom-table-container {
            max-height: 600px;
            overflow-y: auto;
            border: 1px solid #e5e7eb;
            background: white;
            padding: 5px;
            border-radius: 8px;
        }
        .custom-table { width: 100%; border-collapse: collapse; }
        .custom-table th {
            position: sticky; top: 0;
            background: #f8fafc;
            padding: 10px;
            border-bottom: 2px solid #e2e8f0;
            color: #475569;
            z-index: 10;
        }
        .custom-table td {
            border-bottom: 1px solid #f1f5f9;
            padding: 4px;
            text-align: center;
        }

        .ans-input {
            width: 100%; max-width: 100px; padding: 10px; text-align: center;
            border: 1px solid #cbd5e1; border-radius: 6px; font-size: 16px; font-weight: 600;
            color: #334155;
            -moz-appearance: textfield;
            transition: all 0.2s;
        }
        .ans-input:focus {
            border-color: #3b82f6;
            background-color: #eff6ff;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
            outline: none;
        }

        /* 1. 최다 빈도 (빨강) */
        .highlight-row { background-color: #fee2e2; }
        .highlight-row input { background-color: #fff1f2; border-color: #fca5a5; color: #be123c; }

        /* 2. 빈칸 (노랑 Warning) */
        .empty-row { background-color: #fef9c3; }
        .empty-row input { background-color: #fefce8; border-color: #fde047; }

        /* 3. 오류 값 (보라 Error) */
        .invalid-row { background-color: #f3e8ff; }
        .invalid-row input { background-color: #faf5ff; border-color: #d8b4fe; color: #7e22ce; }

        .big-input textarea, .big-input input {
            height: 60px !important;
            min-height: 60px !important;
            font-size: 24px !important;
            text-align: center !important;
            padding: 10px !important;
            line-height: normal !important;
        }

        /* 요약표 스타일 */
        .summary-table { width: 100%; border-collapse: collapse; margin-top: 10px; border: 1px solid #e5e7eb; }
        .summary-table th { background: #f1f5f9; padding: 8px; text-align: center; border-bottom: 2px solid #e2e8f0; }
        .summary-table td { padding: 8px; text-align: center; border-bottom: 1px solid #e5e7eb; }

        .row-max { background-color: #fee2e2; color: #991b1b; font-weight: bold; } /* 최다 */
        .row-min { background-color: #eff6ff; color: #1e40af; font-weight: bold; } /* 최소 */
    </style>
    """

    rows = ""
    for i in range(1, n + 1):
        idx = i - 1
        val_str = ""

        # 값 가져오기
        if idx < len(current_values):
            raw_val = current_values[idx]
            val_str = str(raw_val) if raw_val is not None else ""

        # 클래스 우선순위 결정 (오류 > 빈칸 > 최다빈도)
        row_class = ""

        if idx in invalid_set:
            row_class = "invalid-row" # 보라색
        elif idx in empty_set:
            row_class = "empty-row"   # 노란색
        elif val_str and val_str.isdigit() and int(val_str) in target_nums:
            row_class = "highlight-row" # 빨간색

        # [키보드 로직]
        on_keydown_code = (
            "if(event.key==='Enter'){"
            "  event.preventDefault();"
            f"  var next = document.getElementById('q_{i+1}');"
            "  if(next){ next.focus(); next.select(); }"
            "  else { document.getElementById('analyze_btn').click(); }"
            "} else if(event.key==='ArrowDown'){"
            "  event.preventDefault();"
            f"  var next = document.getElementById('q_{i+1}');"
            "  if(next){ next.focus(); next.select(); }"
            "} else if(event.key==='ArrowUp'){"
            "  event.preventDefault();"
            f"  var prev = document.getElementById('q_{i-1}');"
            "  if(prev){ prev.focus(); prev.select(); }"
            "}"
        )

        rows += f"""
            <tr class="{row_class}">
                <td><b style="color:#64748b;">{i}번</b></td>
                <td>
                    <input type='number'
                           id='q_{i}'
                           value='{val_str}'
                           class='ans-input'
                           onkeydown="{on_keydown_code}"
                           autocomplete='off'>
                </td>
            </tr>
        """

    return f"{style}<div class='custom-table-container'><table class='custom-table'><thead><tr><th>번호</th><th>정답</th></tr></thead><tbody>{rows}</tbody></table></div>"


def init_table(num_questions):
    return create_table_html(num_questions)


def analyze_and_update(num_questions, json_data):
    CHOICES = [1, 2, 3, 4, 5]

    # 원본 데이터(화면 표시용)와 유효 데이터(계산용) 분리
    raw_values_for_display = []
    valid_answers = []

    empty_idxs = []
    invalid_idxs = []

    try:
        nq = int(num_questions)
    except:
        nq = 0
    if nq < 1: nq = 1

    if json_data:
        raw_list = json.loads(json_data)

        # 설정된 문항 수(nq)만큼만 처리
        for i in range(nq):
            if i < len(raw_list):
                s = str(raw_list[i]).strip()
            else:
                s = ""

            if s == "":
                # 1. 빈칸인 경우
                empty_idxs.append(i)
                raw_values_for_display.append(None)
            else:
                try:
                    val = int(float(s))
                    raw_values_for_display.append(val)

                    if val in CHOICES:
                        valid_answers.append(val)
                    else:
                        # 2. 1~5 범위를 벗어난 경우
                        invalid_idxs.append(i)
                except:
                    # 숫자가 아닌 문자
                    invalid_idxs.append(i)
                    raw_values_for_display.append(s)

    total_input = len(valid_answers)

    # ---------------------------------------------------------
    # 통계 계산
    # ---------------------------------------------------------
    counts = {i: 0 for i in CHOICES}
    for ans in valid_answers:
        counts[ans] += 1

    # 최다/최소 계산
    if total_input > 0:
        max_count = max(counts.values())
        min_count = min(counts.values())
        diff = max_count - min_count
    else:
        max_count = 0
        min_count = 0
        diff = 0

    max_targets = []
    min_targets = []
    highlight_nums = []

    # 차이가 2개 이상일 때만 하이라이트
    if diff >= 2:
        max_targets = [k for k, v in counts.items() if v == max_count]
        min_targets = [k for k, v in counts.items() if v == min_count]
        highlight_nums = max_targets

    # ---------------------------------------------------------
    # 요약표 생성
    # ---------------------------------------------------------
    summary_rows = ""
    for i in CHOICES:
        cnt = counts[i]
        ratio = (cnt / total_input) * 100 if total_input > 0 else 0

        row_style_class = ""
        if i in max_targets:
            row_style_class = "row-max"
        elif i in min_targets:
            row_style_class = "row-min"

        summary_rows += f"""
            <tr class="{row_style_class}">
                <td>{i}번</td>
                <td>{cnt}개</td>
                <td>{ratio:.1f}%</td>
            </tr>
        """

    summary_html = f"""
    <h3>정답 개수 요약</h3>
    <table class='summary-table'>
        <thead>
            <tr>
                <th>번호</th>
                <th>개수</th>
                <th>비율</th>
            </tr>
        </thead>
        <tbody>
            {summary_rows}
        </tbody>
    </table>
    """

    # ---------------------------------------------------------
    # 상세 분석 메시지
    # ---------------------------------------------------------
    detail_lines = []
    warning_msgs = []

    # 1. 입력 오류 경고
    if invalid_idxs:
        invalid_str = ", ".join([f"{x+1}번" for x in invalid_idxs])
        warning_msgs.append(f"<b>{invalid_str}</b>에 1~5 이외의 숫자가 있습니다. (보라색 칸)")

    if empty_idxs:
        empty_str = ", ".join([f"{x+1}번" for x in empty_idxs])
        warning_msgs.append(f"<b>{empty_str}</b> 정답이 비어있습니다. (노란색 칸)")

    # 2. 분포 분석
    if total_input > 0:
        if diff >= 2:
            detail_lines.append("<hr style='margin: 15px 0; border: 0; border-top: 1px solid #e5e7eb;'>")
            detail_lines.append("### 최다 빈도 문항 분석")

            for t in max_targets:
                # 유효 데이터 내에서 해당 번호 위치 찾기
                q_list = []
                for idx, val in enumerate(raw_values_for_display):
                    if val == t:
                        q_list.append(f"{idx + 1}번")
                q_str = ", ".join(q_list)

                detail_lines.append(f"**{t}번이 가장 많습니다! ({len(q_list)}개)**")
                detail_lines.append(f"<div style='font-size: 24px; font-weight: bold; color: #dc2626; margin: 10px 0; line-height: 1.4;'>{q_str}</div>")

            recommend_str = ", ".join([str(x)+"번" for x in min_targets])
            detail_lines.append(
                f"**TIP: 위 문항 중 일부를 가장 적은 <span style='font-size: 24px; font-weight: bold; color: #2563eb;'>{recommend_str}(으)로 변경</span>을 고려해보세요!**"
            )

            warning_msgs.append(f"{', '.join([str(x)+'번' for x in max_targets])} 정답이 다른 번호보다 2개 이상 많습니다.")

        elif diff == 0:
            warning_msgs.append("모든 번호의 개수가 완벽하게 같습니다!")
        else:
            warning_msgs.append("정답 분포가 대체로 고릅니다. (1개 차이)")

    else:
        warning_msgs.append("유효한 정답이 하나도 없습니다.")

    # 메시지 합치기
    full_warning_msg = "<br>".join(warning_msgs)
    full_result_html = summary_html + "\n\n" + "\n".join(detail_lines)

    progress_ratio = min(total_input / nq, 1.0)
    progress_text = f"입력 현황: {total_input} / {nq} 문항 ({progress_ratio * 100:.1f}%)"

    # 표 재생성
    new_table_html = create_table_html(nq, raw_values_for_display, highlight_nums, empty_idxs, invalid_idxs)

    return new_table_html, full_result_html, full_warning_msg, progress_text


# JS for extracting data from inputs
get_data_js = """
(num, _ignored) => {
    const inputs = document.querySelectorAll('.ans-input');
    const values = Array.from(inputs).map(i => i.value);
    return [num, JSON.stringify(values)];
}
"""


def create_counter_demo():
    """12345 카운터 Gradio 앱 생성"""

    # 시스템 다크모드 자동 감지 CSS
    custom_css = """
    /* 라이트 모드에서 HTML 테이블 스타일 */
    .custom-table-container { background: #ffffff !important; }
    .custom-table th { background: #f8fafc !important; color: #475569 !important; }
    .custom-table td { border-bottom-color: #e2e8f0 !important; }
    .ans-input { background: #ffffff !important; border-color: #cbd5e1 !important; color: #334155 !important; }
    .summary-table th { background: #f1f5f9 !important; }
    .summary-table td { border-bottom-color: #e5e7eb !important; }

    /* 다크 모드 자동 감지 */
    @media (prefers-color-scheme: dark) {
        .custom-table-container { background: #1e293b !important; border-color: #334155 !important; }
        .custom-table th { background: #334155 !important; color: #e2e8f0 !important; border-bottom-color: #475569 !important; }
        .custom-table td { border-bottom-color: #334155 !important; }
        .ans-input { background: #1e293b !important; border-color: #475569 !important; color: #f1f5f9 !important; }
        .ans-input:focus { background-color: #334155 !important; border-color: #60a5fa !important; }
        .highlight-row { background-color: #7f1d1d !important; }
        .highlight-row input { background-color: #991b1b !important; border-color: #f87171 !important; color: #fecaca !important; }
        .empty-row { background-color: #713f12 !important; }
        .empty-row input { background-color: #854d0e !important; border-color: #fbbf24 !important; }
        .invalid-row { background-color: #581c87 !important; }
        .invalid-row input { background-color: #6b21a8 !important; border-color: #c084fc !important; color: #e9d5ff !important; }
        .summary-table { border-color: #475569 !important; }
        .summary-table th { background: #334155 !important; color: #e2e8f0 !important; border-bottom-color: #475569 !important; }
        .summary-table td { border-bottom-color: #475569 !important; color: #e2e8f0 !important; }
        .row-max { background-color: #7f1d1d !important; color: #fecaca !important; }
        .row-min { background-color: #1e3a5f !important; color: #bfdbfe !important; }
    }
    """

    with gr.Blocks(title="12345 카운터", css=custom_css) as demo:
        gr.Markdown("# 12345 카운터")
        gr.Markdown("> **[Enter]**: 입력 후 아래로 (마지막 문항에선 **자동 분석**) / **[화살표]**: 위아래 이동")

        with gr.Row():
            # 왼쪽: 설정 및 버튼
            with gr.Column(scale=1):
                gr.Markdown("### 설정")

                num_questions = gr.Textbox(
                    label="문항 수",
                    placeholder="예: 30",
                    value="",
                    elem_classes="big-input"
                )

                set_btn = gr.Button("새 표 만들기", variant="secondary")

                gr.Markdown("---")

                analyze_btn = gr.Button("분석하기 (결과 보기)", variant="primary", size="lg", elem_id="analyze_btn")

            # 가운데: 표
            with gr.Column(scale=1.5):
                gr.Markdown("### 정답 입력")
                table_html = gr.HTML(value=create_table_html(0))

            # 오른쪽: 결과
            with gr.Column(scale=1.5):
                summary_out = gr.HTML(value="<h3>정답 개수 요약</h3><p style='color:gray'>(분석 버튼을 누르면 나타납니다)</p>")
                warning_out = gr.HTML("")
                progress_out = gr.Markdown("")

        # ----- 이벤트 연결 -----

        set_btn.click(init_table, inputs=[num_questions], outputs=[table_html])
        num_questions.submit(init_table, inputs=[num_questions], outputs=[table_html])

        analyze_btn.click(
            analyze_and_update,
            inputs=[num_questions, table_html],
            outputs=[table_html, summary_out, warning_out, progress_out],
            js=get_data_js
        )

    return demo


# 단독 실행용
if __name__ == "__main__":
    demo = create_counter_demo()
    demo.launch()
