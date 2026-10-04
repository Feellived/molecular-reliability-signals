#!/usr/bin/env python
"""상세 보고서의 부록 표를 산출물에서 직접 만들어 끼운다.

22종 × 10열을 손으로 옮겨 적으면 어딘가 틀린다. 자리표시자만 본문에 두고
수치는 매번 CSV에서 읽는다.
"""
import html
from pathlib import Path

import pandas as pd

J = Path("/Users/zzuhyeong2/Library/CloudStorage/GoogleDrive-a01056371120@gmail.com/"
         "My Drive/Conference_2026/Juhyeong")
PROC = J / "data/processed"
PAGE = Path(__file__).parent / "detailed/index.html"


def cell(value: float, digits: int = 4, sign: bool = True) -> str:
    """양수는 강조, 음수는 경고색. 둘 다 아니면 그냥 둔다."""
    text = f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"
    text = text.replace("-", "−")
    klass = " class='pos'" if sign and value > 0 else (" class='neg'" if sign and value < 0 else "")
    return f"<span{klass}>{text}</span>"


def appendix_a() -> str:
    b1 = pd.read_csv(PROC / "scores_role4_r2/b1_statistics/b1_by_dataset.csv")
    b1 = b1.sort_values("effect__기준+B", ascending=False)
    rows = []
    for _, r in b1.iterrows():
        cyp = r.dataset.startswith("cyp")
        lo = f"{r.scaffold_ci_low:+.3f}".replace("-", "−")
        hi = f"{r.scaffold_ci_high:+.3f}".replace("-", "−")
        mark = "<strong>" if r.p_bh < 0.05 else ""
        end = "</strong>" if r.p_bh < 0.05 else ""
        rows.append(
            f"<tr{' class=\"hi\"' if cyp else ''}>"
            f"<td>{html.escape(r.dataset)}</td>"
            f"<td class='n'>{int(r.n_test):,}</td>"
            f"<td class='n'>{int(r.n_scaffold):,}</td>"
            f"<td class='n'>{r['aurc__기준']:.4f}</td>"
            f"<td class='n'>{r['aurc__기준+B']:.4f}</td>"
            f"<td class='n'>{mark}{cell(r['effect__기준+B'])}{end}</td>"
            f"<td class='n'>[{lo}, {hi}]</td>"
            f"<td class='n'>{r.p_bootstrap:.3f}</td>"
            f"<td class='n'>{mark}{r.p_bh:.3f}{end}</td>"
            f"<td class='n'>{cell(r['effect__기준+B(MAD)'])}</td>"
            f"<td class='n'>{cell(r['effect__기준+A'])}</td></tr>")
    head = ("<tr><th>물성</th><th class='n'>시험</th><th class='n'>골격</th>"
            "<th class='n'>기준 AURC</th><th class='n'>+B AURC</th><th class='n'>효과</th>"
            "<th class='n'>골격 95% 구간</th><th class='n'>p</th><th class='n'>BH</th>"
            "<th class='n'>효과 (MAD)</th><th class='n'>효과 (A축)</th></tr>")
    cap = ("파랗게 칠한 행이 CYP 물성 6종이다. 굵은 글씨는 BH 보정 후 p &lt; 0.05인 3종. "
           "solubility_aqsoldb는 허용성 판정에서 B1 두 축이 모두 주의라 주 분석에서 빠져 효과가 정확히 0이다.")
    return (f"<div class='tw'><table><thead>{head}</thead><tbody>{''.join(rows)}</tbody>"
            f"<caption>{cap}</caption></table></div>")


def appendix_b() -> str:
    t = pd.read_csv(PROC / "pipeline_yoonsoo/reports/06_transformation_allowance_final.csv",
                    encoding="utf-8-sig")
    badge = {"허용": "<span class='pos'>허용</span>",
             "주의": "<span class='dim'>주의</span>",
             "충돌": "<span class='neg'>충돌</span>"}
    rows = []
    for _, r in t.iterrows():
        cyp = r.dataset.startswith("cyp")
        task = "회귀" if r.task_type == "regression" else "분류"
        rows.append(
            f"<tr{' class=\"hi\"' if cyp else ''}>"
            f"<td>{html.escape(r.dataset)}</td>"
            f"<td class='dim'>{task}</td>"
            f"<td>{badge.get(r.B1_tautomer, r.B1_tautomer)}</td>"
            f"<td>{badge.get(r.B1_protonation, r.B1_protonation)}</td>"
            f"<td>{badge.get(r.B2_salt, r.B2_salt)}</td>"
            f"<td>{badge.get(r.B3_stereo, r.B3_stereo)}</td></tr>")
    head = ("<tr><th>물성</th><th>과제</th><th>B1 호변이성질체</th><th>B1 양성자화</th>"
            "<th>B2 염 형태</th><th>B3 입체 표기</th></tr>")
    cap = ("B2는 허용 판정을 받고도 사전 선별에서 22종 모두 탈락해 변형을 생성하지 않았다. "
           "B3는 solubility_aqsoldb·ld50_zhu·dili 세 종이 같은 이유로 빠져 19종이다.")
    return (f"<div class='tw'><table><thead>{head}</thead><tbody>{''.join(rows)}</tbody>"
            f"<caption>{cap}</caption></table></div>")


def main() -> int:
    page = PAGE.read_text(encoding="utf-8")
    for marker, builder in (("<!--TABLE:appendix_a-->", appendix_a),
                            ("<!--TABLE:appendix_b-->", appendix_b)):
        assert marker in page, f"자리표시자 없음: {marker}"
        page = page.replace(marker, builder())
    PAGE.write_text(page, encoding="utf-8")
    print(f"표 2개 삽입 · {len(page):,}바이트")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
