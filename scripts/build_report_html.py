#!/usr/bin/env python
"""최종 보고서 HTML. 본문은 윤문을 마친 마크다운에서 그대로 가져온다."""

import base64
import json
import re
from pathlib import Path

ROOT = Path("/Users/zzuhyeong2/Library/CloudStorage/GoogleDrive-a01056371120@gmail.com/"
            "My Drive/Conference_2026/Juhyeong")
SCRATCH = Path("/private/tmp/claude-501/-Users-zzuhyeong2-Library-CloudStorage-"
               "GoogleDrive-a01056371120-gmail-com-My-Drive-Conference-2026/"
               "5908be5f-8d56-4ccc-a07d-acca1ce3af1d/scratchpad")
FIGS = json.loads((SCRATCH / "figs.json").read_text())
MD = (ROOT / "docs/최종보고서_MIST.md").read_text(encoding="utf-8")

# 인용 블록으로 표시해 둔 그림 자리를 실제 그림으로 바꾼다.
FIGURE_MAP = {
    "R5": ("R5", "골격 거리별 축 크기",
           "학습 골격에서 멀어질수록 오차는 커지지만 축 값은 오히려 줄어든다."),
    "R6": ("R6", "신호 겹침 구조",
           "우리 축과 기존 신호가 상위 집합에서 거의 겹치지 않는다."),
    "R1": ("R1", "제거 실험",
           "기준 모형에 무엇을 더했을 때 위험 선별이 얼마나 개선되는가. 가로선이 95퍼센트 신뢰구간."),
    "R4": ("R4", "안전·위험 20퍼센트",
           "상위·하위 20퍼센트를 걸러냈을 때 크게 틀린 예측이 얼마나 섞이는가."),
    "R2": ("R2", "물성별 효과",
           "CYP 여섯 종이 예외 없이 상위권에 몰려 있다."),
    "R3": ("R3", "사전 등록 재현과 확장",
           "결과를 보기 전에 구성을 고정하고 한 번만 평가했다. 체크 표시는 신뢰구간이 0을 배제한 물성."),
}

def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def inline(s: str) -> str:
    s = esc(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<!\S)(https://\S+)", r'<a href="\1">\1</a>', s)
    return s

def render(md: str) -> str:
    out, lines, i = [], md.splitlines(), 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("## "):
            title = line[3:]
            anchor = "s" + re.sub(r"[^0-9]", "", title.split(".")[0]) if title[0].isdigit() else ""
            attr = f' id="{anchor}"' if anchor else ""
            out.append(f"<h2{attr}>{inline(title)}</h2>")
        elif line.startswith("### "):
            out.append(f"<h3>{inline(line[4:])}</h3>")
        elif line.startswith("> **그림"):
            key = re.search(r"그림 (R\d)", line).group(1)
            num, title, cap = FIGURE_MAP[key]
            out.append(f'<figure><img src="{FIGS[key]}" alt="{esc(title)}">'
                       f'<figcaption><b>그림 {num[1:]}</b> · {esc(title)}<br>{esc(cap)}</figcaption></figure>')
        elif line.startswith("| "):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip("|").split("|")])
                i += 1
            head, body = rows[0], rows[2:]
            th = "".join(f"<th>{inline(c)}</th>" for c in head)
            tb = "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in body)
            out.append(f'<div class="tw"><table><thead><tr>{th}</tr></thead><tbody>{tb}</tbody></table></div>')
            continue
        elif line.startswith("```"):
            block = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(lines[i]); i += 1
            out.append("<pre><code>" + esc("\n".join(block)) + "</code></pre>")
        elif line.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(f"<li>{inline(lines[i][2:])}</li>"); i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue
        elif line.strip() == "---":
            out.append('<hr>')
        elif line.strip():
            out.append(f"<p>{inline(line)}</p>")
        i += 1
    return "\n".join(out)

body = render("## 요약" + MD.split("## 요약", 1)[1])

NAV = [("s1", "1 문제"), ("s2", "2 방법"), ("s3", "3 RQ1"), ("s4", "4 RQ2"), ("s5", "5 RQ3"),
       ("s6", "6 효과의 위치"), ("s7", "7 새 데이터 검증"), ("s8", "8 실패한 시도"),
       ("s9", "9 데모"), ("s10", "10 주장의 범위"), ("s11", "11 한계"), ("s12", "12 남은 과제")]
nav = "".join(f'<a href="#{i}">{esc(t)}</a>' for i, t in NAV)

HEAD = """<h1>같은 분자를 다르게 적으면<br>모델이 답을 바꾼다</h1>
<p class="sub">입력 상태 민감성을 ADMET 예측의 신뢰성 신호로 쓸 수 있는가</p>
<p class="meta">MIST · Molecular Input-State sensitiviTy · TOBIG's 컨퍼런스 2026 · 2026년 9월 27일</p>
<div class="kpis">
  <div class="kpi"><span class="k">+0.0286</span><span class="v">기준 대비 AURC 개선<br>신뢰구간 [+0.0031, +0.0561]</span></div>
  <div class="kpi"><span class="k">6<small>/6</small></span><span class="v">CYP 물성에서<br>모두 개선</span></div>
  <div class="kpi"><span class="k">2<small>/2</small></span><span class="v">사전 등록한 새 데이터에서<br>재현 성공</span></div>
  <div class="kpi"><span class="k">287,819</span><span class="v">만들어 채점한<br>분자 변형</span></div>
</div>"""

html = f"""<title>MIST 입력 상태 민감성</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Serif+KR:wght@500;700&family=IBM+Plex+Sans+KR:wght@400;500;600&family=IBM+Plex+Mono:wght@400&display=swap">
<style>
:root {{
  color-scheme: light;
  --ink: #17201d; --ink-2: #44514c; --muted: #7d8a85;
  --line: #e0e6e3; --line-2: #eef2f0;
  --bg: #f7f9f8; --card: #ffffff; --raise: #fbfcfc;
  --accent: #0d6155; --accent-soft: #e6f0ed;
  --blue: #2a78d6; --amber: #a4522a;
  --serif: "Noto Serif KR", Georgia, serif;
  --sans: "IBM Plex Sans KR", -apple-system, BlinkMacSystemFont, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --ink: #e9eeec; --ink-2: #a9b5b0; --muted: #7f8c87;
    --line: #2a3431; --line-2: #212a28;
    --bg: #111716; --card: #18201e; --raise: #1c2422;
    --accent: #63c3ad; --accent-soft: #16322c;
    --blue: #6aa7e8; --amber: #d89268;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --ink: #e9eeec; --ink-2: #a9b5b0; --muted: #7f8c87;
  --line: #2a3431; --line-2: #212a28;
  --bg: #111716; --card: #18201e; --raise: #1c2422;
  --accent: #63c3ad; --accent-soft: #16322c;
  --blue: #6aa7e8; --amber: #d89268;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--bg); color: var(--ink);
  font-family: var(--sans); font-size: 15.5px; line-height: 1.75;
  -webkit-font-smoothing: antialiased;
}}
.wrap {{ max-width: 860px; margin: 0 auto; padding-inline: 20px; padding-block: 0; }}
header.top {{ border-bottom: 1px solid var(--line); background: var(--card); padding-block: 64px 44px; }}
h1 {{
  font-family: var(--serif); font-weight: 700; font-size: clamp(30px, 5.4vw, 46px);
  line-height: 1.26; letter-spacing: -.028em; margin: 0; text-wrap: balance;
}}
.sub {{ margin: 16px 0 0; font-size: clamp(16px, 2.3vw, 19px); color: var(--ink-2); font-weight: 500; text-wrap: balance; }}
.meta {{ margin: 20px 0 0; font-size: 12.5px; color: var(--muted); letter-spacing: .015em; }}
.kpis {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 38px; }}
.kpi {{ border-top: 2px solid var(--accent); padding-top: 12px; }}
.kpi .k {{ display: block; font-family: var(--serif); font-size: clamp(21px, 3.2vw, 29px); font-weight: 700; letter-spacing: -.03em; color: var(--accent); line-height: 1.15; }}
.kpi .k small {{ font-size: .58em; color: var(--muted); font-weight: 500; }}
.kpi .v {{ display: block; margin-top: 6px; font-size: 12px; color: var(--muted); line-height: 1.5; }}
nav.toc {{ position: sticky; top: env(safe-area-inset-top, 0px); z-index: 5; background: var(--bg);
  border-bottom: 1px solid var(--line); overflow-x: auto; }}
nav.toc .inner {{ display: flex; gap: 2px; max-width: 860px; margin: 0 auto; padding: 8px 20px; }}
nav.toc a {{ white-space: nowrap; font-size: 12.5px; color: var(--muted); text-decoration: none;
  padding: 5px 10px; border-radius: 7px; }}
nav.toc a:hover {{ color: var(--accent); background: var(--accent-soft); }}
main {{ padding-block: 8px 80px; }}
h2 {{
  font-family: var(--serif); font-size: 24px; font-weight: 700; letter-spacing: -.024em;
  margin: 58px 0 0; padding-top: 10px; text-wrap: balance; scroll-margin-top: 60px;
}}
h3 {{ font-size: 16.5px; font-weight: 600; margin: 34px 0 0; letter-spacing: -.012em; color: var(--ink); text-wrap: balance; }}
p {{ margin: 13px 0 0; color: var(--ink-2); max-width: 68ch; }}
p strong, li strong {{ color: var(--ink); font-weight: 600; }}
ul {{ margin: 13px 0 0; padding-left: 20px; color: var(--ink-2); max-width: 68ch; }}
li {{ margin-top: 6px; }}
hr {{ border: 0; border-top: 1px solid var(--line); margin: 46px 0 0; }}
code {{ font-family: var(--mono); font-size: .86em; background: var(--raise); border: 1px solid var(--line-2);
  border-radius: 5px; padding: 1px 5px; color: var(--ink-2); }}
pre {{ margin: 18px 0 0; background: var(--card); border: 1px solid var(--line); border-radius: 12px;
  padding: 16px 18px; overflow-x: auto; }}
pre code {{ background: none; border: 0; padding: 0; font-size: 12.5px; line-height: 1.8; }}
a {{ color: var(--accent); }}
.tw {{ margin: 18px 0 0; overflow-x: auto; border: 1px solid var(--line); border-radius: 12px; background: var(--card); }}
table {{ width: 100%; border-collapse: collapse; font-size: 13.5px; }}
th, td {{ text-align: left; padding: 11px 15px; border-bottom: 1px solid var(--line-2); }}
tbody tr:last-child td {{ border-bottom: 0; }}
th {{ font-weight: 500; font-size: 11.5px; color: var(--muted); background: var(--raise); letter-spacing: .02em; white-space: nowrap; }}
td {{ color: var(--ink-2); font-variant-numeric: tabular-nums; }}
td strong {{ color: var(--ink); }}
figure {{ margin: 26px 0 0; background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 18px; }}
figure img {{ display: block; width: 100%; max-width: 100%; height: auto; border-radius: 6px; }}
figcaption {{ margin-top: 13px; font-size: 12.5px; color: var(--muted); line-height: 1.6; }}
figcaption b {{ color: var(--accent); font-weight: 600; }}
footer {{ border-top: 1px solid var(--line); padding-block: 26px 40px; }}
footer p {{ color: var(--muted); font-size: 12.5px; margin: 0; }}
@media (max-width: 720px) {{
  .kpis {{ grid-template-columns: repeat(2, 1fr); gap: 16px 12px; }}
  header.top {{ padding-block: 42px 34px; }}
  h2 {{ font-size: 21px; margin-top: 46px; }}
}}
</style>
<header class="top"><div class="wrap">{HEAD}</div></header>
<nav class="toc"><div class="inner">{nav}</div></nav>
<main class="wrap">
{body}
</main>
<footer><div class="wrap"><p>TOBIG's 컨퍼런스 2026 · MIST · 모든 수치는 산출물과 기계 대조로 확인했다</p></div></footer>
"""
out = SCRATCH / "report.html"
out.write_text(html, encoding="utf-8")
print(f"{out} · {len(html)//1024} KB")
