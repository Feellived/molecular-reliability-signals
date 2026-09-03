const $ = (id) => document.getElementById(id);
const pct = (v) => v === null || v === undefined ? "—" : `${(v * 100).toFixed(0)}번째 백분위`;
const num = (v, d = 3) => v === null || v === undefined ? "—" : Number(v).toFixed(d);

let catalogue = [];

async function loadDatasets() {
  const res = await fetch("/api/datasets");
  const data = await res.json();
  catalogue = data.datasets;
  const select = $("dataset");
  for (const item of catalogue) {
    const option = document.createElement("option");
    option.value = item.dataset;
    option.textContent = `${item.dataset} (${item.task_type === "classification" ? "분류" : "회귀"})`;
    select.appendChild(option);
  }
}

function setStatus(message, isError = false) {
  const node = $("status");
  node.hidden = !message;
  node.textContent = message || "";
  node.classList.toggle("error", isError);
}

// 계획서 7.3절은 세 축을 분리해 표시하고 단일 점수로 합산하지 말 것을 정한다.
function renderAxes(axes) {
  const representation = axes["표현 안정성"];
  const state = axes["입력 상태 민감성"];
  const space = axes["화학 공간 위치"];
  const cards = [
    {
      title: "표현 안정성",
      usable: representation.usable,
      percentile: representation.percentile,
      caption: representation.usable
        ? `등가 표기 ${representation.n_variants}종에서의 흔들림`
        : representation.reason,
    },
    {
      title: "입력 상태 민감성",
      usable: true,
      percentile: state.percentile,
      caption: `조건이 성립하는 축의 변형을 합친 흔들림`,
    },
    {
      title: "화학 공간 위치",
      usable: true,
      percentile: space.percentile,
      caption: `가까운 5개와의 유사도 ${num(space.nearest5_tanimoto, 2)} · 이웃 ${space["neighbors_over_0.40"]}개`,
    },
  ];
  $("axes").innerHTML = cards.map((card) => {
    const value = card.percentile;
    const hot = value !== null && value !== undefined && value >= 0.85;
    const width = value === null || value === undefined ? 0 : value * 100;
    return `<div class="axis${card.usable ? "" : " off"}">
      <h3>${card.title}</h3>
      <div class="num">${pct(value)}</div>
      <div class="bar${hot ? " hot" : ""}"><span style="width:${width}%"></span></div>
      <div class="cap">${card.caption}</div>
    </div>`;
  }).join("");
}

// 쓰지 않은 축을 왜 제외했는지 보여주는 것이 이 도구의 고유한 부분이다.
function renderDetail(axes) {
  const rows = axes["입력 상태 민감성"].axes.map((axis) => {
    if (!axis.usable) {
      return `<tr class="off"><td>${axis.name}</td><td>제외</td><td colspan="3">${axis.reason}</td></tr>`;
    }
    const example = axis.examples[0];
    return `<tr>
      <td>${axis.name}</td>
      <td>${axis.n_variants}종</td>
      <td>${num(axis.dispersion, 4)}</td>
      <td>${pct(axis.percentile)}</td>
      <td class="mono">${example ? `${example.smiles} → ${example.prediction}` : "변형 없음"}</td>
    </tr>`;
  }).join("");
  $("detail").innerHTML = `<table>
    <thead><tr><th>축</th><th>변형</th><th>흩어짐</th><th>순위</th><th>변형 예시</th></tr></thead>
    <tbody>${rows}</tbody></table>`;
}

function renderMeta(data) {
  const risk = data.combined_risk;
  $("meta-body").innerHTML = `<table><tbody>
    <tr><th>정준 SMILES</th><td class="mono">${data.canonical_smiles}</td></tr>
    <tr><th>결합 위험 점수</th><td>${risk ? `${num(risk.score)} · ${risk.note}` : "산출하지 않음"}</td></tr>
    <tr><th>설정 지문</th><td class="mono">${data.settings_digest}</td></tr>
  </tbody></table>`;
}

function render(data) {
  const classification = data.task_type === "classification";
  $("pred-label").textContent = classification ? "양성 확률" : "예측값";
  $("pred-value").textContent = num(data.prediction, classification ? 3 : 2);
  $("pred-interval").textContent = data.interval
    ? `${num(data.interval.lower, 2)} ~ ${num(data.interval.upper, 2)} (${Math.round(data.interval.coverage * 100)}% 구간)`
    : "";

  $("verdict-level").textContent = data.verdict.level;
  $("verdict-level").className = `badge ${data.verdict.level}`;
  $("verdict-headline").textContent = data.verdict.headline;
  $("verdict-notes").innerHTML = data.verdict.notes.map((n) => `<li>${n}</li>`).join("");

  renderAxes(data.reliability_axes);
  renderDetail(data.reliability_axes);
  renderMeta(data);
  $("result").hidden = false;
}

async function submit(event) {
  event.preventDefault();
  const smiles = $("smiles").value.trim();
  const dataset = $("dataset").value;
  if (!smiles) return;
  $("submit").disabled = true;
  $("result").hidden = true;
  setStatus("변형을 만들고 두 모델로 채점하는 중. 물성을 처음 부르면 몇 초 걸린다.");
  try {
    const res = await fetch("/api/score", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset, smiles }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "채점에 실패했다");
    setStatus("");
    render(data);
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    $("submit").disabled = false;
  }
}

$("query").addEventListener("submit", submit);
document.querySelectorAll(".examples button").forEach((button) => {
  button.addEventListener("click", () => {
    $("smiles").value = button.dataset.smiles;
    $("dataset").value = button.dataset.dataset;
    $("query").requestSubmit();
  });
});
loadDatasets();
