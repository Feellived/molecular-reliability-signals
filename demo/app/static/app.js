const $ = (id) => document.getElementById(id);
const HIGH = 0.85;
const has = (v) => v !== null && v !== undefined;
const pctText = (v) => has(v) ? `${Math.round(v * 100)}` : "—";
const num = (v, d = 3) => has(v) ? Number(v).toFixed(d) : "—";

// 두 가지로 돈다. 서버가 있으면 입력한 분자를 그때 채점하고, 정적 배포에서는
// 미리 채점해둔 결과를 읽는다. data/index.json이 있는지로 판별한다.
let MODE = "api";
let CATALOGUE = [];
let BUILT_AT = "";

const label = (task) => (task === "classification" ? "분류" : "회귀");

async function boot() {
  try {
    const res = await fetch("data/index.json", { cache: "no-store" });
    if (res.ok) {
      CATALOGUE = (await res.json()).molecules;
      MODE = "static";
      return setupStatic();
    }
  } catch (error) { /* 서버 모드로 떨어진다 */ }
  return setupApi();
}

async function setupApi() {
  const info = await (await fetch("/api/datasets")).json();
  const { datasets } = info;
  BUILT_AT = (info.created_at || "").slice(0, 10);
  const select = $("dataset");
  for (const item of datasets) {
    const option = document.createElement("option");
    option.value = item.dataset;
    option.textContent = `${item.dataset} · ${label(item.task_type)}`;
    select.appendChild(option);
  }
}

// 정적 모드에서는 SMILES를 직접 받을 수 없다. 미리 채점해둔 분자 중에서 고른다.
function setupStatic() {
  document.body.classList.add("static-mode");
  const datasets = [...new Set(CATALOGUE.map((m) => m.dataset))];
  const select = $("dataset");
  for (const name of datasets) {
    const item = CATALOGUE.find((m) => m.dataset === name);
    const option = document.createElement("option");
    option.value = name;
    option.textContent = `${name} · ${label(item.task_type)}`;
    select.appendChild(option);
  }

  const field = $("smiles").closest(".field");
  field.innerHTML = `<label for="molecule">분자</label>
    <select id="molecule"></select>`;
  select.addEventListener("change", fillMolecules);
  fillMolecules();

  const note = document.createElement("p");
  note.className = "static-note";
  note.textContent = `미리 채점해둔 ${CATALOGUE.length}개 분자를 담았다. `
    + "임의의 분자를 넣으려면 저장소를 내려받아 서버 판으로 실행한다.";
  $("landing").appendChild(note);
}

function fillMolecules() {
  const dataset = $("dataset").value;
  const molecule = $("molecule");
  molecule.innerHTML = "";
  for (const item of CATALOGUE.filter((m) => m.dataset === dataset)) {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = (item.name ? `${item.name} · ` : "")
      + `${item.verdict} · ${item.smiles.slice(0, 26)}${item.smiles.length > 26 ? "…" : ""}`;
    molecule.appendChild(option);
  }
}

function setStatus(message, isError = false) {
  const node = $("status");
  node.hidden = !message;
  node.textContent = message || "";
  node.classList.toggle("error", isError);
}

// 계획서 7.3절: 세 축을 분리해 표시하고 단일 점수로 합산하지 않는다.
function renderAxes(axes) {
  const a = axes["표현 안정성"];
  const b = axes["입력 상태 민감성"];
  const c = axes["화학 공간 위치"];
  const cards = [
    { title: "표현 안정성", usable: a.usable, value: a.percentile,
      cap: a.usable ? `등가 표기 ${a.n_variants}종에서의 흔들림` : a.reason },
    { title: "입력 상태 민감성", usable: true, value: b.percentile,
      cap: "조건이 성립하는 축의 변형을 합친 흔들림" },
    { title: "화학 공간 위치", usable: true, value: c.percentile,
      cap: `가까운 5개와 유사도 ${num(c.nearest5_tanimoto, 2)} · 이웃 ${c["neighbors_over_0.40"]}개` },
  ];
  $("axes").innerHTML = cards.map((card) => {
    const hot = has(card.value) && card.value >= HIGH;
    const width = has(card.value) ? card.value * 100 : 0;
    const body = card.usable
      ? `<div class="num">${pctText(card.value)}<small>백분위</small></div>
         <div class="bar"><span style="--w:${width}%"></span></div>`
      : `<div class="num">산출 불가</div><div class="bar"></div>`;
    return `<div class="axis${hot ? " hot" : ""}${card.usable ? "" : " off"}">
      <h3>${card.title}</h3>${body}<div class="cap">${card.cap}</div></div>`;
  }).join("");
}

// 거의 같아 보이는 분자가 다른 답을 받는 장면이 이 연구의 핵심이다.
function renderShifts(data) {
  const parts = [];
  for (const axis of data.reliability_axes["입력 상태 민감성"].axes) {
    for (const example of axis.examples || []) {
      parts.push({ label: axis.name, ...example });
    }
  }
  const block = $("shift-block");
  if (!parts.length) { block.hidden = true; return; }
  block.hidden = false;
  parts.sort((x, y) => Math.abs(y.shift) - Math.abs(x.shift));
  const card = (label, svg, value, shift, isOrigin) => {
    const delta = isOrigin ? ""
      : `<span class="delta ${shift < 0 ? "down" : "up"}">${shift > 0 ? "+" : ""}${num(shift, 3)}</span>`;
    return `<div class="shift${isOrigin ? " origin" : ""}">
      <div class="top"><span>${label}</span>${isOrigin ? "<span>기준</span>" : ""}</div>
      <div class="art">${svg || ""}</div>
      <div class="bottom"><span class="val">${num(value, 3)}</span>${delta}</div>
    </div>`;
  };
  $("shifts").innerHTML =
    card("원본 표기", data.svg, data.prediction, 0, true) +
    parts.slice(0, 2).map((p) => card(p.label, p.svg, p.prediction, p.shift, false)).join("");
}

// 변형 예측이 어디에 흩어지는지를 점으로 그린다. 흔들림이 무엇인지가
// 숫자보다 눈으로 먼저 들어온다. 눈금은 같은 모델끼리만 공유한다.
function strip(origin, spread, lo, hi) {
  if (!spread || !spread.length) return "";
  const W = 190, H = 26, pad = 7;
  const at = (v) => pad + ((v - lo) / (hi - lo || 1)) * (W - pad * 2);
  const dots = spread.map((v, i) =>
    `<circle cx="${at(v.prediction).toFixed(1)}" cy="${H / 2}" r="3.4"
       fill="currentColor" style="animation-delay:${420 + i * 45}ms"
       ><title>${v.prediction}</title></circle>`).join("");
  return `<svg class="strip" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">
    <line x1="${pad}" y1="${H / 2}" x2="${W - pad}" y2="${H / 2}" stroke="var(--line)" stroke-width="2"/>
    ${dots}
    <line x1="${at(origin).toFixed(1)}" y1="4" x2="${at(origin).toFixed(1)}" y2="${H - 4}"
      stroke="var(--ink)" stroke-width="1.6"/>
  </svg>`;
}

// 쓰지 않은 축을 왜 제외했는지 보여주는 것이 이 도구의 고유한 부분이다.
function renderDetail(data) {
  const axes = data.reliability_axes;
  const state = axes["입력 상태 민감성"];
  const representation = axes["표현 안정성"];

  // 지문 모델 축끼리 눈금을 맞춘다. 언어 모델의 표현 안정성은 별도 눈금이다.
  const bValues = state.axes.flatMap((a) => (a.spread || []).map((v) => v.prediction));
  const bLo = Math.min(data.prediction, ...bValues), bHi = Math.max(data.prediction, ...bValues);
  const aSpread = representation.spread || [];
  const aOrigin = aSpread.length ? aSpread[0].prediction - aSpread[0].shift : 0;
  const aValues = aSpread.map((v) => v.prediction);
  const aLo = Math.min(aOrigin, ...aValues), aHi = Math.max(aOrigin, ...aValues);

  const row = (name, axis, origin, lo, hi, model) => {
    if (!axis.usable) {
      return `<tr class="off"><td>${name}</td><td colspan="4">제외 — ${axis.reason}</td></tr>`;
    }
    const hot = has(axis.percentile) && axis.percentile >= HIGH;
    const range = (axis.spread || []).length
      ? `${num(Math.min(origin, ...axis.spread.map((v) => v.prediction)), 3)} ~ ${num(Math.max(origin, ...axis.spread.map((v) => v.prediction)), 3)}`
      : "—";
    return `<tr${hot ? ' class="hot"' : ""}>
      <td>${name}<span class="model">${model}</span></td>
      <td>${axis.n_variants}종</td>
      <td class="strip-cell">${strip(origin, axis.spread, lo, hi)}<span class="range">${range}</span></td>
      <td>${pctText(axis.percentile)} 백분위</td>
    </tr>`;
  };

  const rows = [row("표현 안정성", representation, aOrigin, aLo, aHi, "언어 모델")]
    .concat(state.axes.map((a) => row(a.name, a, data.prediction, bLo, bHi, "지문 모델")))
    .join("");
  $("detail").innerHTML = `<table>
    <thead><tr><th>축</th><th>만든 변형</th><th>예측 분포 (세로선이 원본)</th><th>순위</th></tr></thead>
    <tbody>${rows}</tbody></table>`;
}

function renderMeta(data) {
  const risk = data.combined_risk;
  $("meta-body").innerHTML = `<table><tbody>
    <tr><th>결합 위험 점수</th><td>${risk ? `${num(risk.score)} — ${risk.note}` : "산출하지 않음"}</td></tr>
    <tr><th>사용 신호</th><td class="mono">${risk ? risk.features.join(", ") : "—"}</td></tr>
    <tr><th>설정 지문</th><td class="mono">${data.settings_digest}</td></tr>
    <tr><th>산출물 생성</th><td>${BUILT_AT || "—"}</td></tr>
  </tbody></table>`;
}

function render(data) {
  const classification = data.task_type === "classification";
  $("structure").innerHTML = data.svg || "";
  $("pred-label").textContent = classification ? "양성 확률" : "예측값";
  $("pred-value").textContent = num(data.prediction, classification ? 3 : 2);
  $("canonical").textContent = data.canonical_smiles;

  const set = data.prediction_set;
  if (data.interval) {
    $("pred-band").textContent =
      `${num(data.interval.lower, 2)} ~ ${num(data.interval.upper, 2)} · ${Math.round(data.interval.coverage * 100)}% 컨포멀 구간`;
  } else if (set) {
    const labels = set.labels.map((l) => (l === 1 ? "양성" : "음성")).join(", ") || "없음";
    $("pred-band").textContent = `${Math.round(set.coverage * 100)}% 예측 집합 {${labels}}`
      + (set.size === 1 ? " · 한 라벨로 확신" : " · 가르지 못함");
  } else {
    $("pred-band").textContent = "";
  }

  const level = data.verdict.level;
  $("verdict").className = `verdict ${level}`;
  $("verdict-level").textContent = level;
  $("verdict-level").className = `badge ${level}`;
  $("verdict-headline").textContent = data.verdict.headline;
  const notes = data.verdict.notes.map((n) => `<li>${n}</li>`);
  // 등급 경계는 임의값이 아니라 시험 분자에서 잰 실제 미달률로 정했다.
  if (data.verdict.evidence) notes.push(`<li class="evidence">${data.verdict.evidence.text}</li>`);
  $("verdict-notes").innerHTML = notes.join("");

  renderAxes(data.reliability_axes);
  renderShifts(data);
  renderDetail(data);
  renderMeta(data);
  document.querySelectorAll("#result [data-enter]").forEach((node) => {
    node.style.setProperty("--i", node.dataset.enter);
  });
  $("result").hidden = false;
}

async function fetchResult() {
  if (MODE === "static") {
    const res = await fetch(`data/${$("molecule").value}.json`, { cache: "no-store" });
    if (!res.ok) throw new Error("미리 채점한 결과를 읽지 못했다");
    return res.json();
  }
  const res = await fetch("/api/score", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataset: $("dataset").value, smiles: $("smiles").value.trim() }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "채점에 실패했다");
  return data;
}

async function submit(event) {
  event.preventDefault();
  if (MODE === "api" && !$("smiles").value.trim()) return;
  $("submit").disabled = true;
  $("submit").classList.add("busy");
  $("landing").hidden = true;
  $("result").hidden = true;
  $("skeleton").hidden = false;
  setStatus(MODE === "static" ? "결과를 불러오는 중."
    : "변형을 만들고 두 모델로 채점하는 중. 물성을 처음 부르면 몇 초 걸린다.");
  try {
    const data = await fetchResult();
    setStatus("");
    $("skeleton").hidden = true;
    render(data);
  } catch (error) {
    $("skeleton").hidden = true;
    if (!document.querySelector("#result:not([hidden])")) $("landing").hidden = false;
    setStatus(error.message, true);
  } finally {
    $("submit").disabled = false;
    $("submit").classList.remove("busy");
  }
}

$("query").addEventListener("submit", submit);
document.querySelectorAll(".examples button").forEach((button) => {
  button.addEventListener("click", () => {
    $("dataset").value = button.dataset.dataset;
    if (MODE === "static") {
      fillMolecules();
      const match = CATALOGUE.find((m) => m.dataset === button.dataset.dataset
        && m.smiles === button.dataset.smiles);
      if (match) $("molecule").value = match.id;
    } else {
      $("smiles").value = button.dataset.smiles;
    }
    $("query").requestSubmit();
  });
});
boot();
