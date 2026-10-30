const $ = (sel) => document.querySelector(sel);
const EMOJI = { sadness: "😢", joy: "😄", love: "❤️", anger: "😠", fear: "😨", surprise: "😲", disgust: "🤢", neutral: "😐" };
const EMO_COLORS = {
  sadness: "#60a5fa", joy: "#facc15", love: "#f472b6", anger: "#ef4444",
  fear: "#a78bfa", surprise: "#22d3ee", disgust: "#84cc16", neutral: "#94a3b8",
};
const SENT_COLORS = { positive: "#22c55e", neutral: "#94a3b8", negative: "#ef4444" };

Chart.defaults.color = "#8d93b5";
Chart.defaults.borderColor = "#2a3052";
Chart.defaults.font.family = "Segoe UI, system-ui, sans-serif";

const charts = {};
function drawChart(id, config) {
  charts[id]?.destroy();
  charts[id] = new Chart(document.getElementById(id), config);
}

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x, d = 1) => `${(x * 100).toFixed(d)}%`;
const kpi = (k, v) => `<div class="kpi"><div class="v">${v}</div><div class="k">${k}</div></div>`;
const pill = (cls, text) => `<span class="pill ${cls}">${esc(text)}</span>`;

async function api(url, options) {
  const res = await fetch(url, options);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

/* ---------- tabs ---------- */
const loaded = {};
document.querySelectorAll(".tab").forEach((btn) =>
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b === btn));
    document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === btn.dataset.tab));
    const tab = btn.dataset.tab;
    if (!loaded[tab]) {
      loaded[tab] = true;
      ({ insights: loadInsights, explorer: initExplorer, model: loadModel }[tab] || (() => {}))();
    }
  })
);

/* ---------- analyze ---------- */
function bars(container, entries, fmt = (v) => pct(v)) {
  container.innerHTML = entries.map(([name, v, color]) => `
    <div class="bar"><span class="bar-name">${esc(name)}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${Math.max(0, v) * 100}%;${color ? `background:${color}` : ""}"></div></div>
      <span class="bar-val">${fmt(v)}</span></div>`).join("");
}

async function analyze() {
  const text = $("#inputText").value.trim();
  if (!text) return;
  $("#analyzeBtn").disabled = true;
  try {
    const r = await api("/api/analyze", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }),
    });
    const { emotion: e, sentiment: s, emotion_classic: c } = r;
    const ai = r.models === "transformer";
    $("#resultEmpty").classList.add("hidden");
    $("#resultBody").classList.remove("hidden");
    $("#resEmoji").textContent = e.emoji;
    $("#resEmotion").textContent = e.label;
    $("#resConf").textContent = `${pct(e.confidence)} confidence · ${ai ? "DistilRoBERTa" : "TF-IDF model"}`;
    const badge = $("#resSentiment");
    badge.className = `badge ${s.label}`;
    badge.textContent = s.label;
    const score = ai ? s.roberta.polarity : s.vader.compound;
    $("#resScore").textContent = score.toFixed(3);
    $("#resSource").textContent = ai ? "RoBERTa, positive − negative probability" : "VADER compound";
    $("#gaugeMarker").style.left = `${((score + 1) / 2) * 100}%`;

    const row = (name, label, detail) => `<div class="cmp-row"><span>${name}</span><span>${label}</span><span class="muted">${detail}</span></div>`;
    $("#compareRows").innerHTML = [
      ai && row("Sentiment · RoBERTa", pill(s.roberta.label, s.roberta.label), pct(s.roberta.confidence)),
      row("Sentiment · VADER", pill(s.vader.label, s.vader.label), `compound ${s.vader.compound.toFixed(3)}`),
      row("Sentiment · AFINN", pill(s.afinn.label, s.afinn.label), `score ${s.afinn.score}`),
      ai && row("Emotion · DistilRoBERTa", `${EMOJI[e.label]} ${e.label}`, pct(e.confidence)),
      row("Emotion · TF-IDF (classic)", `${EMOJI[c.label]} ${c.label}`, pct(c.confidence)),
    ].filter(Boolean).join("");

    bars($("#emotionBars"), Object.entries(e.scores).map(([k, v]) => [`${EMOJI[k]} ${k}`, v, EMO_COLORS[k]]));
    const sp = ai ? s.roberta.scores : { positive: s.vader.pos, neutral: s.vader.neu, negative: s.vader.neg };
    $("#sentBarsTitle").textContent = ai ? "RoBERTa sentiment probabilities" : "VADER breakdown";
    bars($("#sentBars"), ["positive", "neutral", "negative"].map((k) => [k, sp[k], SENT_COLORS[k]]));
  } catch (err) {
    alert(err.message);
  } finally {
    $("#analyzeBtn").disabled = false;
  }
}
$("#analyzeBtn").addEventListener("click", analyze);
$("#inputText").addEventListener("keydown", (ev) => { if (ev.key === "Enter" && ev.ctrlKey) analyze(); });
document.querySelectorAll(".chip").forEach((c) =>
  c.addEventListener("click", () => { $("#inputText").value = c.dataset.example; analyze(); }));

/* ---------- batch ---------- */
let batchRows = [];

function renderBatchTable() {
  const f = $("#batchFilter").value;
  const rows = batchRows.filter((r) => !f || r.sentiment === f).slice(0, 1000);
  $("#batchTable tbody").innerHTML = rows.map((r) => `
    <tr><td class="text">${esc(r.text)}</td><td>${pill(r.sentiment, r.sentiment)}</td>
    <td>${r.score.toFixed(3)}</td><td>${pill(r.vader, r.vader)}</td>
    <td>${EMOJI[r.emotion]} ${esc(r.emotion)}</td><td>${pct(r.confidence)}</td></tr>`).join("");
}

async function runBatch() {
  const file = $("#csvFile").files[0];
  const lines = $("#batchText").value.split("\n").map((l) => l.trim()).filter(Boolean);
  if (!file && !lines.length) { $("#batchStatus").textContent = "Choose a CSV file or paste some lines."; return; }

  $("#batchBtn").disabled = true;
  const fast = $("#fastMode").checked;
  $("#batchStatus").textContent = fast ? "Analyzing…" : "Analyzing with transformer models (about 1 minute per 400 rows on CPU)…";
  try {
    let data;
    if (file) {
      const fd = new FormData();
      fd.append("file", file);
      if ($("#csvColumn").value.trim()) fd.append("column", $("#csvColumn").value.trim());
      if (fast) fd.append("fast", "1");
      data = await api("/api/batch", { method: "POST", body: fd });
    } else {
      data = await api("/api/batch", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ texts: lines, fast }),
      });
    }
    batchRows = data.rows;
    const s = data.summary;
    const models = data.models === "transformer" ? "transformer models" : "lexicon + classic model";
    $("#batchStatus").textContent = `Analyzed ${s.count} rows from column "${data.column}" with ${models}` +
      `${data.truncated ? ` (first ${data.limit.toLocaleString()} rows only)` : ""}.`;
    $("#batchKpis").innerHTML = [
      kpi("Rows", s.count),
      kpi("Avg score", s.avg_score.toFixed(3)),
      kpi("Positive", pct(s.sentiment.positive / s.count)),
      kpi("Negative", pct(s.sentiment.negative / s.count)),
    ].join("");
    drawChart("batchSentChart", doughnut(s.sentiment, SENT_COLORS));
    drawChart("batchEmoChart", doughnut(s.emotion, EMO_COLORS));
    renderBatchTable();
    $("#downloadBtn").disabled = false;
  } catch (err) {
    $("#batchStatus").textContent = `Error: ${err.message}`;
  } finally {
    $("#batchBtn").disabled = false;
  }
}

function downloadCsv() {
  const q = (v) => `"${String(v).replace(/"/g, '""')}"`;
  const header = ["text", "sentiment", "score", "vader", "compound", "afinn", "emotion", "confidence"];
  const csv = [header.join(","), ...batchRows.map((r) => header.map((h) => q(r[h])).join(","))].join("\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
  a.download = "sentiment_results.csv";
  a.click();
}
$("#batchBtn").addEventListener("click", runBatch);
$("#downloadBtn").addEventListener("click", downloadCsv);
$("#batchFilter").addEventListener("change", renderBatchTable);

function doughnut(counts, colors) {
  const labels = Object.keys(counts);
  return {
    type: "doughnut",
    data: { labels, datasets: [{ data: labels.map((l) => counts[l]), backgroundColor: labels.map((l) => colors[l]), borderWidth: 0 }] },
    options: { plugins: { legend: { position: "bottom" } }, cutout: "60%" },
  };
}

/* ---------- insights ---------- */
async function loadInsights() {
  const d = await api("/api/insights");
  const sc = d.sentiment_counts;
  $("#insightKpis").innerHTML = [
    kpi("Reviews analyzed", d.total_reviews.toLocaleString()),
    kpi("Apps compared", d.apps),
    kpi("Positive reviews", pct(sc.positive / d.total_reviews)),
    kpi("Negative reviews", pct(sc.negative / d.total_reviews)),
    kpi(`Avg ${d.sentiment_source} score`, d.avg_score.toFixed(3)),
    kpi(`${d.sentiment_source} ↔ VADER agreement`, pct(d.vader_agreement)),
  ].join("");

  const ordered = { positive: sc.positive, neutral: sc.neutral, negative: sc.negative };
  drawChart("sentPie", doughnut(ordered, SENT_COLORS));

  const emos = d.emotion_labels;
  drawChart("emoBar", {
    type: "bar",
    data: { labels: emos.map((e) => `${EMOJI[e]} ${e}`), datasets: [{ data: emos.map((e) => d.emotion_counts[e]), backgroundColor: emos.map((e) => EMO_COLORS[e]), borderRadius: 6 }] },
    options: { plugins: { legend: { display: false } } },
  });

  const apps = d.per_app;
  drawChart("appChart", {
    type: "bar",
    data: {
      labels: apps.map((a) => a.app),
      datasets: ["positive", "neutral", "negative"].map((k) => ({
        label: k, data: apps.map((a) => +(a[k] / a.reviews * 100).toFixed(1)), backgroundColor: SENT_COLORS[k],
      })),
    },
    options: {
      indexAxis: "y",
      scales: { x: { stacked: true, max: 100, ticks: { callback: (v) => v + "%" } }, y: { stacked: true } },
      plugins: { tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${c.raw}%` } } },
    },
  });

  drawChart("trendChart", {
    type: "line",
    data: {
      labels: d.trend.map((t) => t.month),
      datasets: [
        { label: "Avg score", data: d.trend.map((t) => t.avg_score), borderColor: "#7c5cff", tension: .3, yAxisID: "y" },
        { label: "% negative", data: d.trend.map((t) => +(t.negative_share * 100).toFixed(1)), borderColor: "#ef4444", tension: .3, yAxisID: "y1" },
      ],
    },
    options: { scales: { y: { position: "left" }, y1: { position: "right", grid: { drawOnChartArea: false }, ticks: { callback: (v) => v + "%" } } } },
  });

  const binLabels = d.score_histogram.map((_, i) => (-1 + i * 0.1).toFixed(1));
  drawChart("histChart", {
    type: "bar",
    data: {
      labels: binLabels,
      datasets: [{ data: d.score_histogram, backgroundColor: binLabels.map((b) => (+b < -0.05 ? "#ef4444" : +b >= 0.05 ? "#22c55e" : "#94a3b8")), borderRadius: 3 }],
    },
    options: { plugins: { legend: { display: false } } },
  });

  const cloud = (el, words) => {
    const max = words[0]?.count || 1;
    el.innerHTML = words.map((w) => `<span style="font-size:${12 + (w.count / max) * 22}px" title="${w.count}">${esc(w.word)}</span>`).join("");
  };
  cloud($("#posWords"), d.top_words.positive);
  cloud($("#negWords"), d.top_words.negative);

  $("#appTable tbody").innerHTML = apps.map((a) => {
    const topEmo = Object.entries(a.emotions).sort((x, y) => y[1] - x[1])[0][0];
    return `<tr><td><b>${esc(a.app)}</b></td><td>${a.reviews}</td><td>${a.avg_score.toFixed(3)}</td>
      <td>${pct(a.positive / a.reviews)}</td><td>${pct(a.negative / a.reviews)}</td><td>${pct(a.reply_rate)}</td>
      <td>${EMOJI[topEmo]} ${topEmo}</td></tr>`;
  }).join("");
}

/* ---------- explorer ---------- */
let page = 1;

async function initExplorer() {
  const d = await api("/api/insights");
  $("#fApp").innerHTML += d.per_app.map((a) => a.app).sort().map((a) => `<option>${esc(a)}</option>`).join("");
  $("#fEmo").innerHTML += d.emotion_labels.map((e) => `<option value="${e}">${EMOJI[e]} ${e}</option>`).join("");
  loadReviews();
}

async function loadReviews() {
  const params = new URLSearchParams({
    app: $("#fApp").value, sentiment: $("#fSent").value, emotion: $("#fEmo").value,
    sort: $("#fSort").value, q: $("#fQuery").value, page, size: 15,
  });
  const d = await api(`/api/reviews?${params}`);
  const pages = Math.max(1, Math.ceil(d.total / d.size));
  $("#reviewCount").textContent = `${d.total.toLocaleString()} matching reviews`;
  $("#pageInfo").textContent = `Page ${d.page} of ${pages}`;
  $("#prevPage").disabled = page <= 1;
  $("#nextPage").disabled = page >= pages;
  $("#reviewList").innerHTML = d.items.map((r) => `
    <div class="review">
      <div class="review-meta"><b>${esc(r.app)}</b> · ${esc(r.userName)} · ${esc(r.at)} · 👍 ${r.thumbsUpCount}
        ${pill(r.sentiment, `${r.sentiment} ${(+r.score).toFixed(2)}`)} ${pill("neutral", `${EMOJI[r.emotion]} ${r.emotion}`)}</div>
      <div>${esc(r.content)}</div>
      ${r.replyContent ? `<div class="reply"><b>Developer reply:</b> ${esc(r.replyContent)}</div>` : ""}
    </div>`).join("") || `<div class="empty">No reviews match these filters.</div>`;
}

const search = () => { page = 1; loadReviews(); };
$("#fBtn").addEventListener("click", search);
$("#fQuery").addEventListener("keydown", (e) => { if (e.key === "Enter") search(); });
["#fApp", "#fSent", "#fEmo", "#fSort"].forEach((s) => $(s).addEventListener("change", search));
$("#prevPage").addEventListener("click", () => { page--; loadReviews(); });
$("#nextPage").addEventListener("click", () => { page++; loadReviews(); });

/* ---------- model ---------- */
function renderComparison(cmp) {
  const sc = cmp.spotcheck;
  const dom = cmp.in_domain;
  $("#comparisonSection").classList.remove("hidden");
  $("#comparisonNote").textContent =
    `Spot check: ${sc.size} hand-labelled app-review-style sentences (emotion accuracy uses the ${sc.emotion_size} ` +
    `sentences labelled with an emotion both models know). Original test set: ${dom.size.toLocaleString()} sentences from ` +
    `Emotions_training.csv with those same 5 emotions, the data the classic model was trained on.`;

  const pctAxis = { min: 0, max: 1, ticks: { callback: (v) => `${Math.round(v * 100)}%` } };
  const pctTip = { callbacks: { label: (c) => `${c.dataset.label}: ${pct(c.raw)}` } };
  drawChart("cmpEmotionChart", {
    type: "bar",
    data: {
      labels: ["Review-style spot check", "Original test set"],
      datasets: [
        { label: "Transformer (DistilRoBERTa)", data: [sc.emotion_accuracy.transformer, dom.transformer], backgroundColor: "#7c5cff" },
        { label: "Classic (TF-IDF)", data: [sc.emotion_accuracy.classic, dom.classic], backgroundColor: "#94a3b8" },
      ],
    },
    options: { scales: { y: pctAxis }, plugins: { title: { display: true, text: "Emotion accuracy" }, tooltip: pctTip } },
  });
  const sa = sc.sentiment_accuracy;
  drawChart("cmpSentimentChart", {
    type: "bar",
    data: {
      labels: ["RoBERTa", "VADER", "AFINN"],
      datasets: [{ label: "Accuracy", data: [sa.roberta, sa.vader, sa.afinn], backgroundColor: ["#7c5cff", "#94a3b8", "#64748b"] }],
    },
    options: {
      scales: { y: pctAxis },
      plugins: { legend: { display: false }, title: { display: true, text: "Sentiment accuracy (spot check)" }, tooltip: pctTip },
    },
  });

  const mark = (pred, truth) => `<td class="${pred === truth ? "ok" : "bad"}">${EMOJI[pred] || ""} ${esc(pred)}</td>`;
  $("#spotTable tbody").innerHTML = sc.examples.map((r) => `
    <tr><td class="text">${esc(r.text)}</td><td>${EMOJI[r.emotion]} ${esc(r.emotion)}</td>
    ${mark(r.ai_emotion, r.emotion)}${mark(r.classic_emotion, r.emotion)}
    <td>${esc(r.sentiment)}</td>${mark(r.roberta, r.sentiment)}${mark(r.vader, r.sentiment)}</tr>`).join("");
}

async function loadModel() {
  const m = await api("/api/metrics");
  if (m.comparison) renderComparison(m.comparison);
  const base = m.baseline_notebook;
  $("#modelKpis").innerHTML = [
    kpi("Test accuracy", pct(m.accuracy.test)),
    kpi("Validation accuracy", pct(m.accuracy.validation)),
    kpi("Macro F1", pct(m.macro_f1)),
    kpi("Notebook baseline (test)", pct(base.test)),
    kpi("Training samples", m.split_sizes.train.toLocaleString()),
  ].join("");

  const labels = m.labels;
  drawChart("f1Chart", {
    type: "bar",
    data: {
      labels: labels.map((l) => `${EMOJI[l]} ${l}`),
      datasets: [
        { label: "Precision", data: labels.map((l) => m.report[l].precision), backgroundColor: "#7c5cff" },
        { label: "Recall", data: labels.map((l) => m.report[l].recall), backgroundColor: "#22d3ee" },
        { label: "F1", data: labels.map((l) => m.report[l]["f1-score"]), backgroundColor: "#22c55e" },
      ],
    },
    options: { scales: { y: { min: 0, max: 1 } } },
  });
  drawChart("distChart", doughnut(m.distribution, EMO_COLORS));

  const cm = m.confusion_matrix;
  const n = labels.length;
  let html = `<div class="cm" style="grid-template-columns:90px repeat(${n},1fr)"><div class="h">true ↓ / pred →</div>`;
  html += labels.map((l) => `<div class="h">${l}</div>`).join("");
  cm.forEach((row, i) => {
    const total = row.reduce((a, b) => a + b, 0);
    html += `<div class="h">${labels[i]}</div>`;
    html += row.map((v, j) => {
      const a = total ? v / total : 0;
      const color = i === j ? `rgba(34,197,94,${0.15 + a * 0.85})` : `rgba(239,68,68,${a * 1.5})`;
      return `<div style="background:${color}" title="${pct(a)} of true ${labels[i]}">${v}</div>`;
    }).join("");
  });
  $("#cm").innerHTML = html + "</div>";

  $("#topWords").innerHTML = labels.map((l) =>
    `<div class="tw"><b style="color:${EMO_COLORS[l]}">${EMOJI[l]} ${l}</b> ${m.top_words[l].map(esc).join(", ")}</div>`).join("");
}
