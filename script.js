const canvas = document.getElementById("latticeCanvas");
const ctx = canvas.getContext("2d");

let width = 0;
let height = 0;
let nodes = [];

function resizeCanvas() {
  const ratio = window.devicePixelRatio || 1;
  width = canvas.clientWidth;
  height = canvas.clientHeight;
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  createNodes();
}

function createNodes() {
  const spacing = Math.max(70, Math.min(128, width / 10));
  nodes = [];

  for (let y = -spacing; y < height + spacing; y += spacing) {
    for (let x = width * 0.36; x < width + spacing; x += spacing) {
      const offset = Math.round(y / spacing) % 2 === 0 ? spacing * 0.42 : 0;
      nodes.push({
        x: x + offset,
        y,
        baseX: x + offset,
        baseY: y,
        phase: Math.random() * Math.PI * 2,
        radius: 2.5 + Math.random() * 3.5,
      });
    }
  }
}

function draw(time) {
  ctx.clearRect(0, 0, width, height);
  ctx.lineWidth = 1;

  const t = time * 0.001;
  nodes.forEach((node) => {
    node.x = node.baseX + Math.sin(t + node.phase) * 10;
    node.y = node.baseY + Math.cos(t * 0.8 + node.phase) * 8;
  });

  for (let i = 0; i < nodes.length; i += 1) {
    for (let j = i + 1; j < nodes.length; j += 1) {
      const a = nodes[i];
      const b = nodes[j];
      const dx = a.x - b.x;
      const dy = a.y - b.y;
      const distance = Math.sqrt(dx * dx + dy * dy);

      if (distance < 145) {
        const alpha = 1 - distance / 145;
        ctx.strokeStyle = `rgba(31, 122, 90, ${alpha * 0.28})`;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }
    }
  }

  nodes.forEach((node, index) => {
    const palette = ["#1f7a5a", "#147484", "#b47b24", "#a94b42"];
    ctx.fillStyle = palette[index % palette.length];
    ctx.beginPath();
    ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = "rgba(255, 255, 255, 0.75)";
    ctx.beginPath();
    ctx.arc(node.x - node.radius * 0.28, node.y - node.radius * 0.28, node.radius * 0.35, 0, Math.PI * 2);
    ctx.fill();
  });

  requestAnimationFrame(draw);
}

window.addEventListener("resize", resizeCanvas);
resizeCanvas();
requestAnimationFrame(draw);

const statusEl = document.getElementById("resultStatus");
const metaEl = document.getElementById("resultMeta");
const gridEl = document.getElementById("resultGrid");

function formatValue(value) {
  if (value === null || value === undefined) return "N/A";
  if (typeof value === "boolean") return value ? "True" : "False";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(4);
  return String(value);
}

function renderCandidateTable(candidates) {
  if (!candidates || candidates.length === 0) {
    return "<p>暂无候选结果。</p>";
  }
  const columns = Object.keys(candidates[0]);
  const head = columns.map((column) => `<th>${column}</th>`).join("");
  const rows = candidates
    .map((candidate) => {
      const cells = columns.map((column) => `<td>${formatValue(candidate[column])}</td>`).join("");
      return `<tr>${cells}</tr>`;
    })
    .join("");
  return `<table class="candidate-table"><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table>`;
}

function renderResults(payload) {
  statusEl.textContent = `结果生成时间：${payload.generated_at}`;
  const env = payload.environment || {};
  const summary = payload.run_summary || {};
  metaEl.innerHTML = [
    `Python ${env.python}`,
    `Torch ${env.torch}`,
    `CUDA ${env.cuda}`,
    `GPU ${env.cuda_available ? "可用" : "不可用"}`,
    `Seed ${summary.seed ?? "N/A"}`,
  ]
    .map((item) => `<span>${item}</span>`)
    .join("");

  gridEl.innerHTML = payload.models
    .map((model) => {
      const metrics = Object.entries(model.metrics || {})
        .map(([name, value]) => `<div><strong>${formatValue(value)}</strong><span>${name}</span></div>`)
        .join("");
      const details = model.dataset_details || {};
      const visual = model.visualization
        ? `<img class="model-visual" src="${model.visualization}" alt="${model.title} 可视化结果" loading="lazy" />`
        : "";
      return `
        <article class="result-card">
          <h3>${model.title}</h3>
          <p>${model.method}</p>
          <div class="dataset-box">
            <strong>数据集</strong>
            <span>${model.dataset}</span>
            <span>样本数：${formatValue(details.records_used)}</span>
            <span>目标变量：${details.target || "N/A"}</span>
            <span>划分方式：${details.split || "N/A"}</span>
            <span>说明：${details.note || "N/A"}</span>
          </div>
          <div class="metric-row">${metrics}</div>
          ${visual}
          ${renderCandidateTable(model.candidates)}
        </article>
      `;
    })
    .join("");
}

fetch("results/model_results.json", { cache: "no-store" })
  .then((response) => {
    if (!response.ok) {
      throw new Error("result file not found");
    }
    return response.json();
  })
  .then(renderResults)
  .catch(() => {
    statusEl.textContent = "还没有找到 results/model_results.json，请先运行 models/run_pipeline.py。";
    metaEl.innerHTML = "";
    gridEl.innerHTML = "";
  });
