const form = document.querySelector("#selector-form");
const workspace = document.querySelector(".workspace");
const runButton = document.querySelector("#run-button");
const formError = document.querySelector("#form-error");
const progressPanel = document.querySelector("#progress-panel");
const progressTitle = document.querySelector("#progress-title");
const progressDetail = document.querySelector("#progress-detail");
const progressPercent = document.querySelector("#progress-percent");
const progressBar = document.querySelector("#progress-bar");
const resultPanel = document.querySelector("#result-panel");
const resultSummary = document.querySelector("#result-summary");
const quotaWarning = document.createElement("div");
const quotaWarningTitle = document.createElement("strong");
const quotaWarningText = document.createElement("p");
quotaWarning.className = "quota-warning";
quotaWarning.setAttribute("role", "alert");
quotaWarningTitle.textContent = "Cuota CDA incompleta · descarga disponible";
quotaWarning.append(quotaWarningTitle, quotaWarningText);
quotaWarning.hidden = true;
resultSummary.after(quotaWarning);
const errorPanel = document.querySelector("#error-panel");
const errorMessage = document.querySelector("#error-message");
const errorDetails = document.querySelector("#error-details");
const errorDetailsWrapper = document.querySelector("#error-details-wrapper");
const sampleSizeInput = document.querySelector("#sample-size");
const manualSettings = document.querySelector("#manual-settings");
const cdaModeNote = document.querySelector("#cda-mode-note");
const runAgainButton = document.querySelector("#run-again");
const retryButton = document.querySelector("#retry-button");

const inputs = {
  preselection: document.querySelector("#preselection-file"),
  previous: document.querySelector("#previous-file"),
  lima: document.querySelector("#lima-file"),
  cda: document.querySelector("#cda-file"),
};

const downloads = {
  selection: document.querySelector("#download-selection"),
  supervision: document.querySelector("#download-supervision"),
};

const stats = {
  total: document.querySelector("#stat-total"),
  off: document.querySelector("#stat-off"),
  on: document.querySelector("#stat-on"),
};

const allowedExtensions = {
  preselection: [".xlsx"],
  previous: [".xlsx", ".xlsb"],
  lima: [".xlsx"],
  cda: [".xlsx"],
};

let worker;
let currentRunId = 0;
let isRunning = false;
let resultUrls = [];

const numberFormatter = new Intl.NumberFormat("es-PE");

function extensionOf(fileName) {
  const dot = fileName.lastIndexOf(".");
  return dot >= 0 ? fileName.slice(dot).toLowerCase() : "";
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 KB";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / 1024 ** index;
  return `${value.toLocaleString("es-PE", { maximumFractionDigits: index > 1 ? 1 : 0 })} ${units[index]}`;
}

function normalizedSampleSize() {
  const cleaned = sampleSizeInput.value.replace(/[\s.,]/g, "");
  if (!/^\d+$/.test(cleaned)) return null;
  const value = Number.parseInt(cleaned, 10);
  return Number.isSafeInteger(value) && value > 0 ? value : null;
}

function setFormError(message = "") {
  formError.textContent = message;
  formError.hidden = !message;
}

function clearFileErrors() {
  Object.entries(inputs).forEach(([key, input]) => {
    input.removeAttribute("aria-invalid");
    document.querySelector(`[data-file-card="${key}"]`).classList.remove("has-error");
  });
  sampleSizeInput.removeAttribute("aria-invalid");
}

function validateForm() {
  clearFileErrors();
  const issues = [];

  Object.entries(inputs).forEach(([key, input]) => {
    const file = input.files?.[0];
    const card = document.querySelector(`[data-file-card="${key}"]`);
    if (!file && key !== "cda") {
      issues.push("Selecciona los tres archivos de entrada.");
      card.classList.add("has-error");
      input.setAttribute("aria-invalid", "true");
      return;
    }
    if (!file) return;
    if (!allowedExtensions[key].includes(extensionOf(file.name))) {
      issues.push(`El archivo “${file.name}” no tiene el formato esperado.`);
      card.classList.add("has-error");
      input.setAttribute("aria-invalid", "true");
    } else if (file.size === 0) {
      issues.push(`El archivo “${file.name}” está vacío.`);
      card.classList.add("has-error");
      input.setAttribute("aria-invalid", "true");
    }
  });

  const useCda = Boolean(inputs.cda.files?.[0]);
  const sampleSize = useCda ? null : normalizedSampleSize();
  if (!useCda && sampleSize === null) {
    issues.push("Escribe la muestra total como un número entero positivo.");
    sampleSizeInput.setAttribute("aria-invalid", "true");
  }

  const uniqueIssues = [...new Set(issues)];
  setFormError(uniqueIssues.join(" "));
  return uniqueIssues.length === 0 ? { sampleSize, useCda } : null;
}

function setProgress({ title, detail, percent }) {
  const safePercent = Math.max(0, Math.min(100, Number(percent) || 0));
  if (title) progressTitle.textContent = title;
  if (detail) progressDetail.textContent = detail;
  progressPercent.textContent = `${Math.round(safePercent)}%`;
  progressBar.style.width = `${safePercent}%`;
}

function showOnly(panel) {
  workspace.hidden = panel !== workspace;
  progressPanel.hidden = panel !== progressPanel;
  resultPanel.hidden = panel !== resultPanel;
  errorPanel.hidden = panel !== errorPanel;
}

function scrollToPanel(panel) {
  panel.scrollIntoView({ behavior: "smooth", block: "start" });
}

function revokeDownloads() {
  resultUrls.forEach((url) => URL.revokeObjectURL(url));
  resultUrls = [];
  Object.values(downloads).forEach((link) => link.removeAttribute("href"));
}

function makeWorker() {
  if (worker) return worker;

  worker = new Worker(new URL("./worker.mjs", import.meta.url), {
    type: "module",
    name: "lindley-selection-worker",
  });
  worker.addEventListener("message", handleWorkerMessage);
  worker.addEventListener("error", (event) => {
    const detail = event.message || "El proceso en segundo plano se detuvo inesperadamente.";
    resetWorker();
    showFailure(
      "No fue posible iniciar el motor de selección. Verifica la conexión a internet e inténtalo de nuevo.",
      detail,
    );
  });
  worker.addEventListener("messageerror", () => {
    resetWorker();
    showFailure("El navegador no pudo leer el resultado del cálculo.", "Error al transferir datos desde el proceso de selección.");
  });
  return worker;
}

function resetWorker() {
  if (worker) {
    worker.terminate();
    worker = undefined;
  }
  isRunning = false;
}

function friendlyError(message) {
  const text = String(message || "").trim();
  if (!text) return "Ocurrió un error inesperado durante la selección.";
  if (/memory|out of bounds|allocation|heap/i.test(text)) {
    return "El navegador se quedó sin memoria. Cierra otras pestañas, vuelve a abrir esta página e inténtalo de nuevo.";
  }
  if (/failed to fetch|networkerror|load.*pyodide|package/i.test(text)) {
    return "No se pudo cargar el motor. Revisa la conexión a internet e inténtalo de nuevo.";
  }
  return text.split("\n")[0];
}

function handleWorkerMessage(event) {
  const message = event.data || {};
  if (message.runId && message.runId !== currentRunId) return;

  if (message.type === "progress") {
    setProgress(message);
    return;
  }

  if (message.type === "result") {
    isRunning = false;
    runButton.disabled = false;
    showResult(message.summary, message.files);
    return;
  }

  if (message.type === "error") {
    isRunning = false;
    runButton.disabled = false;
    if (message.fatal) resetWorker();
    showFailure(friendlyError(message.message), message.details || message.message);
  }
}

function showResult(summary, files) {
  revokeDownloads();

  for (const file of files || []) {
    const link = downloads[file.key];
    if (!link || !(file.buffer instanceof ArrayBuffer)) continue;
    const blob = new Blob([file.buffer], {
      type: file.mime || "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    });
    const url = URL.createObjectURL(blob);
    resultUrls.push(url);
    link.href = url;
    link.download = file.name;
  }

  const total = Number(summary?.actualTotal || 0);
  const requested = Number(summary?.requestedTotal || 0);
  const off = Number(summary?.actualOff || 0);
  const on = Number(summary?.actualOn || 0);
  const reviews = Number(summary?.controlsReview || 0);
  const unnamedMandatory = Number(summary?.mandatoryUnnamedTotal || 0);
  quotaWarningText.textContent = String(summary?.quotaWarning || "").trim();
  quotaWarning.hidden = !quotaWarningText.textContent;
  stats.total.textContent = numberFormatter.format(total);
  stats.off.textContent = numberFormatter.format(off);
  stats.on.textContent = numberFormatter.format(on);

  resultSummary.textContent = reviews
    ? `Se generaron los dos archivos con ${numberFormatter.format(total)} titulares. La hoja CONTROL CUOTAS conserva ${numberFormatter.format(reviews)} advertencias para revisar; no impidieron la descarga.`
    : `Se generaron los dos archivos con ${numberFormatter.format(total)} titulares y los controles terminaron sin novedades.`;
  if (unnamedMandatory) {
    resultSummary.textContent += ` ${numberFormatter.format(unnamedMandatory)} titulares Titán/Fénix sin NOMBRE cuentan dentro de las cuotas CDA y están señalados en CONTROL CUOTAS.`;
  }
  if (requested > total) {
    resultSummary.textContent += ` Faltan ${numberFormatter.format(requested - total)} titulares porque uno o más CDA no tienen capacidad seleccionable suficiente; revisa CONTROL CUOTAS.`;
  }

  showOnly(resultPanel);
  scrollToPanel(resultPanel);
}

function showFailure(message, details = "") {
  isRunning = false;
  runButton.disabled = false;
  errorMessage.textContent = message;
  errorDetails.textContent = String(details || "").trim();
  errorDetailsWrapper.hidden = !errorDetails.textContent;
  errorDetailsWrapper.open = false;
  showOnly(errorPanel);
  scrollToPanel(errorPanel);
}

async function startSelection({ sampleSize, useCda }) {
  if (isRunning) return;
  isRunning = true;
  currentRunId += 1;
  revokeDownloads();
  runButton.disabled = true;
  showOnly(progressPanel);
  setProgress({
    title: "Leyendo los archivos…",
    detail: "Preparando los datos para el cálculo local.",
    percent: 5,
  });
  scrollToPanel(progressPanel);

  try {
    const preselection = inputs.preselection.files[0];
    const previous = inputs.previous.files[0];
    const lima = inputs.lima.files[0];
    const cda = inputs.cda.files?.[0];
    const [preselectionBuffer, previousBuffer, limaBuffer, cdaBuffer] = await Promise.all([
      preselection.arrayBuffer(),
      previous.arrayBuffer(),
      lima.arrayBuffer(),
      cda?.arrayBuffer(),
    ]);

    const increaseScope = form.querySelector('input[name="increaseScope"]:checked')?.value || "OFF";
    const transferables = [preselectionBuffer, previousBuffer, limaBuffer];
    if (cdaBuffer) transferables.push(cdaBuffer);
    makeWorker().postMessage(
      {
        type: "run",
        runId: currentRunId,
        payload: {
          sampleSize,
          increaseScope,
          inputs: {
            preselection: { name: preselection.name, buffer: preselectionBuffer },
            previous: { name: previous.name, buffer: previousBuffer },
            lima: { name: lima.name, buffer: limaBuffer },
            cda: useCda ? { name: cda.name, buffer: cdaBuffer } : null,
          },
        },
      },
      transferables,
    );
  } catch (error) {
    showFailure("No fue posible leer uno de los archivos seleccionados.", error?.stack || String(error));
  }
}

Object.entries(inputs).forEach(([key, input]) => {
  input.addEventListener("change", () => {
    const card = document.querySelector(`[data-file-card="${key}"]`);
    const name = document.querySelector(`[data-file-name="${key}"]`);
    const file = input.files?.[0];
    card.classList.toggle("has-file", Boolean(file));
    card.classList.remove("has-error");
    input.removeAttribute("aria-invalid");
    name.textContent = file ? `${file.name} · ${formatBytes(file.size)}` : "Seleccionar archivo";
    if (key === "cda") {
      manualSettings.hidden = Boolean(file);
      cdaModeNote.hidden = !file;
      sampleSizeInput.required = !file;
    }
    setFormError();
  });
});

sampleSizeInput.addEventListener("blur", () => {
  const value = normalizedSampleSize();
  if (value !== null) sampleSizeInput.value = numberFormatter.format(value);
});

sampleSizeInput.addEventListener("focus", () => {
  const value = normalizedSampleSize();
  if (value !== null) sampleSizeInput.value = String(value);
});

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const settings = validateForm();
  if (settings !== null) startSelection(settings);
});

runAgainButton.addEventListener("click", () => {
  showOnly(workspace);
  scrollToPanel(workspace);
});

retryButton.addEventListener("click", () => {
  showOnly(workspace);
  scrollToPanel(workspace);
});

window.addEventListener("beforeunload", () => {
  revokeDownloads();
  if (worker) worker.terminate();
});
