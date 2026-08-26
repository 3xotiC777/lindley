import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v314.0.5/full/pyodide.mjs";

const PYODIDE_INDEX_URL = "https://cdn.jsdelivr.net/pyodide/v314.0.5/full/";
const EXCEL_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet";

const engineSources = [
  { name: "config.py", published: "./python/config.py", development: "../config.py" },
  { name: "ingest.py", published: "./python/ingest.py", development: "../ingest.py" },
  { name: "selector_engine.py", published: "./python/selector_engine.py", development: "../selector_engine.py" },
  { name: "exporter.py", published: "./python/exporter.py", development: "../exporter.py" },
  { name: "browser_runner.py", published: "./python/browser_runner.py" },
];

let runtimePromise;
let runtimeReady = false;
let activeRun = false;
let activeRunId = 0;
let currentProgress = 0;

function postProgress(title, detail, percent) {
  currentProgress = Math.max(currentProgress, Number(percent) || 0);
  self.postMessage({
    type: "progress",
    runId: activeRunId,
    title,
    detail,
    percent: currentProgress,
  });
}

async function fetchSource(source) {
  const candidates = [source.published, source.development].filter(Boolean);
  let lastError;
  for (const relativePath of candidates) {
    const url = new URL(relativePath, self.location.href);
    try {
      const response = await fetch(url, { cache: "no-cache" });
      if (!response.ok) {
        lastError = new Error(`${response.status} al cargar ${url.pathname}`);
        continue;
      }
      return await response.text();
    } catch (error) {
      lastError = error;
    }
  }
  throw new Error(`No se pudo cargar ${source.name}: ${lastError?.message || "archivo no disponible"}`);
}

function ensureDirectory(fs, path) {
  try {
    fs.mkdirTree(path);
  } catch (error) {
    if (!/exist/i.test(String(error))) throw error;
  }
}

function removeTree(fs, path) {
  let entries;
  try {
    entries = fs.readdir(path);
  } catch {
    return;
  }

  for (const entry of entries) {
    if (entry === "." || entry === "..") continue;
    const child = `${path}/${entry}`;
    try {
      const mode = fs.stat(child).mode;
      if (fs.isDir(mode)) {
        removeTree(fs, child);
        fs.rmdir(child);
      } else {
        fs.unlink(child);
      }
    } catch {
      // Cleanup is best-effort; a page refresh also clears the in-memory FS.
    }
  }
}

function translateEngineLog(line) {
  const text = String(line || "").trim();
  if (!text) return;
  if (/Loading .*preselection/i.test(text)) {
    postProgress("Leyendo la preselección…", "Validando la base y separando puntos elegibles.", 47);
  } else if (/Loading .*previous selection/i.test(text)) {
    postProgress("Leyendo la selección anterior…", "Calculando las cuotas históricas por ciudad y canal.", 53);
  } else if (/Running selection rules/i.test(text)) {
    postProgress("Aplicando las reglas…", "Distribuyendo titulares y suplentes por ruta.", 65);
  } else if (/Selection complete/i.test(text)) {
    postProgress("Validando el resultado…", "Revisando cuotas, rutas y puntos obligatorios.", 82);
  } else if (/Exporting main selection workbook/i.test(text)) {
    postProgress("Creando el archivo principal…", "Preparando las hojas BD y CONTROL CUOTAS.", 88);
  } else if (/Exporting ARCHIVO_DEF_SUP/i.test(text)) {
    postProgress("Creando el archivo DEF / SUP…", "Finalizando los entregables de Excel.", 95);
  }
}

async function initializeRuntime() {
  postProgress(
    "Cargando el motor de selección…",
    "La primera ejecución descarga Python y las librerías de Excel.",
    12,
  );

  const pyodide = await loadPyodide({ indexURL: PYODIDE_INDEX_URL });
  pyodide.setStdout({ batched: translateEngineLog });

  postProgress("Preparando las librerías…", "Cargando cálculo tabular y lectura de archivos.", 23);
  await pyodide.loadPackage(["numpy", "pandas", "micropip"]);

  const micropip = pyodide.pyimport("micropip");
  try {
    await micropip.install(["openpyxl==3.1.5", "pyxlsb==1.0.10"]);
  } finally {
    micropip.destroy();
  }

  postProgress("Cargando las reglas Lindley…", "Usando el mismo motor Python de la aplicación de escritorio.", 34);
  const sourcePairs = await Promise.all(
    engineSources.map(async (source) => [source.name, await fetchSource(source)]),
  );
  ensureDirectory(pyodide.FS, "/app");
  for (const [name, contents] of sourcePairs) {
    pyodide.FS.writeFile(`/app/${name}`, contents, { encoding: "utf8" });
  }
  pyodide.runPython(`
import importlib
import sys
if "/app" not in sys.path:
    sys.path.insert(0, "/app")
importlib.invalidate_caches()
`);

  runtimeReady = true;
  return pyodide;
}

function getRuntime() {
  if (!runtimePromise) runtimePromise = initializeRuntime();
  return runtimePromise;
}

function extensionForPrevious(fileName) {
  return String(fileName || "").toLowerCase().endsWith(".xlsb") ? ".xlsb" : ".xlsx";
}

function concisePythonError(details) {
  const lines = String(details || "")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  for (let index = lines.length - 1; index >= 0; index -= 1) {
    const match = lines[index].match(/(?:ValueError|KeyError|FileNotFoundError|RuntimeError|TypeError):\s*(.+)$/);
    if (match?.[1]) return match[1].replace(/^['"]|['"]$/g, "");
  }
  return lines.at(-1) || "Ocurrió un error inesperado durante la selección.";
}

async function executeSelection(payload) {
  const pyodide = await getRuntime();
  const fs = pyodide.FS;
  const workRoot = "/work";
  const inputDir = `${workRoot}/input`;
  const outputDir = `${workRoot}/output`;
  ensureDirectory(fs, inputDir);
  ensureDirectory(fs, outputDir);
  removeTree(fs, inputDir);
  removeTree(fs, outputDir);

  try {
  postProgress("Copiando los archivos…", "Los datos permanecen en la memoria de este navegador.", 40);
  const previousExtension = extensionForPrevious(payload.inputs.previous.name);
  const inputPaths = {
    preselection: `${inputDir}/preselection.xlsx`,
    previous: `${inputDir}/previous${previousExtension}`,
    lima: `${inputDir}/lima.xlsx`,
  };

  fs.writeFile(inputPaths.preselection, new Uint8Array(payload.inputs.preselection.buffer));
  fs.writeFile(inputPaths.previous, new Uint8Array(payload.inputs.previous.buffer));
  fs.writeFile(inputPaths.lima, new Uint8Array(payload.inputs.lima.buffer));

  const parameters = JSON.stringify({
    preselectionPath: inputPaths.preselection,
    previousPath: inputPaths.previous,
    limaPath: inputPaths.lima,
    outputDir,
    totalTarget: Number(payload.sampleSize),
    increaseScope: String(payload.increaseScope || "OFF"),
  });

  pyodide.globals.set("browser_job_parameters", parameters);
  let resultJson;
  try {
    resultJson = await pyodide.runPythonAsync(`
import json
from browser_runner import run_browser_selection

_browser_job = json.loads(browser_job_parameters)
json.dumps(
    run_browser_selection(
        preselection_path=_browser_job["preselectionPath"],
        previous_path=_browser_job["previousPath"],
        lima_path=_browser_job["limaPath"],
        output_dir=_browser_job["outputDir"],
        total_target=_browser_job["totalTarget"],
        increase_scope=_browser_job["increaseScope"],
    ),
    ensure_ascii=False,
)
`);
  } finally {
    pyodide.globals.delete("browser_job_parameters");
  }

  const summary = JSON.parse(String(resultJson));
  postProgress("Preparando las descargas…", "Copiando los resultados desde el motor local.", 98);

  const files = [
    {
      key: "selection",
      name: summary.selectionFile,
      mime: EXCEL_MIME,
      buffer: fs.readFile(`${outputDir}/${summary.selectionFile}`, { encoding: "binary" }).slice().buffer,
    },
    {
      key: "supervision",
      name: summary.supervisionFile,
      mime: EXCEL_MIME,
      buffer: fs.readFile(`${outputDir}/${summary.supervisionFile}`, { encoding: "binary" }).slice().buffer,
    },
  ];

  postProgress("Listo", "Los dos archivos ya están disponibles para descargar.", 100);
  return { summary, files };
  } finally {
    removeTree(fs, inputDir);
    removeTree(fs, outputDir);
  }
}

self.addEventListener("message", async (event) => {
  const message = event.data || {};
  if (message.type !== "run") return;

  if (activeRun) {
    self.postMessage({
      type: "error",
      runId: message.runId,
      message: "Ya hay una selección en curso.",
      details: "Espera a que termine el proceso actual antes de iniciar otro.",
      fatal: false,
    });
    return;
  }

  activeRun = true;
  activeRunId = Number(message.runId) || Date.now();
  currentProgress = 0;
  try {
    const { summary, files } = await executeSelection(message.payload);
    const transferables = files.map((file) => file.buffer);
    self.postMessage({ type: "result", runId: activeRunId, summary, files }, transferables);
  } catch (error) {
    const details = error?.stack || String(error);
    const memoryFailure = /memory|out of bounds|allocation|heap/i.test(details);
    const fatal = !runtimeReady || memoryFailure;
    if (fatal) {
      runtimePromise = undefined;
      runtimeReady = false;
    }
    self.postMessage({
      type: "error",
      runId: activeRunId,
      message: concisePythonError(details),
      details,
      fatal,
    });
  } finally {
    activeRun = false;
  }
});
