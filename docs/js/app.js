/** Browser-only SMILES analysis, deterministic variants, and shareable state. */
import { analyze } from "./align.js";
import { loadModels, loadTokenizer } from "./models.js";
import { element, renderPanel } from "./render.js";

export const EXAMPLES = [
  { label: "Aspirin", smiles: "CC(=O)Oc1ccccc1C(=O)O" },
  { label: "Caffeine", smiles: "Cn1c(=O)c2c(ncn2C)n(C)c1=O" },
  { label: "Bromochlorophene", smiles: "Oc1c(Br)cc(Cl)cc1Cc1cc(Cl)cc(Br)c1O" },
  { label: "Indole", smiles: "c1ccc2[nH]ccc2c1" },
  { label: "D-Glucose (open chain)", smiles: "O=C[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO" },
  { label: "Cyclododecane", smiles: "C%10CCCCCCCCCCC%10" },
];
const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
let rdkit;
let models = [];
let selected = [];
let generation = 0;
let debounce;
let sketcher = null;
let sketcherMuted = false;
let variantSource = null;
const panels = new Map();
const loaded = new Map();
const themeQuery = matchMedia("(prefers-color-scheme: dark)");

function isDark() {
  return document.documentElement.dataset.theme ? document.documentElement.dataset.theme === "dark" : themeQuery.matches;
}
function setThemeLabel() {
  $("theme").textContent = isDark() ? "Light mode" : "Dark mode";
}
setThemeLabel();
$("theme").addEventListener("click", () => {
  document.documentElement.dataset.theme = isDark() ? "light" : "dark";
  setThemeLabel();
  if (rdkit) renderAll();
});
themeQuery.addEventListener("change", () => { setThemeLabel(); if (rdkit) renderAll(); });

if (params.has("s")) $("smiles").value = params.get("s");
if (params.has("p")) $("prefix").value = params.get("p");
$("special").checked = params.get("sp") !== "0";
const initialVariant = params.get("v") || "typed";
if (initialVariant === "canonical" || initialVariant.startsWith("rooted:")) {
  $("variant").value = initialVariant === "canonical" ? "canonical" : "rooted";
  const root = Number(initialVariant.split(":")[1]);
  $("root-atom").value = Number.isInteger(root) && root >= 0 ? String(root) : "0";
}
$("root-label").hidden = $("variant").value !== "rooted";

function syncURL() {
  const state = new URLSearchParams({
    s: $("variant").value !== "typed" && variantSource !== null ? variantSource : $("smiles").value,
    p: $("prefix").value, m: selected.join(","),
    sp: $("special").checked ? "1" : "0",
    v: $("variant").value === "rooted" ? `rooted:${$("root-atom").value}` : $("variant").value,
  });
  history.replaceState(null, "", `${location.pathname}?${state}${location.hash}`);
}
function inputError(message = "") {
  $("input-error").hidden = !message;
  $("input-error").textContent = message;
}
function updateExamples() {
  [...$("examples").children].forEach((button, i) => button.setAttribute("aria-pressed", String(EXAMPLES[i].smiles === $("smiles").value)));
}
function editSmiles(value) {
  if (value !== undefined) $("smiles").value = value;
  $("variant").value = "typed";
  $("root-label").hidden = true;
  variantSource = null;
  inputError();
  updateExamples();
  scheduleRender();
}
for (const example of EXAMPLES) {
  const button = element("button", "", example.label);
  button.type = "button";
  button.addEventListener("click", () => editSmiles(example.smiles));
  $("examples").append(button);
}
updateExamples();

function applyVariant(schedule = true) {
  const variant = $("variant").value;
  $("root-label").hidden = variant !== "rooted";
  if (variant === "typed") { variantSource = null; if (schedule) scheduleRender(); return; }
  if (!rdkit) return;
  variantSource ??= $("smiles").value;
  let mol;
  try {
    mol = rdkit.get_mol(variantSource, JSON.stringify({ removeHs: false }));
    if (!mol?.is_valid()) throw new Error("Enter a valid SMILES before choosing a variant.");
    const n = JSON.parse(mol.get_json()).molecules[0].atoms.length;
    $("root-atom").max = String(Math.max(0, n - 1));
    const k = Number($("root-atom").value);
    if (variant === "rooted" && (!Number.isInteger(k) || k < 0 || k >= n)) throw new Error(`Choose an atom index from 0 to ${n - 1}.`);
    $("smiles").value = variant === "canonical" ? mol.get_smiles() : mol.get_smiles(JSON.stringify({ rootedAtAtom: k, canonical: false }));
    inputError();
    updateExamples();
    if (schedule) scheduleRender();
  } catch (error) { inputError(error.message); }
  finally { mol?.delete(); }
}

function makePanel(model) {
  const panel = element("article", "panel");
  panel.dataset.model = model.id;
  const header = element("div", "panel-top");
  const title = element("div");
  title.append(element("h3", "", model.label));
  const source = element("a", "panel-source", `${model.repo} · ${model.sha.slice(0, 8)}`);
  source.href = `https://huggingface.co/${model.repo}/tree/${model.sha}`;
  title.append(source);
  const meta = element("div", "panel-meta");
  const bytes = model.mirrored_tokenizer_json_bytes || model.tokenizer_json_bytes;
  const size = element("span", "size", `${(bytes / 1e6).toFixed(1)} MB`);
  size.title = "Tokenizer JSON size before transfer compression";
  meta.append(size);
  const remove = element("button", "remove", "×");
  remove.type = "button";
  remove.setAttribute("aria-label", `Remove ${model.label}`);
  remove.addEventListener("click", () => {
    selected = selected.filter((id) => id !== model.id);
    panel.remove(); panels.delete(model.id); updateDropdown(); renderAll();
  });
  meta.append(remove); header.append(title, meta); panel.append(header);
  panels.set(model.id, panel);
  $("panels").append(panel);
  return panel;
}
function loadingBody(panel, label) {
  const body = element("div", "panel-body loading");
  const progress = element("progress");
  progress.setAttribute("aria-label", `Loading ${label}`);
  body.append(element("span", "", `Loading ${label} tokenizer…`), progress);
  panel.querySelector(".panel-body")?.remove();
  panel.append(body);
  panel.dataset.state = "loading";
}
function updateDropdown() {
  $("add-model").replaceChildren(new Option("+ Add a tokenizer", ""));
  for (const model of models.filter((item) => !selected.includes(item.id))) {
    $("add-model").add(new Option(model.label, model.id));
  }
}
async function renderAll() {
  if (!rdkit) return;
  const current = ++generation;
  const smiles = $("smiles").value;
  const prefix = $("prefix").value;
  const special = $("special").checked;
  syncURL();
  $("panels").setAttribute("aria-busy", "true");
  $("app-status").textContent = selected.length ? "" : "Choose a tokenizer to start comparing.";
  await Promise.all(selected.map(async (id) => {
    const model = models.find((item) => item.id === id);
    const panel = panels.get(id) || makePanel(model);
    if (!loaded.has(id)) loadingBody(panel, model.label);
    try {
      const tokenizer = await loadTokenizer(model);
      loaded.set(id, tokenizer);
      if (current !== generation || !panels.has(id)) return;
      const result = analyze(smiles, tokenizer, prefix, special, rdkit);
      renderPanel(panel, result, smiles, rdkit, tokenizer);
    } catch (error) {
      if (current !== generation || !panels.has(id)) return;
      panel.querySelector(".panel-body")?.remove();
      const body = element("div", "panel-body");
      body.append(element("p", "banner", `Could not analyze with ${model.label}: ${error.message}`));
      const retry = element("button", "", "Try again");
      retry.type = "button";
      retry.addEventListener("click", renderAll);
      body.append(retry); panel.append(body); panel.dataset.state = "error";
    }
  }));
  if (current === generation) $("panels").setAttribute("aria-busy", "false");
}
function scheduleRender() {
  clearTimeout(debounce);
  debounce = setTimeout(renderAll, 180);
}

function initSketcher() {
  if (sketcher || !window.JSApplet || !$("sketcher-details").open) return;
  try {
    const width = Math.min($("jsme").clientWidth || 320, 640);
    sketcher = new window.JSApplet.JSME("jsme", `${width}px`, "340px", { options: "oldlook,star" });
    sketcher.readGenericMolecularInput($("smiles").value);
    sketcher.setCallBack("AfterStructureModified", () => {
      if (!sketcherMuted) editSmiles(sketcher.smiles());
    });
    $("sketcher-status").textContent = "Drawing updates the SMILES above. The sketcher may normalize its spelling.";
    window.bondariesSketcher = sketcher;
  } catch (error) { $("sketcher-status").textContent = `Sketcher unavailable: ${error.message}`; }
}
window.addEventListener("jsme-ready", initSketcher);
$("sketcher-details").addEventListener("toggle", initSketcher);
window.addEventListener("resize", () => {
  if (sketcher && $("sketcher-details").open) sketcher.setSize(`${Math.min($("jsme").clientWidth, 640)}px`, "340px");
});
$("load-sketcher").addEventListener("click", () => {
  $("sketcher-details").open = true;
  initSketcher();
  if (sketcher) {
    sketcherMuted = true;
    try { sketcher.readGenericMolecularInput($("smiles").value); }
    finally { sketcherMuted = false; }
  }
});
$("smiles").addEventListener("input", () => editSmiles());
$("prefix").addEventListener("input", scheduleRender);
$("special").addEventListener("change", renderAll);
$("variant").addEventListener("change", () => applyVariant());
$("root-atom").addEventListener("input", () => applyVariant());
$("add-model").addEventListener("change", () => {
  if (!$("add-model").value) return;
  selected.push($("add-model").value); updateDropdown(); renderAll();
});

async function start() {
  try {
    [rdkit, models] = await Promise.all([
      window.initRDKitModule({ locateFile: () => "https://unpkg.com/@rdkit/rdkit@2026.9.1/dist/RDKit_minimal.wasm" }),
      loadModels(),
    ]);
    window.RDKit = rdkit;
    if ($("variant").value !== "typed") applyVariant(false);
    const requested = params.has("m") ? params.get("m").split(",") : models.filter((model) => model.role === "default").map((model) => model.id);
    selected = [...new Set(requested)].filter((id) => models.some((model) => model.id === id));
    updateDropdown();
    for (const model of models) {
      const item = element("li");
      const source = element("a", "", model.label);
      source.href = `https://huggingface.co/${model.repo}/tree/${model.sha}`;
      item.append(source, ` · ${model.sha.slice(0, 10)} · ${model.license_name || model.license} · ${model.mirrored ? "mirrored here (minified)" : "loads from Hugging Face"}`);
      $("attributions").append(item);
    }
    await renderAll();
  } catch (error) {
    $("app-status").textContent = `Could not start the explorer: ${error.message}. Reload to try again.`;
    $("app-status").classList.add("banner");
    $("panels").setAttribute("aria-busy", "false");
  }
}
start();
