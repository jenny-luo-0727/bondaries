/** Load pinned tokenizer definitions, sharing in-flight work across panels. */
const TOKENIZER_MODULE = "https://cdn.jsdelivr.net/npm/@huggingface/tokenizers@0.2.0/dist/tokenizers.mjs";
const CACHE_NAME = "bondaries-tokenizers-v1";
const tokenizerPromises = new Map();
let modelsPromise;

async function fetchJSON(url, persistent = true) {
  let cache;
  if (persistent) {
    try {
      cache = await globalThis.caches?.open(CACHE_NAME);
      const response = await cache?.match(url);
      if (response?.ok) return await response.json();
    } catch {
      // Storage can be disabled or full; fetching still works without it.
      cache = undefined;
    }
  }
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Unable to load ${url}: HTTP ${response.status}`);
  const data = await response.clone().json();
  try {
    await cache?.put(url, response);
  } catch {
    // A quota failure must not prevent the current tokenizer from loading.
  }
  return data;
}

/** Resolve the generated browser manifest. Failed requests can be retried. */
export function loadModels() {
  if (!modelsPromise) {
    modelsPromise = fetchJSON(new URL("../models.json", import.meta.url), false).catch((error) => {
      modelsPromise = undefined;
      throw error;
    });
  }
  return modelsPromise;
}

/** Resolve a tokenizer from a mirrored definition or its pinned upstream commit. */
export function loadTokenizer(model) {
  const key = `${model.repo}@${model.sha}:${Boolean(model.mirrored)}`;
  if (!tokenizerPromises.has(key)) {
    const base = model.mirrored
      ? new URL(`../tokenizers/${model.id}/${model.sha}/`, import.meta.url).href
      : `https://huggingface.co/${model.repo}/resolve/${model.sha}/`;
    const promise = Promise.all([
      import(TOKENIZER_MODULE), fetchJSON(`${base}tokenizer.json`), fetchJSON(`${base}tokenizer_config.json`),
    ]).then(([{ Tokenizer }, definition, config]) => new Tokenizer(definition, config)).catch((error) => {
      tokenizerPromises.delete(key);
      throw error;
    });
    tokenizerPromises.set(key, promise);
  }
  return tokenizerPromises.get(key);
}
