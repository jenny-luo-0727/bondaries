"""Exercise the static explorer using local assets and a fully intercepted network."""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import mimetypes
import os
import tempfile
import threading
from pathlib import Path
from urllib.parse import unquote, urlparse

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
CDN_ASSETS = {
    "https://cdn.jsdelivr.net/npm/@huggingface/tokenizers@0.2.0/": "@huggingface/tokenizers",
    "https://unpkg.com/@rdkit/rdkit@2026.9.1/": "@rdkit/rdkit",
    "https://cdn.jsdelivr.net/npm/jsme-editor@2024.4.29/": "jsme-editor",
}


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    """Serve the site without interleaving request logs with check results."""

    def log_message(self, format, *args):
        """Suppress routine HTTP access logging."""


def main():
    """Run browser checks and save screenshots to the requested artifact directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=Path, default=os.environ.get("BONDARIES_SHOTS"))
    args = parser.parse_args()
    if args.shots is None:
        parser.error("set --shots or BONDARIES_SHOTS to an artifact directory")
    print(f"Browser temporary directory: {tempfile.gettempdir()}", flush=True)
    shots = Path(args.shots)
    shots.mkdir(parents=True, exist_ok=True)
    hf_home = Path(os.environ["HF_HOME"])
    manifest = json.loads((ROOT / "docs/models.json").read_text())
    defaults = [model for model in manifest if model["role"] == "default"]
    assert len(defaults) == 5, f"Expected five defaults, found {len(defaults)}"
    allowed_models = {(model["repo"], model["sha"]) for model in manifest}
    blocked = []
    served = []
    referrers = []
    checks = []
    grid = []
    errors = []
    handler = functools.partial(QuietHandler, directory=str(ROOT / "docs"))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f"http://127.0.0.1:{server.server_address[1]}"

    def route_request(route):
        """Fulfill pinned external URLs from disk and deny every other remote URL."""
        url = route.request.url
        if route.request.headers.get("referer"):
            referrers.append(route.request.headers["referer"])
        if url.startswith(origin + "/"):
            route.continue_()
            return
        local = None
        for prefix, package in CDN_ASSETS.items():
            if url.startswith(prefix):
                relative = unquote(urlparse(url[len(prefix):]).path)
                if ".." not in Path(relative).parts:
                    local = ROOT / "node_modules" / package / relative
                break
        if url.startswith("https://huggingface.co/"):
            parts = unquote(urlparse(url).path).strip("/").split("/")
            if len(parts) == 5 and parts[2] == "resolve":
                repo, sha, name = "/".join(parts[:2]), parts[3], parts[4]
                if (repo, sha) in allowed_models and name in {
                    "tokenizer.json", "tokenizer_config.json"
                }:
                    local = hf_home / "hub" / ("models--" + repo.replace("/", "--"))
                    local = local / "snapshots" / sha / name
        if local and local.is_file():
            mime = mimetypes.guess_type(local.name)[0] or "application/octet-stream"
            if local.suffix in {".js", ".mjs"}:
                mime = "application/javascript"
            route.fulfill(path=local, content_type=mime, headers={"Access-Control-Allow-Origin": "*"})
            served.append(url)
        else:
            blocked.append(url)
            route.abort("blockedbyclient")

    def ready(page, smiles=None, count=5):
        """Wait until every selected panel reflects the current input."""
        try:
            page.wait_for_function(
                """({smiles, count}) => {
                  const panels = [...document.querySelectorAll('.panel')];
                  const failure = panels.find(p => p.dataset.state === 'error');
                  if (failure) throw new Error(failure.innerText);
                  const status = document.querySelector('#app-status');
                  if (status.classList.contains('banner')) throw new Error(status.innerText);
                  return document.querySelector('#panels').getAttribute('aria-busy') === 'false'
                    && panels.length === count
                    && panels.every(p => p.dataset.state === 'ready'
                      && (smiles == null || p.dataset.smiles === smiles));
                }""",
                arg={"smiles": smiles, "count": count}, timeout=20000,
            )
        except Exception:
            print(page.evaluate("({busy: document.querySelector('#panels').getAttribute('aria-busy'), panels: [...document.querySelectorAll('.panel')].map(p => ({...p.dataset}))})"), flush=True)
            print(page.evaluate("document.body.innerText"), flush=True)
            print({"served": served, "blocked": blocked, "errors": errors}, flush=True)
            page.screenshot(path=shots / "failure.png", full_page=True)
            raise

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            context = browser.new_context(viewport={"width": 1280, "height": 1000}, color_scheme="light")
            context.route("**/*", route_request)
            page = context.new_page()
            page.on("pageerror", lambda error: (errors.append(str(error)), print(f"Page error: {error}", flush=True)))
            page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
            print("Loading local page", flush=True)
            page.goto(origin + "/", wait_until="domcontentloaded")
            print("Page DOM loaded; waiting for panels", flush=True)
            ready(page)
            examples = page.evaluate("async () => (await import('./js/app.js')).EXAMPLES")
            for example in examples:
                page.get_by_role("button", name=example["label"], exact=True).click()
                ready(page, example["smiles"])
                rows = page.locator(".panel").evaluate_all("""panels => panels.map(panel => ({
                  model: panel.dataset.model, colored: panel.dataset.colored,
                  svg: Boolean(panel.querySelector('svg')),
                  tokens: panel.analysisResult.token_count,
                  crimes: Object.fromEntries(Object.entries(panel.analysisResult.crimes).map(([k,v]) => [k,v.length]))
                }))""")
                for row in rows:
                    assert row["colored"] == "true" and row["svg"], (example, row)
                    grid.append({"example": example["label"], "smiles": example["smiles"], **row})
            assert not errors, errors
            checks.append("PASS all 5 defaults × 6 examples: 30 colored structures, no console errors")

            page.get_by_role("button", name="D-Glucose (open chain)", exact=True).click()
            ready(page, examples[4]["smiles"])
            first = page.locator(".panel").first
            first.locator('.chip[data-token-index="0"]').hover()
            assert first.locator(".atom-active").count() > 0
            page.mouse.move(0, 0)
            first.locator(".molecule ellipse.atom-0").first.hover()
            assert first.locator(".chip.is-active").count() > 0
            page.mouse.move(0, 0)
            checks.append("PASS token → atom and atom → token hover links")

            for theme in ("light", "dark"):
                page.emulate_media(color_scheme=theme)
                for width in (1280, 390):
                    page.set_viewport_size({"width": width, "height": 1000})
                    ready(page, examples[4]["smiles"])
                    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), (theme, width)
                    path = shots / f"{theme}-{width}.png"
                    page.screenshot(path=path, full_page=True)
                    checks.append(f"PASS {theme} {width}px: no horizontal scroll; screenshot {path.name}")
            page.set_viewport_size({"width": 1280, "height": 1000})
            page.emulate_media(color_scheme="light")
            page.locator("#theme").click()
            ready(page)
            assert page.locator("html").get_attribute("data-theme") == "dark"
            page.locator("#theme").click()
            ready(page)
            assert page.locator("html").get_attribute("data-theme") == "light"
            checks.append("PASS manual theme toggle overrides the system theme")
            page.select_option("#add-model", "gpt2")
            ready(page, count=6)
            assert page.locator('.panel[data-model="gpt2"]').get_attribute("data-colored") == "true"
            page.get_by_role("button", name="Remove GPT-2", exact=True).click()
            ready(page)
            checks.append("PASS add/remove tokenizer dropdown with GPT-2")
            page.locator(".panel").first.screenshot(path=shots / "panel-detail.png")

            page.locator("#prefix").fill("SMILES: ")
            page.locator("#special").uncheck()
            ready(page)
            page.wait_for_function("new URLSearchParams(location.search).get('p') === 'SMILES: '")
            assert not page.locator(".chip.special").count()
            page.select_option("#variant", "canonical")
            ready(page)
            canonical = page.locator("#smiles").input_value()
            assert canonical
            page.select_option("#variant", "rooted")
            page.locator("#root-atom").fill("2")
            page.wait_for_function("new URLSearchParams(location.search).get('v') === 'rooted:2'")
            ready(page)
            shared = page.url
            exact = page.locator("#smiles").input_value()
            page.goto(shared, wait_until="domcontentloaded")
            ready(page, exact)
            assert page.locator("#prefix").input_value() == "SMILES: "
            assert not page.locator("#special").is_checked()
            assert page.locator("#variant").input_value() == "rooted"
            checks.append("PASS editable canonical/rooted variants, prefix/special toggles, URL round trip")

            source = "CC(O)C(=O)O"
            page.locator("#smiles").fill(source)
            ready(page, source)
            assert page.locator("#variant").input_value() == "typed"
            assert page.evaluate("new URLSearchParams(location.search).get('s')") == source
            page.select_option("#variant", "rooted")
            page.locator("#root-atom").fill("2")
            page.wait_for_function("new URLSearchParams(location.search).get('v') === 'rooted:2'")
            ready(page, "OC(C)C(=O)O")
            shared = page.url
            assert page.evaluate("new URLSearchParams(location.search).get('s')") == source
            page.locator("#root-atom").fill("3")
            ready(page, "C(C(C)O)(=O)O")
            stepped = page.locator("#smiles").input_value()
            page.goto(shared, wait_until="domcontentloaded")
            ready(page, "OC(C)C(=O)O")
            page.locator("#root-atom").fill("3")
            ready(page, stepped)
            assert page.locator("#smiles").input_value() == stepped
            assert page.evaluate("new URLSearchParams(location.search).get('s')") == source
            checks.append("PASS rooted stepper retains source atom order after URL reload")

            page.locator("#load-sketcher").click()
            page.wait_for_function("Boolean(window.bondariesSketcher)", timeout=60000)
            page.evaluate("window.bondariesSketcher.readGenericMolecularInput('CCO')")
            page.wait_for_function("document.getElementById('smiles').value === 'CCO'", timeout=15000)
            ready(page, "CCO")
            checks.append("PASS JSME structure edit callback → editable SMILES → all panels")
            page.set_viewport_size({"width": 390, "height": 1000})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.locator("#sketcher-details").screenshot(path=shots / "sketcher-390.png")
            checks.append("PASS open JSME sketcher fits 390px viewport")
            page.set_viewport_size({"width": 1280, "height": 1000})
            page.locator("#sketcher-details").evaluate("el => el.open = false")

            page.evaluate("""async () => {
              const { renderPanel } = await import('./js/render.js');
              const panel = document.querySelector('.panel');
              const result = structuredClone(panel.analysisResult);
              result.alignment_ok = false;
              result.error = 'Alignment failed. Atom colors are disabled.';
              renderPanel(panel, result, panel.dataset.smiles, window.RDKit);
            }""")
            first = page.locator(".panel").first
            assert first.locator(".alignment-banner").is_visible()
            assert first.get_attribute("data-colored") == "false"
            assert first.locator(".molecule ellipse").count() == 0
            assert first.locator(".chip:not(.special)").count() == 0
            checks.append("PASS forced alignment failure: visible banner, no token or atom colors")

            page.evaluate("""async () => {
              const { renderPanel } = await import('./js/render.js');
              const panel = document.querySelector('.panel');
              const result = structuredClone(panel.analysisResult);
              result.alignment_ok = true; result.valid = true; result.error = null;
              result.lexemes = result.lexemes.filter(item => item.atom_index !== 0);
              renderPanel(panel, result, panel.dataset.smiles, window.RDKit);
            }""")
            assert "Atom-count mismatch" in first.locator(".alignment-banner").inner_text()
            assert first.get_attribute("data-colored") == "false"
            checks.append("PASS atom-count mismatch: visible banner, no colors")

            page.locator("#smiles").fill("C1invalid")
            ready(page, "C1invalid")
            assert page.locator('.panel[data-colored="false"]').count() == 5
            checks.append("PASS invalid SMILES: all five panels disable colors")
            assert not blocked, blocked
            assert not referrers, referrers
            checks.append("PASS privacy: no Referer headers on routed asset requests")
            checks.append(f"PASS offline routing: {len(served)} external asset requests fulfilled locally; none escaped")
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        report = {"checks": checks, "grid": grid, "errors": errors, "blocked": blocked}
        (shots / "browser-results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
