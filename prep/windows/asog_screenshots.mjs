// Screenshots of an .asog in the animated-sog three.js player (examples/three), headless.
//
// Needs playwright-core in C:\o4d\asog-tools (npm install playwright-core@1.57.0 there) and Playwright's
// Chromium (%LOCALAPPDATA%\ms-playwright\chromium-*), or pass --browser <chrome.exe / msedge.exe>.
// The player must be served (asog_convert.ps1 -Serve, or python -m http.server 8765 in the Open4DGS folder).
//
//   node asog_screenshots.mjs <player url> <out dir> [--times 0.1,2.9] [--side] [--browser exe]
//
// Pauses the player's clock at each time, waits for the depth sort to catch up and saves
// <out>/<name>_t<time>_<view>.png (views: front = the viewer's framing, side = 90 deg around the target).
import fs from "node:fs";
import path from "node:path";
const { chromium } = await import("file:///C:/o4d/asog-tools/node_modules/playwright-core/index.mjs");

const args = process.argv.slice(2);
const url = args[0], out = args[1];
const opt = (k, d) => { const i = args.indexOf(k); return i < 0 ? d : args[i + 1]; };
const times = opt("--times", "0.1,2.9").split(",").map(Number);
const views = args.includes("--side") ? ["front", "side"] : ["front"];
const name = opt("--name", "asog");
let exe = opt("--browser", null);
if (!exe) {
    const base = path.join(process.env.LOCALAPPDATA, "ms-playwright");
    const dirs = fs.readdirSync(base).filter((d) => /^chromium-\d+$/.test(d)).sort();
    exe = path.join(base, dirs.at(-1), "chrome-win64", "chrome.exe");
}
fs.mkdirSync(out, { recursive: true });

const browser = await chromium.launch({ executablePath: exe, headless: true, args: ["--use-angle=d3d11", "--enable-gpu", "--ignore-gpu-blocklist"] });
const page = await browser.newPage({ viewport: { width: 960, height: 1080 } });
page.on("console", (m) => { if (m.type() === "error" || m.type() === "warning") console.log(`[page ${m.type()}] ${m.text()}`); });
page.on("pageerror", (e) => console.log(`[page error] ${e.message}`));
await page.goto(url);
await page.waitForFunction(() => window.asogViewer?.player?.stats || document.getElementById("error"), null, { timeout: 300000 });
const err = await page.evaluate(() => document.getElementById("error")?.textContent);
if (err) { console.log(err); await browser.close(); process.exit(1); }

const info = await page.evaluate(() => {
    const v = window.asogViewer, p = v.player;
    v.controls.autoRotateSpeed = 0;               // keep the camera where we put it
    v.clock.playing = false;
    document.getElementById("play").textContent = "Play";
    const gl = v.renderer.getContext();
    const dbg = gl.getExtension("WEBGL_debug_renderer_info");
    return { splats: p.numSplats, duration: p.duration, bounds: p.bounds, stats: p.stats,
             manifestMotion: p.manifest?.motion ?? null,
             gpu: dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : "?",
             target: v.controls.target.toArray(), camera: v.camera.position.toArray() };
});
console.log(JSON.stringify(info));

const files = [];
for (const view of views) {
    await page.evaluate((view) => {
        const v = window.asogViewer, c = v.camera, t = v.controls.target;
        if (!v.home) v.home = c.position.clone();
        c.position.copy(v.home);
        if (view === "side") c.position.sub(t).applyAxisAngle({ x: 0, y: 1, z: 0, isVector3: true }, Math.PI / 2).add(t);
        v.controls.update();
    }, view);
    for (const t of times) {
        await page.evaluate((t) => window.asogViewer.clock.set(t), t);
        await page.waitForTimeout(2500);          // the sort worker re-sorts for the new time / camera
        const file = path.join(out, `${name}_t${t.toFixed(2)}_${view}.png`);
        await page.screenshot({ path: file });
        files.push(file);
        console.log(`saved ${file}  (player time ${await page.evaluate(() => window.asogViewer.player.time.toFixed(3))} s)`);
    }
}
await browser.close();
