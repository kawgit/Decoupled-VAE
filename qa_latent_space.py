"""Human QA of an encoder's latent space: a local web page for browsing its code tree, one node at a time.

Each node is a code prefix. Its posterior weighs every training image by its chance of producing the prefix,
q(z_1:t | y) (the encoder's own copy, so in-sample), as in the decompression loss. The left of the page shows
the focused node and its children, each with its posterior's mean image as a "profile picture" and its share
of the parent's images; the right shows the focused node's posterior, its heaviest training images first.

The script keeps every slice's answer for every training image in memory and serves each node on request,
computed exactly with the real selector, so any depth can be browsed. Stop it with Ctrl-C.
"""

import argparse
import base64
import io
import json
import threading
import webbrowser
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import torch
from PIL import Image

from dvae.common import add_common_args, add_mode_args, get_device, load_checkpoint
from dvae.data import NUM_PIXELS, load_mnist, to_float
from dvae.models import apply_mode

CHUNK = 4096  # training images per encoder forward pass
CACHE = 512  # nodes whose posterior weights are kept for fast revisits


def png_uri(pixels):
    """uint8 image [28, 28] -> data URI of a PNG."""
    buf = io.BytesIO()
    Image.fromarray(pixels.numpy()).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class Tree:
    """Serves nodes of an encoder's code tree. A node's posterior weights are its parent's times the answers of
    the slice asked at the parent, so they're cached for recently visited nodes."""

    def __init__(self, encoder, images, probs, top):
        self.encoder, self.top = encoder, top
        self.images = images.cpu()  # uint8 [N, 28, 28]
        self.probs = probs  # [N, S, V]
        self.N, self.S, self.V = probs.shape
        flat = to_float(images).flatten(1)
        self.flat = flat
        self.centered = flat - flat.mean(0)  # for the spread's precision
        self.sq = self.centered.pow(2).sum(1)
        self.cache: OrderedDict[tuple[int, ...], tuple[torch.Tensor, tuple[int, ...]]] = OrderedDict()
        self.pngs: dict[int, str] = {}
        self.lock = threading.Lock()

    def weights(self, code):
        """(posterior weights [N], slices asked along the code) of a code."""
        if code in self.cache:
            self.cache.move_to_end(code)
            return self.cache[code]
        if not code:
            out = (torch.ones(self.N, device=self.probs.device), ())
        else:
            w, path = self.weights(code[:-1])
            h = self.next_slice(code[:-1], path)
            if h is None:
                raise ValueError(f"codes end after {self.S} tokens, one per slice")
            out = (w * self.probs[:, h, code[-1]], path + (h,))
        self.cache[code] = out
        if len(self.cache) > CACHE:
            self.cache.popitem(last=False)
        return out

    def next_slice(self, code, path):
        """The slice the knight asks after a code, or None once every slice is used."""
        if len(path) >= self.S:
            return None
        dev = self.probs.device
        prefix = torch.tensor(code, dtype=torch.long, device=dev).view(1, len(code))
        used = torch.tensor(path, dtype=torch.long, device=dev).view(1, len(path))
        return int(self.encoder.pick(prefix, used))

    def summaries(self, W):
        """Posterior summaries of the rows of a weight matrix [n, N]."""
        mass = W.sum(1)
        W2 = W.pow(2).sum(1)
        M = W @ self.centered
        Q = W @ self.sq
        pairs = mass * mass - W2
        spread = 2 * (mass * Q - M.pow(2).sum(1)) / (NUM_PIXELS * pairs.clamp_min(1e-30))
        means = ((W @ self.flat) / mass.clamp_min(1e-30)[:, None]).clamp(0, 1).mul(255).round().to(torch.uint8)
        out = []
        for m, w2, s, ok, img in zip(mass.tolist(), W2.tolist(), spread.tolist(), (pairs > 1e-9 * mass * mass).tolist(),
                                     means.view(-1, 28, 28).cpu()):
            out.append({"mass": m, "effective": m * m / w2 if w2 > 0 else 0.0, "spread": s if ok else None,
                        "mean": png_uri(img)})
        return out

    def image(self, j):
        if j not in self.pngs:
            self.pngs[j] = png_uri(self.images[j])
        return self.pngs[j]

    def entries(self, w, start, count):
        """The posterior's training images from rank start, heaviest first."""
        total = float(w.sum())
        if total <= 0:
            return []
        top = w.topk(min(start + count, self.N))
        return [{"image": self.image(j), "index": j, "weight": v / total}
                for j, v in zip(top.indices[start:].tolist(), top.values[start:].tolist()) if v > 0]

    @torch.no_grad()
    def node(self, code):
        with self.lock:
            w, path = self.weights(code)
            node = self.summaries(w[None])[0]
            node.update(code=list(code), path=list(path), slice=self.next_slice(code, path), children=[],
                        committed=code in self.encoder.commits)
            if node["slice"] is not None and node["mass"] > 0:
                kids = self.summaries(w[None] * self.probs[:, node["slice"]].T)  # one row per token
                for c, kid in enumerate(kids):
                    kid.update(token=c, share=kid["mass"] / node["mass"], committed=code + (c,) in self.encoder.commits)
                node["children"] = kids
            node["entries"] = self.entries(w, 0, self.top)
            return node

    @torch.no_grad()
    def more(self, code, start):
        with self.lock:
            return {"entries": self.entries(self.weights(code)[0], start, self.top)}


def parse_code(text):
    return tuple(int(t) for t in text.split(".") if t != "")


def serve(tree, port, open_browser, meta):
    page = PAGE.replace("__META__", json.dumps(meta)).encode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # keep the terminal quiet
            pass

        def reply(self, body, kind):
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            query = parse_qs(url.query)
            try:
                if url.path == "/":
                    return self.reply(page, "text/html; charset=utf-8")
                code = parse_code(query.get("code", [""])[0])
                if any(not 0 <= t < tree.V for t in code):
                    raise ValueError(f"tokens must be in 0..{tree.V - 1}")
                if url.path == "/node":
                    body = tree.node(code)
                elif url.path == "/more":
                    body = tree.more(code, int(query.get("start", ["0"])[0]))
                else:
                    return self.send_error(404)
                self.reply(json.dumps(body).encode(), "application/json")
            except ValueError as err:
                self.send_error(400, str(err))

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"browse the latent space at {url} (Ctrl-C to stop)", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped")


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--encoder", default="checkpoints/encoder.pt")
    add_mode_args(p, "posterior")
    p.add_argument("--top", type=int, default=48, help="posterior images shown per page")
    p.add_argument("--limit", type=int, default=None, help="use only the first N training images (quick tests)")
    p.add_argument("--port", type=int, default=8000, help="0 picks a free port")
    p.add_argument("--no-browser", action="store_true", help="don't open the page automatically")
    args = p.parse_args()

    device = get_device(args.device)
    encoder, _ = load_checkpoint(args.encoder, "encoder", device)
    images = load_mnist("train", args.data_root, device)[: args.limit]
    x = to_float(images)
    probs = torch.cat([apply_mode(encoder.features(x[i:i + CHUNK]), args.posterior_mode, args.posterior_top_p).exp()
                       for i in range(0, len(x), CHUNK)])  # [N, S, V]: every slice's answer for every training image
    tree = Tree(encoder, images, probs, args.top)
    print(f"{tree.N} training images, {tree.S} slices, V={tree.V} ({device})")
    serve(tree, args.port, not args.no_browser,
          {"encoder": args.encoder, "N": tree.N, "S": tree.S, "V": tree.V, "top": args.top})


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Latent Space Browser</title>
<style>
:root {
  --bg: #f6f6f4; --panel: #ffffff; --ink: #1d1d1f; --muted: #6b6b70; --line: #e2e2df;
  --accent: #2f6fdf; --hover: #eef3fc;
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #141416; --panel: #1d1d20; --ink: #ececef; --muted: #9a9aa3; --line: #2e2e33;
          --accent: #7aa7ff; --hover: #24293a; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.4 system-ui, sans-serif; }
header { padding: 12px 20px; border-bottom: 1px solid var(--line); display: flex; flex-wrap: wrap; gap: 6px 16px;
  align-items: baseline; }
header h1 { font-size: 16px; margin: 0; }
.hint { color: var(--muted); font-size: 12px; }
main { display: grid; grid-template-columns: minmax(380px, 1fr) minmax(380px, 1fr); gap: 16px; padding: 16px 20px; }
@media (max-width: 860px) { main { grid-template-columns: 1fr; } }
section { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 14px; min-width: 0; }
h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); margin: 0 0 10px; }
.crumbs { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 12px; font-family: ui-monospace, monospace; }
.crumbs button { border: 1px solid var(--line); background: none; color: var(--accent); border-radius: 6px;
  padding: 2px 7px; cursor: pointer; font: inherit; }
.crumbs button.here { color: var(--ink); cursor: default; }
.focus { display: flex; gap: 14px; align-items: center; margin-bottom: 14px; }
.stats { display: grid; grid-template-columns: auto auto; gap: 2px 12px; font-size: 13px; }
.stats .k { color: var(--muted); }
.kids { display: grid; grid-template-columns: repeat(auto-fill, minmax(104px, 1fr)); gap: 8px; }
.kid { border: 1px solid var(--line); border-radius: 8px; padding: 6px; cursor: pointer; text-align: center; }
.kid:hover { background: var(--hover); }
.kid .tok { font-family: ui-monospace, monospace; font-weight: 600; }
.kid .bar { height: 5px; background: var(--line); border-radius: 3px; margin: 4px 0 2px; overflow: hidden; }
.kid .bar i { display: block; height: 100%; background: var(--accent); }
.small { color: var(--muted); font-size: 11px; }
img.tile { image-rendering: pixelated; border-radius: 4px; background: #000; display: block; margin: 0 auto; }
.entries { display: grid; grid-template-columns: repeat(auto-fill, minmax(64px, 1fr)); gap: 8px; }
.entry { text-align: center; font-size: 11px; color: var(--muted); }
.note { color: var(--muted); font-size: 12px; margin: 8px 0 12px; }
.more { margin-top: 12px; border: 1px solid var(--line); background: none; color: var(--accent); border-radius: 6px;
  padding: 4px 10px; cursor: pointer; font: inherit; }
.loading { opacity: .55; transition: opacity .1s; }
</style>
</head>
<body>
<header><h1>Latent Space Browser</h1><span class="hint" id="meta"></span>
<span class="hint">Click a child to focus it. Backspace or &larr; goes up. &#9733; marks nodes with a slice
built for them.</span></header>
<main>
  <section id="tree"></section>
  <section id="posterior"></section>
</main>
<script>
const META = __META__;
const nodes = new Map();  // code string -> promise of the node's JSON
let current = null;

function fetchNode(key) {
  if (!nodes.has(key)) {
    const p = fetch("/node?code=" + key).then(r => { if (!r.ok) throw new Error(r.statusText); return r.json(); });
    p.catch(() => nodes.delete(key));
    nodes.set(key, p);
  }
  return nodes.get(key);
}
function keyOf(code) { return code.join("."); }
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}
function tile(src, size) {
  const i = el("img", "tile");
  i.src = src; i.width = i.height = size; i.alt = "";
  return i;
}
function pct(x) { return (100 * x).toFixed(x < 0.001 ? 3 : 1) + "%"; }
function imgs(m) { return m >= 100 ? Math.round(m).toLocaleString() : m.toFixed(1); }
function codeText(code) { return code.length ? code.join("·") : "(empty)"; }

function renderTree(node) {
  const box = document.getElementById("tree");
  box.replaceChildren(el("h2", "", "Tree"));
  const crumbs = el("div", "crumbs");
  for (let i = 0; i <= node.code.length; i++) {
    const here = i === node.code.length, sub = node.code.slice(0, i);
    const b = el("button", here ? "here" : "", i === 0 ? "root" : String(node.code[i - 1]));
    if (!here) b.onclick = () => go(sub);
    crumbs.append(b);
  }
  box.append(crumbs);
  const focus = el("div", "focus"), s = el("div", "stats");
  const rows = [["code", codeText(node.code)], ["depth", node.code.length],
                ["images", imgs(node.mass) + " (" + pct(node.mass / META.N) + " of all)"],
                ["effective images", imgs(node.effective)],
                ["spread", node.spread === null ? "–" : node.spread.toFixed(5)],
                ["asks slice", node.slice === null ? "none left" :
                  node.slice + (node.committed ? " (built for this node)" : " (the selector's pick)")]];
  for (const [k, v] of rows) s.append(el("span", "k", k), el("span", "", String(v)));
  focus.append(tile(node.mean, 112), s);
  box.append(focus);
  if (!node.children.length) {
    box.append(el("div", "note", "No children: every slice has been asked along this code."));
    return;
  }
  box.append(el("h2", "", "Children by share of this node's images"));
  const kids = el("div", "kids");
  for (const c of node.children.slice().sort((a, b) => b.mass - a.mass)) {
    const card = el("div", "kid"), bar = el("div", "bar"), fill = el("i");
    fill.style.width = (100 * c.share) + "%";
    bar.append(fill);
    card.append(tile(c.mean, 64), el("div", "tok", "token " + c.token + (c.committed ? " \u2605" : "")), bar,
                el("div", "", pct(c.share)),
                el("div", "small", imgs(c.mass) + " images"));
    card.onclick = () => go(node.code.concat([c.token]));
    kids.append(card);
  }
  box.append(kids);
}

function renderPosterior(node) {
  const box = document.getElementById("posterior");
  box.replaceChildren(el("h2", "", "Posterior of " + codeText(node.code)));
  const note = el("div", "note"), grid = el("div", "entries");
  let shown = 0, covered = 0;
  const add = list => {
    for (const e of list) {
      covered += e.weight;
      const cell = el("div", "entry");
      cell.title = `training image ${e.index}, cumulative ${pct(covered)}`;
      cell.append(tile(e.image, 56), el("div", "", pct(e.weight)));
      grid.append(cell);
    }
    shown += list.length;
    note.textContent = `The ${shown} heaviest training images, by posterior mass. Together they hold ` +
      `${pct(covered)} of it.`;
  };
  add(node.entries);
  box.append(note, grid);
  if (node.entries.length === META.top) {
    const more = el("button", "more", "Show more");
    more.onclick = async () => {
      more.disabled = true;
      const r = await fetch(`/more?code=${keyOf(node.code)}&start=${shown}`).then(r => r.json());
      add(r.entries);
      more.disabled = false;
      if (r.entries.length < META.top) more.remove();
    };
    box.append(more);
  }
}

function go(code) { location.hash = "#" + keyOf(code); }
async function show() {
  const key = location.hash.slice(1);
  current = key;
  document.querySelector("main").classList.add("loading");
  let node;
  try { node = await fetchNode(key); } catch (err) { if (key) go([]); return; }
  if (current !== key) return;  // the user moved on while it loaded
  document.querySelector("main").classList.remove("loading");
  renderTree(node);
  renderPosterior(node);
  document.title = "Latent Space Browser · " + codeText(node.code);
  // fetch the children ahead of time, heaviest first, so clicking one is instant
  for (const c of node.children.slice().sort((a, b) => b.mass - a.mass)) {
    if (current !== key) break;
    try { await fetchNode(keyOf(node.code.concat([c.token]))); } catch (err) { break; }
  }
}
window.addEventListener("hashchange", show);
window.addEventListener("keydown", e => {
  if ((e.key === "Backspace" || e.key === "ArrowLeft") && current) {
    e.preventDefault();
    go(current.split(".").slice(0, -1).map(Number));
  }
});
document.getElementById("meta").textContent =
  `${META.encoder} · ${META.N.toLocaleString()} training images · ${META.S} slices`;
show();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
