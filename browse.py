"""Human QA of a delta tree (train.py's or train_walk.py's): a local web page for browsing its code tree, one node
at a time.

The tree is deterministic, so every training image lies in exactly one node per depth. The script routes them
all down the tree once and serves each node on request. The left of the page shows the focused node and its
two children, each with its predicted mean image (what the code decodes to) beside the actual mean of its
training images, its share of the parent's images (actual, and as the decoder predicts it) and its mse; the
right shows the node's training images, closest to its predicted mean first. Stop it with Ctrl-C.
"""

import argparse
import base64
import io
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import torch
from PIL import Image

from dvae.data import load_mnist
from dvae.deltatree import D, child_key, load_tree, node_order, pixels, root_frontier, split


def png_uri(pixels):
    """float image [D] in [0, 1] -> data URI of a PNG."""
    img = (pixels.clamp(0, 1) * 255).round().to(torch.uint8).view(28, 28).cpu().numpy()
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class Depth:
    """The nodes the training images reach at one depth."""

    def __init__(self, fr, X, p_rare):
        order, offsets = node_order(fr)
        first = order[offsets[:-1]]
        self.keys = fr.bits.tolist()
        self.index = {k: i for i, k in enumerate(self.keys)}
        self.pred = (X - fr.R)[first].half().cpu()          # each node's predicted mean [Nn, D]
        self.order, self.offsets = order.cpu(), offsets.cpu()
        self.err = (fr.R.square().sum(1)).cpu()               # each image's squared error at its node
        self.p_rare = p_rare                                  # the share the node's split predicts for its rare side


class Tree:
    def __init__(self, decoders, X, mean, steps, batch, tol, top):
        self.X, self.top = X.cpu(), top
        fr = root_frontier(X, mean)
        self.depths = []
        for decoder in decoders:
            child, _, s = split(decoder, fr, steps, batch, tol)
            self.depths.append(Depth(fr, X, (s.m / (1 + s.m)).cpu()))
            fr = child
        self.depths.append(Depth(fr, X, None))  # the leaves
        self.K = len(decoders)
        self.pngs: dict[int, str] = {}

    def locate(self, code):
        """(depth, node index) of a code, or None if no training image reaches it."""
        key = 0
        for depth, token in enumerate(code):
            key = int(child_key(torch.tensor([key]), torch.tensor([bool(token)]), depth)[0])
        i = self.depths[len(code)].index.get(key)
        return None if i is None else (len(code), i)

    def images(self, depth, i):
        d = self.depths[depth]
        return d.order[d.offsets[i]:d.offsets[i + 1]]

    def summary(self, depth, i) -> dict[str, Any]:
        d = self.depths[depth]
        idx = self.images(depth, i)
        return {"images": len(idx), "pred": png_uri(d.pred[i].float()), "actual": png_uri(self.X[idx].mean(0)),
                "mse": (d.err[idx].mean() / D).item()}

    def node(self, code):
        where = self.locate(code)
        if where is None:
            raise ValueError("no training image reaches this code")
        depth, i = where
        node = dict(self.summary(depth, i), code=list(code), depth=depth, N=len(self.X))
        children = []
        p_rare = self.depths[depth].p_rare
        if p_rare is not None:
            p = float(p_rare[i])
            for token in (1, 0):
                kid = self.locate(code + (token,))
                if kid is None:
                    continue
                c = self.summary(*kid)
                c.update(token=token, share=c["images"] / node["images"], predicted=p if token else 1 - p)
                children.append(c)
        return dict(node, children=children, entries=self.entries(depth, i, 0))

    def image(self, j):
        if j not in self.pngs:
            self.pngs[j] = png_uri(self.X[j])
        return self.pngs[j]

    def entries(self, depth, i, start):
        """The node's training images from rank start, closest to its predicted mean first."""
        idx = self.images(depth, i)
        err = self.depths[depth].err[idx]
        ranked = idx[torch.argsort(err)][start:start + self.top]
        return [{"image": self.image(j), "index": j, "mse": (self.depths[depth].err[j] / D).item()}
                for j in ranked.tolist()]

    def more(self, code, start):
        where = self.locate(code)
        return {"entries": [] if where is None else self.entries(*where, start)}


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
                if any(t not in (0, 1) for t in code) or len(code) > tree.K:
                    raise ValueError(f"tokens are 0 or 1, at most {tree.K} of them")
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
    print(f"browse the tree at {url} (Ctrl-C to stop)", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped")


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="checkpoints/deltatree.pt")
    p.add_argument("--top", type=int, default=48, help="training images shown per page")
    p.add_argument("--limit", type=int, default=None, help="use only the first N training images (quick tests)")
    p.add_argument("--port", type=int, default=8000, help="0 picks a free port")
    p.add_argument("--no-browser", action="store_true", help="don't open the page automatically")
    p.add_argument("--device", default=None)
    p.add_argument("--data-root", default="data")
    args = p.parse_args()
    device = torch.device(args.device or ("mps" if torch.backends.mps.is_available() else "cpu"))

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    decoders, a = load_tree(ckpt, device), ckpt["args"]
    X = pixels(load_mnist("train", args.data_root)[:args.limit]).to(device)
    tree = Tree(decoders, X, ckpt["mean"].to(device), a["lloyd_steps"], a["node_batch"], a.get("tol", 0.0), args.top)
    print(f"{len(X)} training images down {tree.K} depths ({device})")
    serve(tree, args.port, not args.no_browser, {"ckpt": args.ckpt, "N": len(X), "K": tree.K, "top": args.top})


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Delta Tree Browser</title>
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
.focus { display: flex; gap: 14px; align-items: center; margin-bottom: 14px; flex-wrap: wrap; }
.pair { display: flex; gap: 6px; }
.pair figure { margin: 0; text-align: center; }
.pair figcaption { color: var(--muted); font-size: 11px; margin-top: 2px; }
.stats { display: grid; grid-template-columns: auto auto; gap: 2px 12px; font-size: 13px; }
.stats .k { color: var(--muted); }
.kids { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 8px; }
.kid { border: 1px solid var(--line); border-radius: 8px; padding: 8px; cursor: pointer; text-align: center; }
.kid:hover { background: var(--hover); }
.kid .tok { font-family: ui-monospace, monospace; font-weight: 600; margin-top: 4px; }
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
<header><h1>Delta Tree Browser</h1><span class="hint" id="meta"></span>
<span class="hint">Click a child to focus it. Backspace or &larr; goes up. Token 1 is a split's rare side.</span></header>
<main>
  <section id="tree"></section>
  <section id="images"></section>
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
function pair(n, size) {
  const box = el("div", "pair");
  for (const [src, label] of [[n.pred, "predicted"], [n.actual, "actual mean"]]) {
    const f = el("figure");
    f.append(tile(src, size), el("figcaption", "", label));
    box.append(f);
  }
  return box;
}
function pct(x) { return (100 * x).toFixed(x < 0.001 ? 3 : 1) + "%"; }
function codeText(code) { return code.length ? code.join("") : "(root)"; }

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
  const rows = [["code", codeText(node.code)], ["depth", node.depth],
                ["images", node.images.toLocaleString() + " (" + pct(node.images / META.N) + " of all)"],
                ["mse", node.mse.toFixed(5)]];
  for (const [k, v] of rows) s.append(el("span", "k", k), el("span", "", String(v)));
  focus.append(pair(node, 112), s);
  box.append(focus);
  if (!node.children.length) {
    box.append(el("div", "note", node.depth >= META.K ? "A leaf: the tree ends here." : "No training image goes further."));
    return;
  }
  box.append(el("h2", "", "Children"));
  const kids = el("div", "kids");
  for (const c of node.children) {
    const card = el("div", "kid"), bar = el("div", "bar"), fill = el("i");
    fill.style.width = (100 * c.share) + "%";
    bar.append(fill);
    card.append(pair(c, 64), el("div", "tok", "token " + c.token + (c.token ? " (rare)" : " (common)")), bar,
                el("div", "", pct(c.share) + " of images"),
                el("div", "small", "predicted " + pct(c.predicted) + " · mse " + c.mse.toFixed(5)));
    card.onclick = () => go(node.code.concat([c.token]));
    kids.append(card);
  }
  box.append(kids);
}

function renderImages(node) {
  const box = document.getElementById("images");
  box.replaceChildren(el("h2", "", "Training images in " + codeText(node.code)));
  const note = el("div", "note"), grid = el("div", "entries");
  let shown = 0;
  const add = list => {
    for (const e of list) {
      const cell = el("div", "entry");
      cell.title = `training image ${e.index}`;
      cell.append(tile(e.image, 56), el("div", "", e.mse.toFixed(4)));
      grid.append(cell);
    }
    shown += list.length;
    note.textContent = `${shown} of ${node.images.toLocaleString()} images, closest to the predicted mean first ` +
      `(each with its mse).`;
  };
  add(node.entries);
  box.append(note, grid);
  if (node.entries.length === META.top && shown < node.images) {
    const more = el("button", "more", "Show more");
    more.onclick = async () => {
      more.disabled = true;
      const r = await fetch(`/more?code=${keyOf(node.code)}&start=${shown}`).then(r => r.json());
      add(r.entries);
      more.disabled = false;
      if (r.entries.length < META.top || shown >= node.images) more.remove();
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
  renderImages(node);
  document.title = "Delta Tree Browser · " + codeText(node.code);
  for (const c of node.children) {  // fetch the children ahead of time, so clicking one is instant
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
  `${META.ckpt} · ${META.N.toLocaleString()} training images · ${META.K} tokens`;
show();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
