"""Probe sets.

Sets
  mm      : image + question items (POPE, MMVP, V*, TextVQA, ZoomBench)           -> routing map and lesion probes
  text    : text-only items (Alpaca chats, mm questions without the image)        -> text load on the expert sets
  ascii   : ASCIIEval concepts as ascii_text / ascii_shuffled / ascii_image / caption / photo (COCO via POPE)
  shapes  : synthetic shapes as raster / svg / ascii_text / ascii_image / caption
  hard    : HR-Bench 4K and 8K, MME-RealWorld-Lite
  arc     : ARC-Easy as text-only multiple choice
Each item: dict(uid, set, source, fmt, concept, qtype, prompt, images(list[PIL]|None), assistant(str|None), letters, gold, meta)
"""
import zlib
import ast, csv, io, json, math, random, re
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont
from moedep import paths

ZOOM = paths.DATA     # POPE, MMVP, HR-Bench, MME-RealWorld-Lite
DEP = paths.DATA      # alpaca, ASCIIEval, ai2_arc
FONT = str(Path(__file__).resolve().parent / "fonts" / "DejaVuSansMono.ttf")   # shipped with the repository
MCQ_SUFFIX = "Answer with the option's letter from the given choices directly."
LETTERS = "ABCD"


def _item(uid, set_, source, fmt, concept, qtype, prompt, images=None, assistant=None, letters=None, gold=None, meta=None):
    return dict(uid=uid, set=set_, source=source, fmt=fmt, concept=concept, qtype=qtype, prompt=prompt,
                images=images, assistant=assistant, letters=letters or [], gold=gold, meta=meta or {})


def _pil(b):
    return Image.open(io.BytesIO(b)).convert("RGB")


def _options(choices, gold_idx, rng):
    order = list(range(len(choices))); rng.shuffle(order)
    lines = [f"{LETTERS[i]}. {choices[j]}" for i, j in enumerate(order)]
    gold = LETTERS[order.index(gold_idx)]
    return "\n".join(lines), gold, list(LETTERS[:len(choices)])


# ------------------------------------------------------------------ mm set
def _pope_rows(n, rng):
    rows = []
    for f in sorted((ZOOM / "POPE" / "data").glob("*.parquet")):
        t = pq.read_table(f, columns=["question_id", "question", "answer", "image_source", "category"]).to_pydict()
        rows += [dict(file=str(f), i=i, **{k: t[k][i] for k in t}) for i in range(len(t["question"]))]
    pick = rng.sample(rows, min(n, len(rows)))
    # load image bytes only for picked rows
    by_file = {}
    for r in pick:
        by_file.setdefault(r["file"], []).append(r)
    for f, rs in by_file.items():
        col = pq.read_table(f, columns=["image"]).column("image").to_pylist()
        for r in rs:
            img = col[r["i"]]; r["bytes"] = img["bytes"] if isinstance(img, dict) else img
    return pick


def mm_items(n_per=100, seed=0):
    rng = random.Random(seed); items = []
    for r in _pope_rows(n_per, rng):
        items.append(_item(f"pope_{r['question_id']}_{r['category']}", "mm", "pope", None, None, "yn",
                           r["question"].strip() + "\nAnswer with yes or no.", [_pil(r["bytes"])],
                           gold=r["answer"], meta={"category": r["category"], "image_source": r["image_source"]}))
    rows = list(csv.DictReader(open(ZOOM / "MMVP" / "Questions.csv")))
    for r in rng.sample(rows, min(n_per, len(rows))):
        opts = re.findall(r"\(([a-z])\)\s*([^()]+?)\s*(?=\([a-z]\)|$)", r["Options"])
        choices = [o[1].strip() for o in opts]; gold_idx = [o[0] for o in opts].index(r["Correct Answer"].strip("() "))
        opt_txt, gold, letters = _options(choices, gold_idx, rng)
        items.append(_item(f"mmvp_{r['Index']}", "mm", "mmvp", None, None, "mcq",
                           f"{r['Question'].strip()}\n{opt_txt}\n{MCQ_SUFFIX}",
                           [Image.open(ZOOM / "MMVP" / "MMVP Images" / f"{r['Index']}.jpg").convert("RGB")],
                           letters=letters, gold=gold))
    from moedep.benchmarks import load_vstar, load_textvqa, load_zoombench
    vs = load_vstar()
    for s in rng.sample(vs, min(n_per, len(vs))):
        items.append(_item(s.uid, "mm", "vstar", None, None, "mcq", s.prompt, [Image.open(s.image_path).convert("RGB")],
                           letters=s.letters, gold=s.answer, meta={"category": s.meta.get("category")}))
    tv = load_textvqa()
    for s in rng.sample(tv, min(n_per, len(tv))):
        items.append(_item(s.uid, "mm", "textvqa", None, None, "free", s.prompt, [_pil(s.image_bytes)], gold=s.answer,
                           meta={"answers": s.meta.get("answers", [])}))
    zb = [s for s in load_zoombench() if s.qtype == "mcq"]
    for s in rng.sample(zb, min(n_per, len(zb))):
        items.append(_item(s.uid, "mm", "zoombench", None, None, "mcq", s.prompt, [_pil(s.image_bytes)],
                           letters=s.letters, gold=s.answer))
    return items


# ------------------------------------------------------------------ text set
def text_items(n_alpaca=400, mm=None, n_qonly=300, seed=0, max_chars=1500):
    rng = random.Random(seed + 1); items = []
    f = next((DEP / "alpaca" / "data").glob("*.parquet"))
    t = pq.read_table(f, columns=["instruction", "input", "output"]).to_pydict()
    idx = rng.sample(range(len(t["instruction"])), n_alpaca)
    for i in idx:
        user = t["instruction"][i].strip() + (("\n\n" + t["input"][i].strip()) if t["input"][i].strip() else "")
        items.append(_item(f"alpaca_{i}", "text", "alpaca", "chat", None, "free", user[:max_chars],
                           assistant=t["output"][i].strip()[:max_chars]))
    if mm:
        for it in rng.sample(mm, min(n_qonly, len(mm))):
            items.append(_item("qonly_" + it["uid"], "text", "qonly_" + it["source"], "qonly", None, it["qtype"],
                               it["prompt"], None, letters=it["letters"], gold=it["gold"], meta={"mm_uid": it["uid"]}))
    return items


# ------------------------------------------------------------------ rendering
def render_ascii(text, font_px=14, pad=12, fg=(0, 0, 0), bg=(255, 255, 255)):
    font = ImageFont.truetype(FONT, font_px)
    lines = text.split("\n") or [""]
    cw = font.getlength("M"); lh = int(font_px * 1.2)
    W = int(max(len(l) for l in lines) * cw) + 2 * pad; H = lh * len(lines) + 2 * pad
    img = Image.new("RGB", (max(W, 64), max(H, 64)), bg); d = ImageDraw.Draw(img)
    for k, l in enumerate(lines):
        d.text((pad, pad + k * lh), l, font=font, fill=fg)
    return img


def _article(w):
    return "an" if w[0].lower() in "aeiou" else "a"


# ------------------------------------------------------------------ ASCIIEval formats
def pope_photo_bank(seed=0, max_per_cat=3):
    """COCO images (via POPE) containing a given object (answer yes). concept -> list[bytes]"""
    bank = {}
    for f in sorted((ZOOM / "POPE" / "data").glob("*.parquet")):
        t = pq.read_table(f, columns=["question", "answer", "image"]).to_pydict()
        for q, a, img in zip(t["question"], t["answer"], t["image"]):
            m = re.match(r"Is there (?:a|an) (.+?) in the image\?", q)
            if m and a.strip().lower() == "yes":
                b = img["bytes"] if isinstance(img, dict) else img
                lst = bank.setdefault(m.group(1).lower(), [])
                if len(lst) < max_per_cat:
                    lst.append(b)
    return bank


def _shuffle_lines(art, rng):
    """Control: same characters per line, positions shuffled within the line (destroys the drawing, keeps token statistics)."""
    out = []
    for line in art.split("\n"):
        chars = list(line); rng.shuffle(chars); out.append("".join(chars))
    return "\n".join(out)


def ascii_items(n_concepts=300, formats=("ascii_text", "ascii_image", "caption", "photo", "ascii_shuffled"), seed=0, max_lines=60):
    rng = random.Random(seed + 2)
    rows = [json.loads(l) for l in open(DEP / "ASCIIEval" / "test.jsonl")]
    rows = [r for r in rows if r["ascii_art"].count("\n") < max_lines]
    by_concept = {}
    for r in rows:
        by_concept.setdefault(r["category-3"], []).append(r)
    concepts = sorted(by_concept); rng.shuffle(concepts); concepts = concepts[:n_concepts]
    bank = pope_photo_bank(seed) if "photo" in formats else {}
    items = []
    for c in concepts:
        r = rng.choice(by_concept[c])
        choices = ast.literal_eval(r["choices"]) if isinstance(r["choices"], str) else r["choices"]
        labels = ast.literal_eval(r["labels"]) if isinstance(r["labels"], str) else r["labels"]
        gold_idx = labels.index(1)
        opt_txt, gold, letters = _options(choices, gold_idx, random.Random(zlib.crc32(c.encode()) & 0xFFFF))
        base = f"{c}".replace(" ", "_")
        meta = {"category1": r["category-1"], "category2": r["category-2"], "style": r["style"], "url": r["url"]}
        q = "What does this ASCII art depict?"
        if "ascii_text" in formats:
            items.append(_item(f"ascii_{base}_text", "ascii", "asciieval", "ascii_text", c, "mcq",
                               f"Here is an ASCII art:\n\n{r['ascii_art']}\n\n{q}\n{opt_txt}\n{MCQ_SUFFIX}",
                               letters=letters, gold=gold, meta=meta))
        if "ascii_shuffled" in formats:
            shuf = _shuffle_lines(r["ascii_art"], random.Random(zlib.crc32(c.encode()) & 0xFFFF))
            items.append(_item(f"ascii_{base}_shuffled", "ascii", "asciieval", "ascii_shuffled", c, "mcq",
                               f"Here is an ASCII art:\n\n{shuf}\n\n{q}\n{opt_txt}\n{MCQ_SUFFIX}",
                               letters=letters, gold=gold, meta=meta))
        if "ascii_image" in formats:
            items.append(_item(f"ascii_{base}_image", "ascii", "asciieval", "ascii_image", c, "mcq",
                               f"{q}\n{opt_txt}\n{MCQ_SUFFIX}", [render_ascii(r["ascii_art"])],
                               letters=letters, gold=gold, meta=meta))
        if "caption" in formats:
            items.append(_item(f"ascii_{base}_caption", "ascii", "asciieval", "caption", c, "mcq",
                               f"Consider a simple line drawing of {_article(c)} {c}.\nWhat does this drawing depict?\n{opt_txt}\n{MCQ_SUFFIX}",
                               letters=letters, gold=gold, meta=meta))
        if "photo" in formats and c.lower() in bank:
            items.append(_item(f"ascii_{base}_photo", "ascii", "asciieval", "photo", c, "mcq",
                               f"Which of the following objects is shown in this image?\n{opt_txt}\n{MCQ_SUFFIX}",
                               [_pil(bank[c.lower()][0])], letters=letters, gold=gold, meta=meta))
    return items


# ------------------------------------------------------------------ synthetic shapes
SHAPES = ["circle", "square", "triangle", "diamond", "star", "cross", "arrow", "hexagon"]
COLORS = {"red": (220, 40, 40), "blue": (40, 80, 220), "green": (40, 160, 60), "black": (0, 0, 0)}


def _poly(shape, cx, cy, r):
    if shape == "square":
        return [(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r), (cx - r, cy + r)]
    if shape == "triangle":
        return [(cx, cy - r), (cx + r * 0.95, cy + r * 0.6), (cx - r * 0.95, cy + r * 0.6)]
    if shape == "diamond":
        return [(cx, cy - r), (cx + r * 0.7, cy), (cx, cy + r), (cx - r * 0.7, cy)]
    if shape == "star":
        pts = []
        for k in range(10):
            a = -math.pi / 2 + k * math.pi / 5; rr = r if k % 2 == 0 else r * 0.45
            pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
        return pts
    if shape == "cross":
        t = r * 0.3
        return [(cx - t, cy - r), (cx + t, cy - r), (cx + t, cy - t), (cx + r, cy - t), (cx + r, cy + t), (cx + t, cy + t),
                (cx + t, cy + r), (cx - t, cy + r), (cx - t, cy + t), (cx - r, cy + t), (cx - r, cy - t), (cx - t, cy - t)]
    if shape == "arrow":
        t = r * 0.28
        return [(cx - r, cy - t), (cx + r * 0.2, cy - t), (cx + r * 0.2, cy - r * 0.6), (cx + r, cy),
                (cx + r * 0.2, cy + r * 0.6), (cx + r * 0.2, cy + t), (cx - r, cy + t)]
    if shape == "hexagon":
        return [(cx + r * math.cos(math.pi / 6 + k * math.pi / 3), cy + r * math.sin(math.pi / 6 + k * math.pi / 3)) for k in range(6)]
    raise ValueError(shape)


def _svg(shape, cx, cy, r, color, filled, size=256):
    rgb = "rgb(%d,%d,%d)" % COLORS[color]
    fill, stroke = (rgb, "none") if filled else ("none", rgb)
    head = f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">\n  <rect width="{size}" height="{size}" fill="white"/>\n'
    if shape == "circle":
        body = f'  <circle cx="{cx:.0f}" cy="{cy:.0f}" r="{r:.0f}" fill="{fill}" stroke="{stroke}" stroke-width="6"/>\n'
    else:
        pts = " ".join(f"{x:.0f},{y:.0f}" for x, y in _poly(shape, cx, cy, r))
        body = f'  <polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="6"/>\n'
    return head + body + "</svg>"


def _raster(shape, cx, cy, r, color, filled, size=256):
    img = Image.new("RGB", (size, size), (255, 255, 255)); d = ImageDraw.Draw(img); rgb = COLORS[color]
    kw = dict(fill=rgb) if filled else dict(outline=rgb, width=6)
    if shape == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], **kw)
    else:
        d.polygon(_poly(shape, cx, cy, r), **kw)
    return img


def _ascii_from_raster(img, cols=40, rows=20):
    g = np.asarray(img.convert("L").resize((cols, rows), Image.BOX))
    return "\n".join("".join("#" if v < 200 else " " for v in row) for row in g)


def shape_items(n_var=6, formats=("raster", "svg", "ascii_text", "ascii_image", "caption"), seed=0, size=256):
    rng = random.Random(seed + 3); items = []
    for shape in SHAPES:
        for v in range(n_var):
            r = rng.uniform(0.28, 0.42) * size; cx = rng.uniform(0.38, 0.62) * size; cy = rng.uniform(0.38, 0.62) * size
            color = rng.choice(list(COLORS)); filled = rng.random() < 0.5
            others = rng.sample([s for s in SHAPES if s != shape], 3); choices = [shape] + others
            opt_txt, gold, letters = _options(choices, 0, rng)
            meta = {"shape": shape, "color": color, "filled": filled, "r": r, "cx": cx, "cy": cy}
            uid = f"shape_{shape}_{v}"; ras = _raster(shape, cx, cy, r, color, filled, size)
            fw = "filled" if filled else "outlined"
            if "raster" in formats:
                items.append(_item(uid + "_raster", "shapes", "synthetic", "raster", shape, "mcq",
                                   f"What shape is shown in this image?\n{opt_txt}\n{MCQ_SUFFIX}", [ras], letters=letters, gold=gold, meta=meta))
            if "svg" in formats:
                items.append(_item(uid + "_svg", "shapes", "synthetic", "svg", shape, "mcq",
                                   f"Here is an SVG file:\n\n{_svg(shape, cx, cy, r, color, filled, size)}\n\nWhat shape does this SVG draw?\n{opt_txt}\n{MCQ_SUFFIX}",
                                   letters=letters, gold=gold, meta=meta))
            asc = _ascii_from_raster(ras)
            if "ascii_text" in formats:
                items.append(_item(uid + "_ascii_text", "shapes", "synthetic", "ascii_text", shape, "mcq",
                                   f"Here is an ASCII art:\n\n{asc}\n\nWhat shape does this ASCII art depict?\n{opt_txt}\n{MCQ_SUFFIX}",
                                   letters=letters, gold=gold, meta=meta))
            if "ascii_image" in formats:
                items.append(_item(uid + "_ascii_image", "shapes", "synthetic", "ascii_image", shape, "mcq",
                                   f"What shape does this ASCII art depict?\n{opt_txt}\n{MCQ_SUFFIX}", [render_ascii(asc)],
                                   letters=letters, gold=gold, meta=meta))
            if "caption" in formats:
                items.append(_item(uid + "_caption", "shapes", "synthetic", "caption", shape, "mcq",
                                   f"Consider {_article(fw)} {fw} {color} {shape} drawn on a white background.\nWhat shape is it?\n{opt_txt}\n{MCQ_SUFFIX}",
                                   letters=letters, gold=gold, meta=meta))
    return items


# ------------------------------------------------------------------ harder perception probes
def hard_items(n_per=100, seed=0):
    """HR-Bench 4K/8K (4-way MCQ on 4K/8K images) and MME-RealWorld-Lite (5-way MCQ). Images are base64 strings."""
    import base64
    rng = random.Random(seed + 4); items = []
    for name in ("hr_bench_4k", "hr_bench_8k"):
        f = ZOOM / "HR-Bench" / f"{name}.parquet"
        meta = pq.read_table(f, columns=["index", "answer", "question", "A", "B", "C", "D", "category"]).to_pydict()
        pick = sorted(rng.sample(range(len(meta["index"])), min(n_per, len(meta["index"]))))
        col = pq.read_table(f, columns=["image"]).column("image")
        for i in pick:
            opts = "\n".join(f"{L}. {meta[L][i]}" for L in "ABCD")
            items.append(_item(f"{name}_{meta['index'][i]}", "hard", name, None, None, "mcq",
                               f"{meta['question'][i].strip()}\n{opts}\n{MCQ_SUFFIX}",
                               [_pil(base64.b64decode(col[i].as_py()))], letters=list("ABCD"), gold=meta["answer"][i].strip(),
                               meta={"category": meta["category"][i]}))
    rows = []
    for f in sorted((ZOOM / "MME-RealWorld-Lite" / "data").glob("*.parquet")):
        t = pq.read_table(f, columns=["index", "question", "multi-choice options", "answer", "category", "l2-category"]).to_pydict()
        rows += [dict(file=str(f), i=i, **{k: t[k][i] for k in t}) for i in range(len(t["index"]))]
    pick = rng.sample(rows, min(n_per, len(rows)))
    by_file = {}
    for r in pick:
        by_file.setdefault(r["file"], []).append(r)
    for f, rs in by_file.items():
        col = pq.read_table(f, columns=["bytes"]).column("bytes")
        for r in rs:
            opts = "\n".join(o.replace("(", "", 1).replace(")", ".", 1) for o in r["multi-choice options"])
            items.append(_item(f"mmerw_{r['index']}", "hard", "mmerw_lite", None, None, "mcq",
                               f"{r['question'].strip()}\n{opts}\n{MCQ_SUFFIX}", [_pil(base64.b64decode(col[r['i']].as_py()))],
                               letters=list("ABCDE"), gold=r["answer"].strip(), meta={"category": r["category"], "l2": r["l2-category"]}))
    return items


def arc_items(n=200, seed=0):
    """ARC-Easy test questions as text-only MCQ (letters A-D/E)."""
    f = sorted((DEP / "ai2_arc" / "ARC-Easy").glob("test-*.parquet"))[0]
    t = pq.read_table(f).to_pydict(); rng = random.Random(seed + 5); items = []
    for i in rng.sample(range(len(t["question"])), min(n, len(t["question"]))):
        ch = t["choices"][i]; letters = [chr(ord("A") + k) for k in range(len(ch["text"]))]
        gold = letters[ch["label"].index(t["answerKey"][i])]
        opts = "\n".join(f"{L}. {x}" for L, x in zip(letters, ch["text"]))
        items.append(_item(f"arc_{t['id'][i]}", "arc", "arc_easy", "text", None, "mcq", f"{t['question'][i].strip()}\n{opts}\n{MCQ_SUFFIX}",
                           letters=letters, gold=gold))
    return items


def build(sets, n_mm=100, n_alpaca=400, n_qonly=300, n_concepts=300, n_var=6, seed=0):
    out = []; mm = None
    if "mm" in sets or "text" in sets:
        mm = mm_items(n_mm, seed)
    if "mm" in sets:
        out += mm
    if "text" in sets:
        out += text_items(n_alpaca, mm, n_qonly, seed)
    if "ascii" in sets:
        out += ascii_items(n_concepts, seed=seed)
    if "shapes" in sets:
        out += shape_items(n_var, seed=seed)
    if "hard" in sets:
        out += hard_items(n_mm, seed)
    if "arc" in sets:
        out += arc_items(200, seed)
    return out
