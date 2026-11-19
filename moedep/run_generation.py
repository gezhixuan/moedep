"""Generation metrics under a lesion.
(1) Teacher-forced NLL of the COCO reference captions (5 per image) for the POPE images of the mm probe set.
(2) Greedy TextVQA answers scored by VQA soft accuracy (10 human answers).
Conditions as in moedep.run_ablation: --mask NAME=path (repeatable), --cond none | NAME:MODE | NAME:MODE:SCOPE | random:NAME:SEED."""
import argparse, collections, json, random, re, time, traceback
from pathlib import Path
import numpy as np
import torch
from moedep.kimi_model import load_runner
from moedep import data as D
from moedep.run_ablation import random_masks
from moedep import paths

COCO = paths.DATA / "coco_ann" / "annotations" / "captions_val2014.json"
CAP_PROMPT = "Describe this image in one sentence."
_ART = {"a", "an", "the"}


def vqa_norm(s):
    s = s.lower().strip(); s = re.sub(r"[^\w\s]", "", s)
    return " ".join(w for w in s.split() if w not in _ART)


def vqa_soft(pred, answers):
    p = vqa_norm(pred); return min(1.0, sum(p == vqa_norm(a) for a in answers) / 3.0)


def caption_items(n_pope, seed):
    caps = collections.defaultdict(list)
    for a in json.load(open(COCO))["annotations"]:
        caps[a["image_id"]].append(a["caption"].strip())
    rows = D._pope_rows(n_pope, random.Random(seed))          # same rows as mm_items(n_pope, seed)
    seen, items = set(), []
    for r in rows:
        src = r["image_source"]; iid = int(re.search(r"(\d+)$", src).group(1))
        if src in seen or iid not in caps:
            continue
        seen.add(src)
        items.append(dict(uid=f"cap_{src}", image=D._pil(r["bytes"]), captions=caps[iid][:5]))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(paths.MODELS / "Qwen3-VL-30B-A3B-Instruct"))
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mask", action="append", default=[]); ap.add_argument("--cond", action="append", default=[])
    ap.add_argument("--n_mm", type=int, default=120); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_pixels", type=int, default=1280 * 32 * 32); ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    masks = {}
    for spec in a.mask:
        name, path = spec.split("=", 1); masks[name] = np.array(json.load(open(path))["mask"], dtype=bool)
    conds = []
    for c in a.cond:
        if c == "none":
            conds.append(("none", None, "reroute", None))
        elif c.startswith("random:"):
            _, name, seed = c.split(":"); conds.append((f"random_{name}_{seed}", random_masks(masks[name], 100 + int(seed)), "reroute", None))
        else:
            parts = c.split(":"); name, mode = parts[0], parts[1]; scope = parts[2] if len(parts) > 2 else None
            conds.append((f"{name}_{mode}" + (f"_{scope}" if scope else ""), masks[name], mode, scope))
    caps = caption_items(a.n_mm, a.seed)
    tv = [it for it in D.mm_items(a.n_mm, a.seed) if it["source"] == "textvqa"]
    if a.limit:
        caps, tv = caps[:a.limit], tv[:a.limit]
    print(f"caption images={len(caps)} textvqa={len(tv)} conditions={[c[0] for c in conds]}", flush=True)
    index = out / "results.jsonl"
    done = {(json.loads(l)["uid"], json.loads(l)["condition"]) for l in open(index) if l.strip()} if index.exists() else set()
    m = load_runner(a.model, attn=a.attn, max_pixels=a.max_pixels)
    t0 = time.time(); n = 0
    with open(index, "a") as fo:
        for it in caps:
            for cname, mask, mode, scope in conds:
                if (it["uid"], cname) in done:
                    continue
                try:
                    m.set_ablation(mask, mode, scope)
                    res = [m.teacher_forced_nll(CAP_PROMPT, [it["image"]], c) for c in it["captions"]]
                    fo.write(json.dumps({"uid": it["uid"], "task": "caption", "condition": cname,
                                         "nll": float(np.mean([r["nll"] for r in res])), "n_captions": len(res),
                                         "n_tokens": int(sum(r["n_tokens"] for r in res))}) + "\n"); n += 1
                except Exception as e:
                    print(f"[FAIL] {it['uid']} {cname}: {type(e).__name__}: {e}", flush=True); traceback.print_exc()
            fo.flush(); m.set_ablation(None); it["image"] = None
            if n % 50 == 0:
                print(f"{n} caption evals, {time.time()-t0:.0f}s", flush=True)
        for it in tv:
            inputs = m.build_inputs(it["prompt"], it["images"], None)
            for cname, mask, mode, scope in conds:
                if (it["uid"], cname) in done:
                    continue
                try:
                    m.set_ablation(mask, mode, scope)
                    pred = m.generate(inputs, max_new_tokens=12)
                    res = m.forward(inputs, capture=False); sc = m.score(res["lp_last"], "free", None, it["gold"])
                    fo.write(json.dumps({"uid": it["uid"], "task": "textvqa", "condition": cname, "pred": pred, "gold": it["gold"],
                                         "soft_acc": vqa_soft(pred, it["meta"]["answers"]), "exact": vqa_norm(pred) == vqa_norm(it["gold"]),
                                         "gold_first_lp": sc["gold_lp"]}) + "\n"); n += 1
                except Exception as e:
                    print(f"[FAIL] {it['uid']} {cname}: {type(e).__name__}: {e}", flush=True); traceback.print_exc()
            fo.flush(); m.set_ablation(None); it["images"] = None
        print(f"done {n} evals in {time.time()-t0:.0f}s", flush=True)
    summarize(index, out / "summary.md")


def summarize(index, dst):
    rows, seen = [], set()
    for l in open(index):
        if l.strip():
            r = json.loads(l)
            if (r["uid"], r["condition"]) not in seen:
                seen.add((r["uid"], r["condition"])); rows.append(r)
    lines = ["| task | condition | n | caption NLL (nats/tok) | d vs none | TextVQA soft acc | exact | d soft vs none |", "|---|---|---|---|---|---|---|---|"]
    for task in ("caption", "textvqa"):
        rs = [r for r in rows if r["task"] == task]
        conds = sorted({r["condition"] for r in rs}, key=lambda c: (c != "none", c))
        base = {r["uid"]: r for r in rs if r["condition"] == "none"}
        for c in conds:
            cr = [r for r in rs if r["condition"] == c]
            if task == "caption":
                nll = np.mean([r["nll"] for r in cr]); d = np.mean([r["nll"] - base[r["uid"]]["nll"] for r in cr if r["uid"] in base]) if c != "none" else 0.0
                lines.append(f"| caption | {c} | {len(cr)} | {nll:.3f} | {d:+.3f} |  |  |  |")
            else:
                sa = np.mean([r["soft_acc"] for r in cr]); ex = np.mean([r["exact"] for r in cr])
                d = np.mean([r["soft_acc"] - base[r["uid"]]["soft_acc"] for r in cr if r["uid"] in base]) if c != "none" else 0.0
                lines.append(f"| textvqa | {c} | {len(cr)} |  |  | {sa:.3f} | {ex:.3f} | {d:+.3f} |")
    Path(dst).write_text("\n".join(lines)); print("\n".join(lines))


if __name__ == "__main__":
    main()
