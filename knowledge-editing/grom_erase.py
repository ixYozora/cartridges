#!/usr/bin/env python
"""GROM teacher-erase for AKEW CounterFact.

Erases the ORIGINAL (pre-edit) facts from the Self-Study teacher with GROM's
closed-form, gradient-free weight edit, so the teacher can no longer reconcile a
counterfactual context with its parametric memory. The ~13% of synthesized targets
that assert the old fact (`fidelity_filter.py` drops them today) are the thing this
is meant to remove at the source.

The closed form, the specificity weight and the key collection are imported from the
authors' package (`knowledge-editing/GROM/`, MIT, github.com/Batorskq/GROM) rather
than reimplemented. What lives here is the adaptation to our setting:

  * data       -- forget/retain requests built from AKEW CounterFact's
                  `requested_rewrite`, which already carries GROM's expected
                  (prompt, subject, target_true) triple.
  * alpha      -- GROM's specificity weight is computed from the DISJOINT held-out
                  facts only, never from the neighborhood prompts. CounterFact
                  neighborhood prompts are other subjects whose *correct* answer is
                  the same `target_true`, so counting them suppresses the suppression:
                  measured over a 50-edit split, mean alpha is 0.68 from disjoint
                  facts alone, 0.44 once neighborhood prompts are mixed in, and
                  exactly 0.0000 (100/100 forget positions dead, edit is a total
                  no-op) when they are the only retain source. That last case is not
                  hypothetical -- it is Step 4, where all 975 edits are erased and no
                  disjoint facts remain. Neighborhood prompts still enter the ridge
                  system as retain KEYS, which is where locality is actually
                  preserved.
  * key form   -- keys are collected in the plain completion form (the
                  CounterFact/ROME convention GROM's ZsRE plugin uses) and/or in the
                  chat form our teacher is actually prompted in. The chat form is
                  built as (generation prompt) + (answer tokens) so the gold tokens
                  are the answer alone: Qwen3's template with enable_thinking=False
                  ends in an empty <think></think> block, which is precisely the
                  state the teacher generates from during synthesis.
  * head first -- GROM's fact-editing (ZsRE) configuration edits the LM HEAD with
                  beta_mlp=0; the MLP band is the TOFU/MUSE setting. Both are
                  available here, and the MLP loop is skipped entirely when
                  beta_mlp == 0 instead of solving a d_inter x d_inter system for a
                  guaranteed-zero update.

Usage:
    # login node, no GPU: data/alpha/tokenization sanity check
    python grom_erase.py --dry-run --num-forget 50

    # the edit itself
    python grom_erase.py --num-forget 50 --seed 1 \
        --out ../checkpoints/qwen3-4b-erased-n50

Writes a HF checkpoint (only the edited matrices differ from the base model, so it
loads with a plain AutoModelForCausalLM and is served by tokasaurus unchanged) plus a
`grom_report.json` recording every knob and diagnostic of the run.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

KE_DIR = Path(__file__).resolve().parent
GROM_DIR = KE_DIR / "GROM"
if not GROM_DIR.exists():
    raise SystemExit(
        f"GROM checkout not found at {GROM_DIR}.\n"
        "  git clone https://github.com/Batorskq/GROM.git knowledge-editing/GROM"
    )
sys.path.insert(0, str(GROM_DIR))

from grom.keys import forward_collect            # noqa: E402
from grom.solver import solve_update             # noqa: E402
from grom.targets import (specificity_alpha,     # noqa: E402
                          suppression_D, unit_unembeddings)

DEFAULT_MODEL = "Qwen/Qwen3-4b"
DEFAULT_DATA = KE_DIR / "samples" / "CounterFact.json"
# Matches the system prompt family the teacher is served under during synthesis.
DEFAULT_SYSTEM = "You are a helpful assistant."


# --------------------------------------------------------------------------- data


def load_entries(path):
    with open(path) as f:
        return json.load(f)


def split_forget_retain(entries, num_forget, seed):
    """Deterministic forget/retain split. num_forget <= 0 means "all edits"."""
    idx = list(range(len(entries)))
    random.Random(seed).shuffle(idx)
    if num_forget <= 0 or num_forget >= len(entries):
        return [entries[i] for i in idx], []
    return [entries[i] for i in idx[:num_forget]], [entries[i] for i in idx[num_forget:]]


def request_pair(entry):
    """(prompt, answer) for the ORIGINAL fact -- what unlearning must suppress."""
    rr = entry["requested_rewrite"]
    prompt = rr["prompt"].format(rr["subject"])
    tgt = rr["target_true"]
    ans = tgt["str"] if isinstance(tgt, dict) else str(tgt)
    return prompt, (ans if ans.startswith(" ") else " " + ans)


def neighborhood_pairs(entry):
    """(neighborhood prompt, target_true) pairs.

    CounterFact builds these from other subjects that genuinely hold the attribute,
    so the gold answer is the same `target_true` we are suppressing -- the hardest
    and most important locality anchor. Retain KEYS only; see the alpha note above.
    """
    _, ans = request_pair(entry)
    return [(p, ans) for p in entry.get("neighborhood_prompts", [])]


def attribute_pairs(entry):
    """(attribute prompt, target_NEW) pairs.

    CounterFact `attribute_prompts` are other subjects that genuinely hold the NEW
    attribute, so the gold token is target_new. That makes them the right alpha source
    when no held-out edits remain (the student erase covers all 975): they keep alpha
    HIGH on the target_true tokens being suppressed -- unlike neighborhood prompts,
    whose gold IS target_true and which collapse alpha to 0 -- while also protecting
    the knowledge the edit is supposed to install.
    """
    rr = entry["requested_rewrite"]
    tgt = rr["target_new"]
    ans = tgt["str"] if isinstance(tgt, dict) else str(tgt)
    ans = ans if ans.startswith(" ") else " " + ans
    return [(p, ans) for p in entry.get("attribute_prompts", [])]


def subject_of(entry):
    return entry["requested_rewrite"]["subject"]


# -------------------------------------------------------------------- tokenizing


def build_plain(tok, prompt, answer):
    """Bare completion form (the CounterFact/ROME convention)."""
    pids = tok(prompt, add_special_tokens=True).input_ids
    ids = tok(prompt + answer, add_special_tokens=True).input_ids
    return ids, len(pids)


def build_chat(tok, prompt, answer, system):
    """Chat form: generation prompt + the assistant RESTATING the stem + the answer.

    The naive version (generation prompt + answer) puts the key at the assistant's
    very first token, where an instruction-tuned model opens a sentence -- measured on
    Qwen3-4b it answers "It" essentially always, gold median rank ~28,800. That is the
    wrong state twice over: it is not where the model emits the fact, so it is neither
    a useful key nor a meaningful probe.

    Our teacher asserts the old fact MID-SENTENCE, having already restated the stem
    ("The twin city of Wellington is Sydney"). So the assistant turn is built as
    stem + answer and only the answer tokens are gold. Built by concatenation rather
    than by rendering a full three-turn chat, which would drag the template's
    <think></think> scaffold and <|im_end|> into the suppressed span.
    """
    chat = [{"role": "system", "content": system},
            {"role": "user", "content": prompt}]
    try:
        pids = tok.apply_chat_template(chat, tokenize=True, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:                       # template without a thinking switch
        pids = tok.apply_chat_template(chat, tokenize=True, add_generation_prompt=True)
    stem_ids = tok(prompt, add_special_tokens=False).input_ids
    ans_ids = tok(answer, add_special_tokens=False).input_ids
    return list(pids) + list(stem_ids) + list(ans_ids), len(pids) + len(stem_ids)


def tokenize_pairs(tok, pairs, forms, system, want_gold=True):
    """-> (ids_list, prompt_lens, gold_lists). One sequence per (pair, form)."""
    ids_list, lp_list, gold_list = [], [], []
    for prompt, answer in pairs:
        for form in forms:
            if form == "plain":
                ids, lp = build_plain(tok, prompt, answer)
            elif form == "chat":
                ids, lp = build_chat(tok, prompt, answer, system)
            else:
                raise ValueError(f"unknown key form: {form}")
            if len(ids) <= lp:              # answer tokenized to nothing
                continue
            ids_list.append(ids)
            lp_list.append(lp)
            if want_gold:
                gold_list.append([ids[p + 1] for p in range(lp - 1, len(ids) - 1)])
    return ids_list, lp_list, (gold_list if want_gold else None)


def subject_last_keys(tok, entries, forms, system):
    """One key position per request, at the final subject token (ROME/MEMIT/GROM-ZsRE
    convention). Returns (ids_list, pos_list)."""
    ids_list, pos_list = [], []
    for e in entries:
        rr = e["requested_rewrite"]
        subject = rr["subject"]
        filled = rr["prompt"].format(subject)
        cut = filled.find(subject) + len(subject)
        for form in forms:
            if form == "plain":
                prefix = tok(filled[:cut], add_special_tokens=True).input_ids
                full = tok(filled, add_special_tokens=True).input_ids
            else:
                prefix, _ = build_chat(tok, filled[:cut], "", system)
                full, _ = build_chat(tok, filled, "", system)
            if not full:
                continue
            ids_list.append(full)
            pos_list.append(max(min(len(prefix) - 1, len(full) - 1), 0))
    return ids_list, pos_list


@torch.no_grad()
def collect_at_positions(model, tok, ids_list, pos_list, device, module, bs=16):
    """Capture `module`'s input at one explicit position per sequence -> (feat_dim, N)."""
    cap = {}
    handle = module.register_forward_pre_hook(lambda m, inp: cap.__setitem__("x", inp[0]))
    feats = []
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for i in range(0, len(ids_list), bs):
        chunk, ps = ids_list[i:i + bs], pos_list[i:i + bs]
        width = max(len(x) for x in chunk)
        inp = torch.full((len(chunk), width), pad, dtype=torch.long)
        att = torch.zeros((len(chunk), width), dtype=torch.long)
        for j, ids in enumerate(chunk):
            inp[j, :len(ids)] = torch.tensor(ids)
            att[j, :len(ids)] = 1
        model.model(input_ids=inp.to(device), attention_mask=att.to(device))
        for j in range(len(chunk)):
            feats.append(cap["x"][j, ps[j], :].float().cpu().unsqueeze(0))
    handle.remove()
    return torch.cat(feats, 0).T.contiguous()


def build_alpha(tok, forget_golds, retain_pairs, vocab_size, device, enabled=True):
    """GROM's specificity weight, computed from DISJOINT retain facts only.

    Returns (alpha, mean_alpha_over_forget_positions).
    """
    flat = torch.tensor([g for gs in forget_golds for g in gs])
    if not enabled:
        alpha = torch.ones(vocab_size, dtype=torch.double, device=device)
        return alpha, 1.0
    forget_counts = torch.bincount(flat, minlength=vocab_size).double()
    retain_counts = torch.zeros(vocab_size, dtype=torch.double)
    for _, answer in retain_pairs:
        for t in tok(answer, add_special_tokens=False).input_ids:
            if t < vocab_size:
                retain_counts[t] += 1
    alpha = specificity_alpha(forget_counts, retain_counts).to(device)
    return alpha, float(alpha[flat.to(device)].mean())


# ------------------------------------------------------------------------- edit


@torch.no_grad()
def probe_recall(model, tok, pairs, device, form, system, bs=8, limit=None):
    """Teacher-forced recall of the answer's first token at the position that predicts it.

    Reported PER KEY FORM and never averaged across forms: the two forms put the model
    in different states and mixing them destroys the measurement (an early version did,
    and the resulting "before" numbers were noise).

    MEDIAN gold rank is the headline. Top-1 is kept but is not the signal: on an
    instruction-tuned teacher the argmax at a mid-sentence position is usually
    punctuation or markdown, so top-1 reads ~0% even when the fact sits at rank 5.
    """
    ids_list, lp_list, gold_list = tokenize_pairs(tok, pairs, [form], system)
    if limit and len(ids_list) > limit:
        ids_list, lp_list, gold_list = ids_list[:limit], lp_list[:limit], gold_list[:limit]
    if not ids_list:
        return {"n": 0}
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    top1 = 0
    logprobs, ranks = [], []
    for i in range(0, len(ids_list), bs):
        lps = lp_list[i:i + bs]
        golds = [g[0] for g in gold_list[i:i + bs]]
        prompts = [ids[:lp] for ids, lp in zip(ids_list[i:i + bs], lps)]
        width = max(len(x) for x in prompts)
        inp = torch.full((len(prompts), width), pad, dtype=torch.long)
        att = torch.zeros((len(prompts), width), dtype=torch.long)
        for j, ids in enumerate(prompts):
            inp[j, :len(ids)] = torch.tensor(ids)
            att[j, :len(ids)] = 1
        logits = model(input_ids=inp.to(device), attention_mask=att.to(device)).logits
        for j, ids in enumerate(prompts):
            dist = torch.log_softmax(logits[j, len(ids) - 1].float(), dim=-1)
            g = golds[j]
            top1 += int(dist.argmax().item() == g)
            logprobs.append(float(dist[g]))
            ranks.append(int((dist > dist[g]).sum().item()) + 1)
        del logits
    n = len(ids_list)
    ranks.sort()
    return {"n": n,
            "median_gold_rank": ranks[n // 2],
            "mean_gold_logprob": round(sum(logprobs) / n, 4),
            "mean_gold_rank": round(sum(ranks) / n, 1),
            "top1_is_gold_pct": round(100.0 * top1 / n, 2)}


def _grom_attribution_fn():
    """The authors' logit-lens layer scorer, loaded from their scripts/ directory.

    Imported rather than reimplemented so the band we pick for Qwen3-4b is chosen by
    exactly the procedure that picked the bands in their configs.
    """
    import importlib.util
    src = GROM_DIR / "scripts" / "layer_attribution.py"
    spec = importlib.util.spec_from_file_location("grom_layer_attribution", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.direct_logit_effect


def run_attribution(model, tok, device, forget_seqs, retain_seqs, alpha, k):
    """Score every layer by how much it writes the forget gold tokens vs the retain ones.

    S_L = F_L - R_L. Reports the best contiguous band of width k. Needed because the
    published bands are Llama's: tofu10 used [11-15] of 16 layers, the ZsRE hparams
    [14-18] of 28. Qwen3-4b has 36, so neither transfers.
    """
    direct_logit_effect = _grom_attribution_fn()
    WU = model.lm_head.weight.detach()
    n_layers = len(model.model.layers)
    fi, fl, fg = forget_seqs
    ri, rl, rg = retain_seqs
    F = direct_logit_effect(model, tok, fi, fl, fg, WU, n_layers, device, alpha=alpha)
    R = direct_logit_effect(model, tok, ri, rl, rg, WU, n_layers, device, alpha=None)
    S = F - R
    print(f"\n{'layer':>5} {'F_L':>10} {'R_L':>10} {'S_L':>10}")
    for L in range(n_layers):
        print(f"{L:>5} {F[L]:>10.3f} {R[L]:>10.3f} {S[L]:>10.3f}")
    best = max(range(n_layers - k + 1), key=lambda a: S[a:a + k].mean())
    band = list(range(best, best + k))
    print(f"\nbest contiguous band of width {k}: {band} "
          f"(mean S_L {S[best:best+k].mean():.3f})")
    print(f"pass it as: --beta-mlp <strength> --layers {','.join(map(str, band))}")
    return {"per_layer_S": [round(float(x), 4) for x in S], "best_band": band,
            "band_width": k}


GEN_PROBE_PROMPTS = [
    "List exactly 3 short, well-known facts about Paris. Number them 1-3.",
    "Explain in two sentences why the sky appears blue.",
    "Who wrote Pride and Prejudice, and in what century?",
]


@torch.no_grad()
def gen_probe(model, tok, device, system, max_new_tokens=60):
    """Free-form fluency canary: greedy generations plus a degeneracy statistic.

    The recall probes only cover FACTUAL collateral. An over-strong edit shows up as
    broken generation instead -- repetition loops, refusal to stop -- which the rank
    probes would not catch. `rep4` is the fraction of repeated 4-grams; healthy text
    on these prompts sits near 0.
    """
    out = []
    for prompt in GEN_PROBE_PROMPTS:
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": prompt}]
        try:
            ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                          enable_thinking=False, return_tensors="pt")
        except TypeError:
            ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                          return_tensors="pt")
        ids = ids.to(device)
        gen = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        new = gen[0, ids.shape[1]:].tolist()
        grams = [tuple(new[i:i + 4]) for i in range(max(0, len(new) - 3))]
        rep4 = round(1 - len(set(grams)) / len(grams), 3) if grams else 0.0
        out.append({"prompt": prompt, "n_tokens": len(new), "rep4": rep4,
                    "text": tok.decode(new, skip_special_tokens=True)[:220]})
    return out


def apply_erase(model, tok, device, forget_seqs, retain_seqs, alpha, args, report):
    """GROM's closed-form update on the LM head and/or an MLP band, in place."""
    t0 = time.time()
    vocab_size, d_model = model.lm_head.weight.shape

    # Untie before touching the head: with tied weights a head edit would silently
    # rewrite the input embeddings too. Cloning also un-aliases the tensors, which is
    # what makes save_pretrained keep lm_head.weight in the checkpoint.
    model.lm_head.weight = nn.Parameter(model.lm_head.weight.detach().clone())
    model.config.tie_word_embeddings = False

    fi, fl, fg = forget_seqs
    ri, rl, _ = retain_seqs
    flat_gold = torch.tensor([g for gs in fg for g in gs])
    report["edits"] = []

    if args.beta_mlp > 0 and args.layers:
        uhat = unit_unembeddings(model)
        subj = args.mlp_key_pos == "subject_last"
        for layer in args.layers:
            down_proj = model.model.layers[layer].mlp.down_proj
            if subj:
                Kf = collect_at_positions(model, tok, *report["_subj_forget"],
                                          device, down_proj, args.batch_size)
                Kr = collect_at_positions(model, tok, *report["_subj_retain"],
                                          device, down_proj, args.batch_size)
                gold = report["_subj_gold"].to(device)
            else:
                Kf, idx = forward_collect(model, tok, fi, fl, device, module=down_proj,
                                          bs=args.batch_size, max_cols=args.forget_cols)
                Kr, _ = forward_collect(model, tok, ri, rl, device, module=down_proj,
                                        bs=args.batch_size, max_cols=args.retain_cols)
                gold = (flat_gold[idx] if idx is not None else flat_gold).to(device)
            D = suppression_D(uhat, alpha, gold, args.beta_mlp)
            P = solve_update(Kf, Kr, args.w_r, args.rho, device, D=D)
            down_proj.weight.data.add_(P.to(down_proj.weight.device,
                                            down_proj.weight.dtype))
            entry = {"matrix": f"layers.{layer}.mlp.down_proj",
                     "forget_keys": Kf.shape[1], "retain_keys": Kr.shape[1],
                     "P_fro": float(P.float().norm())}
            report["edits"].append(entry)
            print(f"[{time.time()-t0:.0f}s] edited {entry['matrix']}: "
                  f"Kf={tuple(Kf.shape)} Kr={tuple(Kr.shape)} "
                  f"||P||_F={entry['P_fro']:.3f}", flush=True)
        del uhat
        torch.cuda.empty_cache() if device.startswith("cuda") else None

    if args.beta_head > 0:
        Hf, idx = forward_collect(model, tok, fi, fl, device, module=None,
                                  bs=args.batch_size, max_cols=args.forget_cols)
        Hr, _ = forward_collect(model, tok, ri, rl, device, module=None,
                                bs=args.batch_size, max_cols=args.retain_cols)
        gold = (flat_gold[idx] if idx is not None else flat_gold).to(device)
        s = Hf.shape[1]
        # B = D Hf^T / s for the sparse target D[:, j] = -beta*alpha_j*e_{g_j},
        # accumulated directly instead of materialising a (vocab, s) matrix.
        sdev = args.solve_device or device
        gold_s = gold.to(sdev)
        Bsup = torch.zeros(vocab_size, d_model, dtype=torch.float64, device=sdev)
        Hf64 = Hf.double().to(sdev)
        Bsup.index_add_(0, gold_s,
                        Hf64.T * (-(alpha[gold].double().to(sdev)
                                    * args.beta_head / s).unsqueeze(1)))
        del Hf64
        P = solve_update(Hf, Hr, args.w_r, args.rho, sdev, B=Bsup)
        del Bsup
        model.lm_head.weight.data.add_(P.to(model.lm_head.weight.device,
                                            model.lm_head.weight.dtype))
        entry = {"matrix": "lm_head", "forget_keys": Hf.shape[1],
                 "retain_keys": Hr.shape[1], "P_fro": float(P.float().norm())}
        report["edits"].append(entry)
        print(f"[{time.time()-t0:.0f}s] edited lm_head: Hf={tuple(Hf.shape)} "
              f"Hr={tuple(Hr.shape)} ||P||_F={entry['P_fro']:.3f}", flush=True)

    report["edit_seconds"] = round(time.time() - t0, 1)
    for key in ("_subj_forget", "_subj_retain", "_subj_gold"):
        report.pop(key, None)
    print(f"[{time.time()-t0:.0f}s] erase complete", flush=True)


# ------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-file", default=str(DEFAULT_DATA))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default=None, help="output checkpoint dir (required unless --dry-run)")
    ap.add_argument("--num-forget", type=int, default=50,
                    help="edits to erase; <=0 erases all (Step 4)")
    ap.add_argument("--retain-facts", type=int, default=1000,
                    help="held-out edits used as the disjoint fact anchor (ZsRE uses 1000)")
    ap.add_argument("--neighborhood-retain", action="store_true", default=True,
                    help="also anchor on the forget edits' neighborhood prompts (default on)")
    ap.add_argument("--no-neighborhood-retain", dest="neighborhood_retain",
                    action="store_false")
    ap.add_argument("--wiki-jsonl", default=None,
                    help="optional broad-utility anchor, one {'text': ...} per line")
    ap.add_argument("--wiki-lines", type=int, default=0)
    ap.add_argument("--wiki-maxlen", type=int, default=256)
    ap.add_argument("--key-forms", default="plain,chat",
                    help="comma-separated: plain (CounterFact convention) and/or chat "
                         "(how our teacher is actually prompted)")
    ap.add_argument("--chat-system", default=DEFAULT_SYSTEM)
    # GROM knobs; defaults are the authors' ZsRE fact-editing configuration.
    ap.add_argument("--beta-head", type=float, default=20.0)
    ap.add_argument("--beta-mlp", type=float, default=0.0)
    ap.add_argument("--layers", default="", help="MLP band, e.g. 18,19,20,21,22 (beta-mlp>0 only)")
    ap.add_argument("--mlp-key-pos", choices=["answer", "subject_last"], default="subject_last")
    ap.add_argument("--w-r", type=float, default=30.0)
    ap.add_argument("--rho", type=float, default=0.03)
    ap.add_argument("--no-specificity", action="store_true",
                    help="disable GROM's alpha reweighting (alpha == 1)")
    ap.add_argument("--forget-cols", type=int, default=20000)
    ap.add_argument("--retain-cols", type=int, default=40000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dump-forget-subset", default=None, metavar="PATH",
                    help="write the forget edits as a CounterFact-shaped JSON and exit. "
                         "Produced by the SAME split as the erase, so a synthesis run "
                         "over this subset covers exactly the erased facts (Step 3e).")
    ap.add_argument("--max-retain-seqs", type=int, default=0, metavar="N",
                    help="subsample the retain SEQUENCES to N before key collection "
                         "(0 = all). Key collection is one forward pass per sequence, "
                         "so this is the knob that makes a sweep affordable.")
    ap.add_argument("--attribute-retain", action="store_true",
                    help="add CounterFact attribute_prompts (gold = target_new) to the "
                         "alpha/retain source. Automatic when no held-out edits remain.")
    ap.add_argument("--solve-device", default=None,
                    help="device for the closed-form solve (default: --device). Use cpu "
                         "for 7B+: the head target and its solve are ~4.4 GB each in "
                         "float64 at vocab 152064 x d 3584, which will not fit beside a "
                         "bf16 7B on one 32 GB card. The solve is seconds on CPU.")
    ap.add_argument("--attribution", type=int, default=0, metavar="K",
                    help="score every layer by logit-lens attribution and report the "
                         "best contiguous band of width K, then exit (no edit)")
    ap.add_argument("--no-save", action="store_true",
                    help="apply and measure the edit but write no checkpoint "
                         "(sweep mode: a Qwen3-4b checkpoint is ~8.8 GB)")
    ap.add_argument("--report-out", default=None,
                    help="where to write grom_report.json when --no-save is set")
    ap.add_argument("--probe-limit", type=int, default=200,
                    help="sequences per probe set for the before/after recall check")
    ap.add_argument("--no-probe", action="store_true",
                    help="skip the before/after recall measurement")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and report the data only; no model, no GPU")
    args = ap.parse_args()

    args.layers = [int(x) for x in args.layers.split(",") if x.strip()]
    forms = [f.strip() for f in args.key_forms.split(",") if f.strip()]
    if not (args.dry_run or args.dump_forget_subset or args.attribution
            or args.no_save or args.out):
        raise SystemExit("--out is required unless "
                         "--dry-run/--dump-forget-subset/--attribution/--no-save")

    entries = load_entries(args.data_file)
    forget_entries, retain_pool = split_forget_retain(entries, args.num_forget, args.seed)

    if args.dump_forget_subset:
        dest = Path(args.dump_forget_subset)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(forget_entries, indent=1))
        print(f"wrote {len(forget_entries)} forget edits -> {dest}")
        print("case_ids:", [e.get("case_id") for e in forget_entries[:10]], "...")
        return
    retain_entries = retain_pool[:args.retain_facts] if args.retain_facts > 0 else retain_pool

    forget_pairs = [request_pair(e) for e in forget_entries]
    retain_fact_pairs = [request_pair(e) for e in retain_entries]
    if args.attribute_retain or not retain_fact_pairs:
        # No disjoint edits left (or explicitly asked for): anchor on attribute prompts.
        for e in forget_entries:
            retain_fact_pairs += attribute_pairs(e)
    neigh_pairs = []
    if args.neighborhood_retain:
        for e in forget_entries:
            neigh_pairs += neighborhood_pairs(e)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    fi, fl, fg = tokenize_pairs(tok, forget_pairs, forms, args.chat_system)
    ri, rl, rg = tokenize_pairs(tok, retain_fact_pairs + neigh_pairs, forms,
                                args.chat_system, want_gold=True)
    n_fact_retain_seqs = len(ri)   # wiki lines are appended after this point
    if args.wiki_jsonl and args.wiki_lines > 0:
        n = 0
        with open(args.wiki_jsonl) as f:
            for line in f:
                text = (json.loads(line).get("text") or "").strip()
                if not text:
                    continue
                ids = tok(text, add_special_tokens=True, truncation=True,
                          max_length=args.wiki_maxlen).input_ids
                if len(ids) < 2:
                    continue
                ri.append(ids)
                rl.append(1)          # lp=1 -> every position is a retain key
                n += 1
                if n >= args.wiki_lines:
                    break

    if args.max_retain_seqs and len(ri) > args.max_retain_seqs:
        idx = random.Random(args.seed).sample(range(len(ri)), args.max_retain_seqs)
        ri = [ri[i] for i in idx]
        rl = [rl[i] for i in idx]
        rg = [rg[i] for i in idx]
        n_fact_retain_seqs = min(n_fact_retain_seqs, len(ri))

    vocab_size = len(tok)
    alpha_dev = "cpu" if args.dry_run else args.device
    alpha, mean_alpha = build_alpha(tok, fg, retain_fact_pairs, vocab_size, alpha_dev,
                                    enabled=not args.no_specificity)

    report = {
        "model": args.model, "data_file": args.data_file, "seed": args.seed,
        "num_forget_edits": len(forget_entries),
        "num_retain_facts": len(retain_entries),
        "num_alpha_retain_pairs": len(retain_fact_pairs),
        "attribute_retain": bool(args.attribute_retain or not retain_entries),
        "num_neighborhood_retain_prompts": len(neigh_pairs),
        "key_forms": forms,
        "forget_sequences": len(fi), "retain_sequences": len(ri),
        "forget_key_positions": sum(len(g) for g in fg),
        "specificity": not args.no_specificity, "mean_alpha": round(mean_alpha, 4),
        "beta_head": args.beta_head, "beta_mlp": args.beta_mlp, "layers": args.layers,
        "mlp_key_pos": args.mlp_key_pos, "w_r": args.w_r, "rho": args.rho,
    }
    print(json.dumps(report, indent=2), flush=True)

    if not args.no_specificity and not retain_fact_pairs:
        print("\nWARNING: no disjoint retain facts, so the specificity weight "
              "degenerates to alpha == 1 everywhere and protects nothing. At full "
              "scale (--num-forget 0) the wiki anchor is not optional: pass "
              "--wiki-jsonl/--wiki-lines, or accept an unweighted edit.", flush=True)
    if mean_alpha < 0.2:
        print("\nWARNING: mean alpha is very low -- the specificity weight is cancelling "
              "the suppression. Check that neighborhood prompts did not leak into the "
              "alpha computation, or pass --no-specificity.", flush=True)

    if args.dry_run:
        # Reproduce the alpha-source finding that motivates keeping neighborhood
        # prompts out of the specificity weight (see the module docstring).
        print("\n--- specificity weight by retain source ---")
        flat = torch.tensor([g for gs in fg for g in gs])
        alpha_src = "attribute prompts" if not retain_entries else "disjoint facts"
        for label, src in ((f"{alpha_src} only (this script)", retain_fact_pairs),
                           ("+ neighborhood prompts", retain_fact_pairs + neigh_pairs),
                           ("neighborhood prompts only", neigh_pairs)):
            if not src:
                continue
            a, m = build_alpha(tok, fg, src, vocab_size, "cpu")
            dead = int((a[flat] < 0.05).sum())
            print(f"  {label:34s} mean_alpha={m:.4f}  dead forget positions: "
                  f"{dead}/{len(flat)}")
        print("\n--- sample forget sequence ---")
        print(repr(tok.decode(fi[0])))
        print(f"prompt_len={fl[0]}  gold={[tok.decode([g]) for g in fg[0]]}")
        if len(forms) > 1:
            print("\n--- same fact, second key form ---")
            print(repr(tok.decode(fi[1])))
            print(f"prompt_len={fl[1]}  gold={[tok.decode([g]) for g in fg[1]]}")
        print("\n--dry-run: no model loaded, nothing written.")
        return

    from transformers import AutoModelForCausalLM
    print(f"loading {args.model} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model = model.to(args.device).eval()

    if args.attribution:
        # Score on the fact-based retain only (wiki windows carry no gold tokens).
        # Cap the retain side as the authors' script does (--retain_n 600): scoring
        # holds a hooked output for all 36 layers, so the full retain set is wasteful.
        cap = min(n_fact_retain_seqs, 600)
        att = run_attribution(model, tok, args.device, (fi, fl, fg),
                              (ri[:cap], rl[:cap], rg[:cap]), alpha, args.attribution)
        report["attribution"] = att
        if args.report_out:
            Path(args.report_out).write_text(json.dumps(report, indent=2))
            print(f"report -> {args.report_out}")
        return

    if not args.no_probe:
        report["gen_before"] = gen_probe(model, tok, args.device, args.chat_system)

    if args.beta_mlp > 0 and args.mlp_key_pos == "subject_last":
        report["_subj_forget"] = subject_last_keys(tok, forget_entries, forms, args.chat_system)
        report["_subj_retain"] = subject_last_keys(tok, retain_entries, forms, args.chat_system)
        report["_subj_gold"] = torch.tensor(
            [g[0] for g in fg for _ in range(1)][:len(report["_subj_forget"][0])])

    probe_sets = {"forget": forget_pairs,
                  "retain_facts": retain_fact_pairs,
                  "neighborhood": neigh_pairs}
    if not args.no_probe:
        report["recall_before"] = {}
        for name, pairs in probe_sets.items():
            for form in forms:
                if not pairs:
                    continue
                key = f"{name}/{form}"
                report["recall_before"][key] = probe_recall(
                    model, tok, pairs, args.device, form, args.chat_system,
                    limit=args.probe_limit)
                print(f"  before  {key:22s} {report['recall_before'][key]}", flush=True)

    apply_erase(model, tok, args.device, (fi, fl, fg), (ri, rl, None), alpha, args, report)

    if not args.no_probe:
        report["recall_after"] = {}
        print("\n--- median gold rank, before -> after "
              "(forget should blow up, retain should hold) ---", flush=True)
        for name, pairs in probe_sets.items():
            for form in forms:
                if not pairs:
                    continue
                key = f"{name}/{form}"
                after = probe_recall(model, tok, pairs, args.device, form,
                                     args.chat_system, limit=args.probe_limit)
                report["recall_after"][key] = after
                before = report["recall_before"][key]
                print(f"  {key:22s} rank {before['median_gold_rank']:7d} -> "
                      f"{after['median_gold_rank']:7d}    logprob "
                      f"{before['mean_gold_logprob']:8.3f} -> "
                      f"{after['mean_gold_logprob']:8.3f}", flush=True)

    if not args.no_probe:
        report["gen_after"] = gen_probe(model, tok, args.device, args.chat_system)
        worst = max(g["rep4"] for g in report["gen_after"])
        print(f"\n--- fluency canary: worst repeated-4gram fraction after edit = "
              f"{worst:.3f} (healthy ~0) ---", flush=True)
        for g in report["gen_after"]:
            print(f"  rep4={g['rep4']:.3f} n={g['n_tokens']:3d} | {g['text'][:110]!r}",
                  flush=True)

    if args.no_save:
        dest = Path(args.report_out or "grom_report.json")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(report, indent=2))
        print(f"\n--no-save: checkpoint not written; report -> {dest}", flush=True)
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "grom_report.json").write_text(json.dumps(report, indent=2))
    print(f"saved erased teacher to {out}", flush=True)
    print(f"report -> {out / 'grom_report.json'}", flush=True)


if __name__ == "__main__":
    main()
