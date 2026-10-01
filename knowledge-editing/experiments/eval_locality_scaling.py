"""Does collateral damage grow as more edits share one adapter?

The scaling curve (slurm/scaling_curve.sbatch) showed per-edit quality RISING with n:
the corpus teaches one shared behaviour -- "override what you know and assert the
taught fact" -- and more edits mean more evidence for it. Locality is that same
generalisation working against us: nothing in the corpus ever demonstrates "this
question is about a different subject, keep the original answer", so the override may
spill onto neighbours. At n=975 judge-v2 locality was 30.6%, but we have never
measured how it varies with n, and that decides whether it is a recipe problem or the
price of the scaling gain.

CounterFact gives 10 `neighborhood_prompts` per edit: different subjects, same
relation, whose correct answer is that edit's `target_true`. A well-scoped edit leaves
them alone.

  correct%  names target_true and NOT target_new  -- the neighbour survived
  bleed%    names target_new                      -- the edit leaked onto it
  neither%  names neither

Generation matches eval_anyedit_protocol.py exactly (bare Qwen user turn, no system
prompt, temperature 0.001) so locality and efficacy are measured on one instrument.

    python eval_locality_scaling.py \\
        --config n975 Qwen/Qwen2.5-7B-Instruct checkpoints/lora_.../checkpoint-2500 \\
        --out results/locality-scaling-<stamp>/n975.json
"""

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # moved out of knowledge-editing/; import its top-level modules
from eval_anyedit_protocol import generate, resolve
from score_anyedit import bootstrap

KE_DIR = Path(__file__).resolve().parents[1]   # knowledge-editing/


def load_neighborhood(path, per_edit):
    """case_id -> (prompts, gold_old, forbidden_new)."""
    out = {}
    for e in json.loads(Path(path).read_text()):
        rw = e["requested_rewrite"]
        out[e["case_id"]] = (e["neighborhood_prompts"][:per_edit],
                             rw["target_true"]["str"], rw["target_new"]["str"])
    return out


def classify(responses, gold, bad):
    """Per-response 1/0 for correct and bleed, as lists so they can be bootstrapped."""
    correct, bleed = [], []
    for r in responses:
        rl = r.lower()
        has_new = bad.lower() in rl
        correct.append(int(gold.lower() in rl and not has_new))
        bleed.append(int(has_new))
    return correct, bleed


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", nargs="+", action="append", required=True,
                    metavar=("NAME BASE", "ADAPTER"), help="NAME BASE [ADAPTER]; repeatable")
    ap.add_argument("--data-file", default=str(KE_DIR / "samples" / "CounterFact.json"))
    ap.add_argument("--case-ids-file", help="JSON list of case_ids; default is all of them")
    ap.add_argument("--per-edit", type=int, default=2,
                    help="neighborhood prompts per edit (CounterFact ships 10)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.001)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    nb = load_neighborhood(args.data_file, args.per_edit)
    ids = sorted(nb)
    if args.case_ids_file:
        keep = set(json.loads(Path(resolve(args.case_ids_file)).read_text()))
        ids = [i for i in ids if i in keep]

    prompts, golds, bads, owner = [], [], [], []
    for i in ids:
        ps, gold, bad = nb[i]
        for p in ps:
            prompts.append(p); golds.append(gold); bads.append(bad); owner.append(i)
    print(f"{len(ids)} edits x <={args.per_edit} neighbours = {len(prompts)} prompts | "
          f"temp={args.temperature} | no system prompt")

    out_path = Path(resolve(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"data_file": args.data_file, "case_ids": ids, "per_edit": args.per_edit,
              "owner": owner, "configs": {}}

    print(f"\n{'run':<14}{'prompts':>9}{'correct%':>10}{'95% CI':>18}{'bleed%':>9}{'95% CI':>18}")
    for cfg in args.config:
        if len(cfg) not in (2, 3):
            ap.error(f"--config needs NAME BASE [ADAPTER], got {cfg}")
        name, base = cfg[0], resolve(cfg[1])
        adapter = resolve(cfg[2]) if len(cfg) == 3 else None
        t0 = time.time()

        tok = AutoTokenizer.from_pretrained(adapter or base)
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16).to(args.device)
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter)
        model.eval()

        resp = generate(model, tok, prompts, args.device, args.batch_size,
                        args.max_new_tokens, args.temperature)
        correct, bleed = [], []
        for r, g, b in zip(resp, golds, bads):
            c, x = classify([r], g, b)
            correct += c; bleed += x

        cl, ch = bootstrap([float(v) for v in correct])
        bl, bh = bootstrap([float(v) for v in bleed])
        report["configs"][name] = {"base": base, "adapter": adapter, "responses": resp,
                                   "correct": correct, "bleed": bleed}
        n = len(correct)
        print(f"{name:<14}{n:>9}{100 * sum(correct) / n:>10.1f}"
              f"{f'[{100 * cl:.1f}, {100 * ch:.1f}]':>18}{100 * sum(bleed) / n:>9.1f}"
              f"{f'[{100 * bl:.1f}, {100 * bh:.1f}]':>18}   ({time.time() - t0:.0f}s)", flush=True)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        out_path.write_text(json.dumps(report))

    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
