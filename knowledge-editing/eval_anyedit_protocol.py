"""Generate under AnyEdit's exact protocol, then score with their metric.

Our own eval (lora_eval.py) differs from AnyEdit's long-form setup in ways that all
inflate our side, so its responses cannot be put next to their published table. This
script reproduces their generation settings instead, from dsets/counterfact.py and
experiments/evaluate_uns.py:

  prompt     get_qwen_without_answer(q) = "<|im_start|>user\\n{q}<|im_end|>\\n<|im_start|>assistant\\n"
             -- a bare user turn, NO system prompt (apply_chat_template would inject
             Qwen's default "You are Qwen, created by Alibaba Cloud ..." preamble,
             which makes answers markedly chattier and so scores higher)
  Ori.       requested_rewrite.prompt_full
  Para.      paraphrase_prompts[0]            (they use the first one only)
  sampling   max_new_tokens=512, temperature=0.001, do_sample=True
  reference  requested_rewrite.fact_new_uns + '<|im_end|>'   (see score_anyedit)

What still differs after this, and cannot be fixed by re-generating: they apply ONE
edit to a freshly reset model (batch_size = num_edits = 1) where we hold 975 edits at
once, and their edit target is the paragraph itself where ours is a short answer.

    python eval_anyedit_protocol.py \\
        --config stock  Qwen/Qwen2.5-7B-Instruct \\
        --config ins320 checkpoints/qwen2.5-7b-ke-ins320 checkpoints/lora-on-ke-ins320 \\
        --out results/anyedit-protocol-<stamp>.json
"""

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from score_anyedit import MiniLM, hit_rates, load_counterfact, load_targets, print_header, print_row

KE_DIR = Path(__file__).resolve().parent
REPO = KE_DIR.parent

QWEN_TMPL = "<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"


def resolve(p):
    """Absolute path as-is; repo-relative path expanded; anything else is an HF repo id."""
    if Path(p).is_absolute():
        return p
    return str(REPO / p) if (REPO / p).exists() else p


@torch.no_grad()
def generate(model, tok, prompts, device, batch_size, max_new_tokens, temperature):
    """Batched greedy-ish generation. Left padding so the completions start aligned."""
    out = []
    t0 = time.time()
    for i in range(0, len(prompts), batch_size):
        chunk = [QWEN_TMPL.format(q=q) for q in prompts[i:i + batch_size]]
        b = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        gen = model.generate(**b, max_new_tokens=max_new_tokens, temperature=temperature,
                             do_sample=True, pad_token_id=tok.pad_token_id)
        for j in range(len(chunk)):
            out.append(tok.decode(gen[j][b["input_ids"].shape[1]:], skip_special_tokens=True).strip())
        if (i // batch_size) % 10 == 0:
            done = min(i + batch_size, len(prompts))
            print(f"     {done}/{len(prompts)}  ({time.time() - t0:.0f}s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", nargs="+", action="append", required=True,
                    metavar=("NAME BASE", "ADAPTER"), help="NAME BASE [ADAPTER]; repeatable")
    ap.add_argument("--data-file", default=str(KE_DIR / "samples" / "CounterFact.json"))
    ap.add_argument("--case-ids-file", help="JSON list of case_ids to evaluate "
                                            "(scaling curve: score only the edits this adapter was trained on)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.001)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    refs, para0, prompt_full = load_counterfact(args.data_file)
    ids = sorted(refs)
    if args.case_ids_file:
        keep = set(json.loads(Path(resolve(args.case_ids_file)).read_text()))
        missing = keep - set(ids)
        if missing:
            ap.error(f"{len(missing)} case_ids not in {args.data_file}, e.g. {sorted(missing)[:5]}")
        ids = [i for i in ids if i in keep]
    if args.limit:
        ids = ids[:args.limit]
    ori_prompts = [prompt_full[i] for i in ids]
    par_prompts = [para0[i] for i in ids]
    ref_texts = [refs[i] for i in ids]
    all_targets = load_targets(args.data_file)
    tgt = [all_targets[i] for i in ids]
    print(f"{len(ids)} edits | {2 * len(ids)} generations per config | "
          f"temp={args.temperature} max_new_tokens={args.max_new_tokens} | no system prompt")

    lm = MiniLM("cpu")  # 22M params; the GPU is busy holding the 7B
    out_path = Path(resolve(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"protocol": "anyedit", "data_file": args.data_file, "case_ids": ids,
              "temperature": args.temperature, "max_new_tokens": args.max_new_tokens,
              "system_prompt": None, "configs": {}}
    scored = {}

    for cfg in args.config:
        if len(cfg) not in (2, 3):
            ap.error(f"--config needs NAME BASE [ADAPTER], got {cfg}")
        name, base = cfg[0], resolve(cfg[1])
        adapter = resolve(cfg[2]) if len(cfg) == 3 else None
        print(f"\n== {name}: base={base} adapter={adapter}", flush=True)
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

        print("   Ori.", flush=True)
        ori = generate(model, tok, ori_prompts, args.device, args.batch_size,
                       args.max_new_tokens, args.temperature)
        print("   Para.", flush=True)
        par = generate(model, tok, par_prompts, args.device, args.batch_size,
                       args.max_new_tokens, args.temperature)

        ori_sims = lm.cosine(ori, ref_texts)
        par_sims = lm.cosine(par, ref_texts)
        ori_new, ori_old = hit_rates(ori, tgt)
        par_new, par_old = hit_rates(par, tgt)
        scored[name] = (ori_sims, par_sims)
        report["configs"][name] = {
            "base": base, "adapter": adapter,
            "ori": {"responses": ori, "sims": ori_sims, "names_new": ori_new, "names_old": ori_old},
            "para": {"responses": par, "sims": par_sims, "names_new": par_new, "names_old": par_old},
        }
        print(f"   Ori. {100 * sum(ori_sims) / len(ori_sims):.2f} "
              f"(names new {ori_new:.1f}% / old {ori_old:.1f}%) | "
              f"Para. {100 * sum(par_sims) / len(par_sims):.2f} "
              f"(names new {par_new:.1f}% / old {par_old:.1f}%)  "
              f"({time.time() - t0:.0f}s)", flush=True)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        out_path.write_text(json.dumps(report))  # checkpoint after every config

    print()
    print_header()
    for name, (o, p) in scored.items():
        print_row(name, o, p)
    print(f"\nwrote {out_path}")
    print("\nAnyEdit Table 1, AKEW (CounterFact), Qwen2.5-7B-Instruct, for reference:")
    print("  Pre-edited 65.50 / 44.74   MEMIT 77.19 / 56.04   AlphaEdit 80.66 / 56.99")
    print("  UnKE 97.34 / 59.29         AnyEdit 98.08 / 65.40  (Ori. / Para.)")
    print("  NOTE: theirs is one edit into a reset model; ours is 975 edits at once.")


if __name__ == "__main__":
    main()
