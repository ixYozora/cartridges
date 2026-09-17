"""Teacher-forced recall of the OLD and the NEW target for all CounterFact edits, per model.

Answers "did the GROM erase survive LoRA finetuning?": compare where the old target ranks
in the erased+LoRA student against the unerased+LoRA student, and against both bases.
Same measurement as grom_erase.probe_recall -- the rank of the answer's first token at the
position that predicts it -- but for both targets, in two prompt forms:

  plain  bare CounterFact completion ("What is the twin city of Wellington? It is")
  chat   the eval's own format: a user turn only (so the template's default system
         prompt, as in lora_eval.generate_response), then the assistant restating the
         stem, so the probed position is where the model emits the answer

    python probe_recall_lora.py \\
        --config stock   Qwen/Qwen2.5-7B-Instruct \\
        --config dct_ep5 Qwen/Qwen2.5-7B-Instruct checkpoints/lora_qwen2.5-7B-Instruct-20260812_002743 \\
        --out results/recall-probe.json
"""

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

KE_DIR = Path(__file__).resolve().parent


def build_plain(tok, prompt, answer):
    pids = tok(prompt, add_special_tokens=True).input_ids
    ids = tok(prompt + answer, add_special_tokens=True).input_ids
    return ids, len(pids)


def build_chat(tok, prompt, answer):
    pids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=True, add_generation_prompt=True)
    stem = tok(prompt, add_special_tokens=False).input_ids
    ans = tok(answer, add_special_tokens=False).input_ids
    return list(pids) + list(stem) + list(ans), len(pids) + len(stem)


def load_edits(path, limit=0):
    edits = []
    for e in json.loads(Path(path).read_text()):
        rw = e["requested_rewrite"]
        edits.append({"case_id": e["case_id"], "prompt": rw["prompt"].format(rw["subject"]),
                      "old": " " + rw["target_true"]["str"], "new": " " + rw["target_new"]["str"]})
    return edits[:limit] if limit else edits


@torch.no_grad()
def first_token_stats(model, tok, seqs, device, bs):
    """seqs: list of (ids, prompt_len). Returns per-seq (rank, logprob) of ids[prompt_len]."""
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    out = []
    for i in range(0, len(seqs), bs):
        chunk = seqs[i:i + bs]
        prompts = [ids[:lp] for ids, lp in chunk]
        width = max(len(p) for p in prompts)
        inp = torch.full((len(prompts), width), pad, dtype=torch.long)
        att = torch.zeros((len(prompts), width), dtype=torch.long)
        for j, p in enumerate(prompts):
            inp[j, :len(p)] = torch.tensor(p)
            att[j, :len(p)] = 1
        logits = model(input_ids=inp.to(device), attention_mask=att.to(device)).logits
        for j, (ids, lp) in enumerate(chunk):
            dist = torch.log_softmax(logits[j, lp - 1].float(), dim=-1)
            g = ids[lp]
            out.append((int((dist > dist[g]).sum().item()) + 1, float(dist[g])))
        del logits
    return out


def probe(model, tok, edits, device, bs):
    res = {}
    for form, build in (("plain", build_plain), ("chat", build_chat)):
        per = {}
        for side in ("old", "new"):
            seqs = [build(tok, e["prompt"], e[side]) for e in edits]
            per[side] = first_token_stats(model, tok, seqs, device, bs)
        summ = {}
        for side in ("old", "new"):
            ranks = [r for r, _ in per[side]]
            lps = [lp for _, lp in per[side]]
            summ[side] = {"median_rank": statistics.median(ranks),
                          "top1_pct": round(100 * sum(r == 1 for r in ranks) / len(ranks), 2),
                          "top10_pct": round(100 * sum(r <= 10 for r in ranks) / len(ranks), 2),
                          "mean_logprob": round(sum(lps) / len(lps), 4)}
        # Same prefix in both sides for nearly every edit, so the two first-token
        # log-probs come from one distribution and can be compared directly.
        summ["old_above_new_pct"] = round(
            100 * sum(o[1] > n[1] for o, n in zip(per["old"], per["new"])) / len(edits), 2)
        summ["per_edit"] = {"old_rank": [r for r, _ in per["old"]],
                            "new_rank": [r for r, _ in per["new"]]}
        res[form] = summ
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", nargs="+", action="append", required=True,
                    metavar=("NAME BASE", "ADAPTER"),
                    help="NAME BASE_MODEL [ADAPTER_DIR]; repeatable")
    ap.add_argument("--data-file", default=str(KE_DIR / "samples" / "CounterFact.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    edits = load_edits(args.data_file, args.limit)
    report = {"data_file": args.data_file, "n_edits": len(edits),
              "case_ids": [e["case_id"] for e in edits], "configs": {}}
    for cfg in args.config:
        if len(cfg) not in (2, 3):
            ap.error(f"--config needs NAME BASE [ADAPTER], got {cfg}")
        name, base, adapter = cfg[0], cfg[1], (cfg[2] if len(cfg) == 3 else None)
        t0 = time.time()
        print(f"\n== {name}: base={base} adapter={adapter}", flush=True)
        tok = AutoTokenizer.from_pretrained(adapter or base)
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16).to(args.device)
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter)
        model.eval()
        res = probe(model, tok, edits, args.device, args.batch_size)
        report["configs"][name] = {"base": base, "adapter": adapter, **res}
        for form in ("plain", "chat"):
            o, n = res[form]["old"], res[form]["new"]
            print(f"   {form:<5} OLD median rank {o['median_rank']:>8} top1 {o['top1_pct']:>6}% | "
                  f"NEW median rank {n['median_rank']:>8} top1 {n['top1_pct']:>6}% | "
                  f"old>new {res[form]['old_above_new_pct']}%  ({time.time() - t0:.0f}s)", flush=True)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report))  # checkpoint after every config

    print(f"\n{'config':<12}{'form':<7}{'OLD med rank':>13}{'OLD top1%':>10}{'NEW med rank':>13}"
          f"{'NEW top1%':>10}{'old>new%':>9}")
    for name, r in report["configs"].items():
        for form in ("plain", "chat"):
            o, n = r[form]["old"], r[form]["new"]
            print(f"{name:<12}{form:<7}{o['median_rank']:>13}{o['top1_pct']:>10}"
                  f"{n['median_rank']:>13}{n['top1_pct']:>10}{r[form]['old_above_new_pct']:>9}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
