"""AnyEdit/UnKE-protocol scoring: SBERT cosine against the AKEW unstructured fact.

AnyEdit's "Bert Score" column is NOT the bert_score library. From their
experiments/summarize_uns.py:

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    temp_original['Bert Score'] = cosine_scores.diagonal().mean().item()

so it is the cosine similarity of two MiniLM sentence embeddings, where the
reference is requested_rewrite.fact_new_uns (the ~74-word AKEW paragraph) and
the prediction is the model's free generation. Their dsets/counterfact.py:

    "question":      get_qwen_without_answer(record["requested_rewrite"]["prompt_full"]),
    "para_question": get_qwen_without_answer(record["paraphrase_prompts"][0]),
    "answer":        record["requested_rewrite"]["fact_new_uns"] + '<|im_end|>',

MiniLM is reimplemented here on plain transformers (mean-pool over the attention
mask -> L2 normalise, max_seq_length 256), which is exactly what SentenceTransformer
does for this model, so sentence-transformers is not a dependency.

Two entry points:

  rescore stored eval.json runs (no GPU, no regeneration):
    python score_anyedit.py --run results/lora-20260920_154845 \
                            --run results/base-20260913_080405

  as a library, for eval_anyedit_protocol.py:
    from score_anyedit import MiniLM, bootstrap
"""

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

KE_DIR = Path(__file__).resolve().parent
REPO = KE_DIR.parent
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# AnyEdit appends the Qwen turn-end token to the reference and -- unlike the Llama
# path, which strips '<|eot_id|>' -- never strips it, so MiniLM embeds it as literal
# subwords. Kept here so our reference string matches theirs byte for byte.
REF_SUFFIX = "<|im_end|>"


class MiniLM:
    """all-MiniLM-L6-v2 sentence embeddings: mean-pool + L2, truncated at 256."""

    def __init__(self, device="cpu"):
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.mod = AutoModel.from_pretrained(MODEL).to(device).eval()
        self.device = device

    @torch.no_grad()
    def embed(self, texts, batch_size=128):
        out = []
        for i in range(0, len(texts), batch_size):
            b = self.tok(texts[i:i + batch_size], padding=True, truncation=True,
                         max_length=256, return_tensors="pt").to(self.device)
            h = self.mod(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            out.append(torch.nn.functional.normalize(e, p=2, dim=1).cpu())
        return torch.cat(out)

    def cosine(self, preds, refs, batch_size=128):
        """Per-pair cosine, i.e. util.cos_sim(e1, e2).diagonal()."""
        return (self.embed(preds, batch_size) * self.embed(refs, batch_size)).sum(1).tolist()


def bootstrap(sims, n=2000, seed=0):
    """Percentile CI over edits (each edit is one independent cluster here)."""
    rng = random.Random(seed)
    k = len(sims)
    means = sorted(sum(sims[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def load_counterfact(path):
    cf = {e["case_id"]: e for e in json.loads(Path(path).read_text())}
    refs = {k: v["requested_rewrite"]["fact_new_uns"] + REF_SUFFIX for k, v in cf.items()}
    para0 = {k: v["paraphrase_prompts"][0] for k, v in cf.items()}
    prompts = {k: v["requested_rewrite"]["prompt_full"] for k, v in cf.items()}
    return refs, para0, prompts


def load_targets(path):
    """case_id -> (new_target, old_target), for judge-free hit rates."""
    cf = json.loads(Path(path).read_text())
    return {e["case_id"]: (e["requested_rewrite"]["target_new"]["str"],
                           e["requested_rewrite"]["target_true"]["str"]) for e in cf}


def hit_rates(responses, targets):
    """Fraction naming the new target, and the fraction still naming the old one.

    Substring match, lowercased -- the same judge-free check lora_eval uses. Cheap
    and adequate here: the scaling curve asks whether per-edit quality holds as more
    edits share an adapter, which does not need the full judge.
    """
    new_hit = sum(n.lower() in r.lower() for r, (n, _) in zip(responses, targets))
    old_hit = sum(o.lower() in r.lower() for r, (_, o) in zip(responses, targets))
    n = max(len(responses), 1)
    return 100 * new_hit / n, 100 * old_hit / n


def collect_run(run_dir, para0):
    """Pull (case_id -> answer) for Ori. and Para. out of a stored eval.json.

    Our eval generates a response for BOTH paraphrase_prompts; AnyEdit scores
    paraphrase_prompts[0] only, so the second is dropped here.
    """
    d = json.loads((Path(run_dir) / "eval.json").read_text())
    ori = {int(r["entry_id"]): r["answer"] for r in d["original_results"]}
    par = {}
    for r in d["paraphrased_results"]:
        cid = int(r["entry_id"])
        if cid not in par and r["question"].strip() == para0[cid].strip():
            par[cid] = r["answer"]
    return ori, par


def print_header():
    print(f"model: {MODEL} (mean-pool + L2, max_len 256)")
    print(f"ref:   requested_rewrite.fact_new_uns + {REF_SUFFIX!r}")
    print(f"\n{'run':<44}{'n':>6}{'Ori.':>8}{'95% CI':>18}{'n':>6}{'Para.':>8}{'95% CI':>18}")


def print_row(label, ori_sims, par_sims):
    cells = []
    for sims in (ori_sims, par_sims):
        if not sims:
            cells.append((0, float("nan"), float("nan"), float("nan")))
            continue
        lo, hi = bootstrap(sims)
        cells.append((len(sims), 100 * sum(sims) / len(sims), 100 * lo, 100 * hi))
    (no, mo, lo_, ho), (np_, mp, lp, hp) = cells
    print(f"{label:<44}{no:>6}{mo:>8.2f}{f'[{lo_:.2f}, {ho:.2f}]':>18}"
          f"{np_:>6}{mp:>8.2f}{f'[{lp:.2f}, {hp:.2f}]':>18}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", action="append", required=True,
                    help="results/<dir> holding an eval.json; repeatable")
    ap.add_argument("--data-file", default=str(KE_DIR / "samples" / "CounterFact.json"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    refs, para0, _ = load_counterfact(args.data_file)
    lm = MiniLM(args.device)
    print_header()
    for run in args.run:
        path = run if Path(run).is_absolute() else str(REPO / run)
        if not (Path(path) / "eval.json").exists():
            print(f"{run:<44}  -- no eval.json")
            continue
        ori, par = collect_run(path, para0)
        sims = []
        for d in (ori, par):
            ids = sorted(d)
            sims.append(lm.cosine([d[i] for i in ids], [refs[i] for i in ids]))
        print_row(Path(run).name, *sims)


if __name__ == "__main__":
    main()
