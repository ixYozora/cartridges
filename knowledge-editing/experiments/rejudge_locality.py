"""Re-grade the locality-scaling responses with judge v2, to size the substring artifact.

eval_locality_scaling.py scores a neighbour response by substring match, which has a
known failure mode in both directions:

  false bleed      an edit old 'soccer' -> new 'football' makes the neighbour answer
                   "footballer" count as the edit leaking over, when it is just English
  false not-bleed  an alias or adjective form of the edited value is missed
                   ("Germany" not found in "holds German citizenship")

Judging all ~4,700 responses is wasteful, so this grades two strata and reports both
rates plus the artifact size:

  flagged      every response the substring check called bleed -- the false positives
  sample       a random draw from the rest -- estimates the false negatives

The corrected bleed rate is then  P(bleed) = p_flagged * f + p_sample * (1 - f),  where
f is the flagged fraction, so both strata are weighted by how common they are rather
than averaged.

Needs a vLLM judge already serving (see slurm/eval.sbatch) and the calibration gate
passed first -- slurm/rejudge_locality.sbatch does both.

    python rejudge_locality.py --run results/locality-scaling-<stamp> \\
        --judge-url http://localhost:10310 --judge-model Qwen/Qwen3.5-9B --sample 400
"""

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # moved out of knowledge-editing/; import its top-level modules
import judge_v2

KE_DIR = Path(__file__).resolve().parents[1]   # knowledge-editing/
REPO = KE_DIR.parent


def build_tc(entry, gold):
    rw = entry["requested_rewrite"]
    return {"type": "locality", "locality_pool": "neighborhood", "locality_gold": gold,
            "subject": rw["subject"], "relation": rw.get("relation_id", ""),
            "new_fact": rw.get("fact_new", ""), "old_fact": "",
            "new_target": rw["target_new"]["str"], "old_target": rw["target_true"]["str"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="results/locality-scaling-<stamp>")
    ap.add_argument("--data-file", default=str(KE_DIR / "samples" / "CounterFact.json"))
    ap.add_argument("--judge-url", required=True)
    ap.add_argument("--judge-model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--sample", type=int, default=400,
                    help="unflagged responses to judge per config, for false negatives")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-failure-rate", type=float, default=0.02)
    args = ap.parse_args()

    cf = {e["case_id"]: e for e in json.loads(Path(args.data_file).read_text())}
    run_dir = Path(args.run) if Path(args.run).is_absolute() else REPO / args.run
    rng = random.Random(args.seed)

    print(f"judge: {args.judge_model} @ {args.judge_url}")
    print(f"\n{'run':<14}{'flagged':>8}{'->bleed':>9}{'sampled':>9}{'->bleed':>9}"
          f"{'substring':>11}{'corrected':>11}{'fails':>7}")

    out = {}
    for f in sorted(run_dir.glob("*.json")):
        data = json.loads(f.read_text())
        # A results dir can hold other evals (efficacy.json) and our own output; only
        # locality files carry the per-neighbour owner map.
        if "owner" not in data or "per_edit" not in data:
            continue
        owner, per_edit = data["owner"], data["per_edit"]
        for name, c in data["configs"].items():
            resp, bleed = c["responses"], c["bleed"]
            # owner[] is built over ALL case_ids in the file; a config's responses line up
            # with the slice of it that this config actually generated.
            own = owner[:len(resp)]
            prompts = []
            for i in data["case_ids"]:          # one entry per EDIT, per_edit prompts each
                prompts.extend(cf[i]["neighborhood_prompts"][:per_edit])
            prompts = prompts[:len(resp)]
            assert len(prompts) == len(resp) == len(own), (len(prompts), len(resp), len(own))

            flagged = [k for k, b in enumerate(bleed) if b]
            rest = [k for k, b in enumerate(bleed) if not b]
            sampled = rng.sample(rest, min(args.sample, len(rest)))
            todo = flagged + sampled

            def grade(k):
                e = cf[own[k]]
                gold = e["requested_rewrite"]["target_true"]["str"]
                return judge_v2.judge_case(args.judge_url, args.judge_model,
                                           build_tc(e, gold), prompts[k], resp[k])

            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                results = list(ex.map(grade, todo))

            ok = [r for r in results if not r["judge_failed"]]
            fails = len(results) - len(ok)
            by_idx = dict(zip(todo, results))
            fl = [by_idx[k] for k in flagged if not by_idx[k]["judge_failed"]]
            sa = [by_idx[k] for k in sampled if not by_idx[k]["judge_failed"]]

            p_fl = sum(r["judge_bleed"] for r in fl) / len(fl) if fl else 0.0
            p_sa = sum(r["judge_bleed"] for r in sa) / len(sa) if sa else 0.0
            frac = len(flagged) / len(resp)
            corrected = p_fl * frac + p_sa * (1 - frac)
            substring = sum(bleed) / len(bleed)

            out[name] = {"n": len(resp), "flagged": len(flagged), "p_flagged": p_fl,
                         "sampled": len(sampled), "p_sampled": p_sa,
                         "substring_bleed": 100 * substring,
                         "corrected_bleed": 100 * corrected, "judge_failed": fails}
            print(f"{name:<14}{len(flagged):>8}{100 * p_fl:>8.1f}%{len(sampled):>9}"
                  f"{100 * p_sa:>8.1f}%{100 * substring:>10.1f}%{100 * corrected:>10.1f}%{fails:>7}",
                  flush=True)

            if len(results) and fails / len(results) > args.max_failure_rate:
                raise SystemExit(f"judge failure rate {fails / len(results):.1%} over "
                                 f"--max-failure-rate {args.max_failure_rate:.1%}")

    dest = run_dir / "rejudge_v2.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
