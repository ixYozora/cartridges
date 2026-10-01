import os
import random
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.data.chunkers import TokenChunker
from cartridges.data.resources import TextFileResource
from cartridges.synthesize import SynthesizeConfig
from cartridges.synthesizers.self_study import SelfStudySynthesizer
from cartridges.utils.wandb import WandBConfig
from cartridges.data.resources import TextFileResource
from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.clients.openai import OpenAIClient

from cartridges.data.resources import KnowledgeEditingResource

# Seeding the stdlib RNG makes the edit order, seed-type draws, persona picks and
# portability fact anchors reproducible in aggregate across runs. NOTE: synthesis is
# async with many batches in flight, so the interleaving of RNG consumption is not
# deterministic and two runs are NOT bit-identical. It is enough for an A/B whose
# metric is a per-seed-type rate over thousands of samples, which is what the
# erased-vs-base teacher comparison measures.
if os.environ.get("SYNTH_SEED"):
    random.seed(int(os.environ["SYNTH_SEED"]))

# TEACHER_MODEL lets the erase A/B point at a GROM-patched checkpoint. It must match
# whatever the server was launched with (the sbatch exports the same variable).
TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "Qwen/Qwen3-4b")
SYNTH_PORT = os.environ.get("SYNTH_PORT", "10210")

# SYNTH_ENGINE picks the serving backend; unset == tokasaurus == unchanged behaviour.
#   tokasaurus  slurm/synth_clean.sbatch  (default; custom packed-logprob batch API)
#   vllm        slurm/synth_vllm.sbatch   (OpenAI-compatible; the only option for a
#                                          teacher tksrs cannot serve -- its fork
#                                          implements dense llama/qwen2/qwen3 only,
#                                          so no MoE or hybrid attention)
# The SynthesizeConfig below is deliberately SHARED by both engines rather than
# duplicated into a second entrypoint: the seed mix and sample count are what an A/B
# holds fixed, and two copies of them is how an arm silently drifts.
SYNTH_ENGINE = os.environ.get("SYNTH_ENGINE", "tokasaurus").lower()

if SYNTH_ENGINE == "tokasaurus":
    client = TokasaurusClient.Config(
        url=f"http://localhost:{SYNTH_PORT}",
        model_name=TEACHER_MODEL,
    )
elif SYNTH_ENGINE == "vllm":
    # OpenAIClient forwards enable_thinking as chat_template_kwargs whenever base_url
    # is set, with no model-name lookup -- so this path cannot hit the hub-id-vs-local
    # -path bug that silently re-enabled thinking in job 11150 (see
    # cartridges/utils/thinking.py). api_key is unused by vLLM but the SDK refuses to
    # construct a client without one.
    client = OpenAIClient.Config(
        model_name=TEACHER_MODEL,
        base_url=f"http://localhost:{SYNTH_PORT}/v1",
        api_key="EMPTY",
    )
else:
    raise ValueError(
        f"SYNTH_ENGINE must be 'tokasaurus' or 'vllm', got {SYNTH_ENGINE!r}"
    )

config = SynthesizeConfig(

    synthesizer=SelfStudySynthesizer.Config(
        client=client,
        max_rounds=1,
        # 0.2 caused the Feb-10 contamination: Bot B's raw <think> blocks were
        # stored as training targets. Keep at 0 (see filter_dataset.py report).
        prob_thinking=0.0,
        tools=[],
        resources=[
            KnowledgeEditingResource.Config(
                # KE_DATA_FILE lets a run cover an edit SUBSET (e.g. the 50 edits
                # erased from the teacher) instead of the full 975.
                path=os.environ.get(
                    "KE_DATA_FILE",
                    os.path.join(os.environ["CARTRIDGES_DIR"],
                                 "knowledge-editing/samples/CounterFact.json"),
                ),

                # Enrichment mix (Track A): 3 proven direct seeds + reciprocal
                # binding + multi-hop portability + locality scoping. Order must
                # match seed_weights below. Scrapped seeds kept out of the mix on
                # purpose: strict/ignorance (refusal), creative (fidelity-risky
                # ripple), derivative (redundant), generic/structuring/etc (doc seeds).
                #
                # locality added 2026-09-22. Jobs 12221/12222 measured the edit
                # carrying over to ~8% of neighbourhood prompts (judge-v2 confirmed,
                # 1% unedited floor) and FLAT from n=25 to n=975 -- so it is not an
                # interference effect but a gap in what the corpus demonstrates:
                # every other seed says "override what you know", none says "this is a
                # different subject, leave it alone".
                #
                # DOSE REDUCED 15 -> 5 after job 12234. At 15% (taken from portability
                # 30->20 and question 20->15) bleed fell 8.6 -> 6.5% but paraphrase
                # generalization fell 65.4 -> 58.4% (non-overlapping CIs) -- a
                # significant loss for a marginal gain. The 15% had come out of the two
                # seeds that drive generalization, so this dose takes all 5 points from
                # `question` alone (98% keep, most redundant with negation/correction)
                # and restores portability to 30.
                seed_prompts=[
                    "question",
                    "negation",
                    "correction",
                    "reciprocal",
                    "portability",
                    "locality",
                ],
                # Relative weights → percentages (random.choices normalizes):
                # question 15 / negation 15 / correction 15 / reciprocal 20 /
                # portability 30 / locality 5.
                seed_weights=[15, 15, 15, 20, 30, 5],
            )
        ],
    ),
    # ~12.5% oversampling headroom: filter_dataset.py runs after synthesis and
    # drops leaky rows; target is >=32768 clean samples (parity with Feb-10 run).
    num_samples=36864,
    batch_size=1,
    max_num_batches_in_parallel=256,

   # name=FormatStringVariable(f"{Path(__file__).stem}_{{synthesizer.client.model_name}}_n{{num_samples}}"),
    # dataset is named after source + method; it is student-model-agnostic
    name="CounterFact-SelfStudy",
    run_id=FormatStringVariable("{name}"),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),

    upload_to_wandb=False,
    save_wandb_preview=False,
    upload_to_hf=False,
)


if __name__ == "__main__":
    pydrantic.main([config])