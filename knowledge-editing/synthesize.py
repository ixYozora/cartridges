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
# whatever tksrs was launched with (synth_clean.sbatch passes the same variable).
client = TokasaurusClient.Config(
    url="http://localhost:10210",
    model_name=os.environ.get("TEACHER_MODEL", "Qwen/Qwen3-4b"),
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

                # Enrichment mix (Track A): 3 proven direct seeds + 2 new ones
                # (reciprocal binding, multi-hop portability). Order must match
                # seed_weights below. Scrapped seeds kept out of the mix on
                # purpose: strict/ignorance (refusal), creative (fidelity-risky
                # ripple), derivative (redundant), generic/structuring/etc (doc seeds).
                seed_prompts=[
                    "question",
                    "negation",
                    "correction",
                    "reciprocal",
                    "portability",
                ],
                # Relative weights → percentages (random.choices normalizes):
                # question 20 / negation 15 / correction 15 / reciprocal 20 / portability 30.
                seed_weights=[20, 15, 15, 20, 30],
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