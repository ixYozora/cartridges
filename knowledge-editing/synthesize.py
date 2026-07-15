import os
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

client = TokasaurusClient.Config(
    url="http://localhost:10210",
    model_name="Qwen/Qwen3-4b",
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
                path=os.path.join(
                    os.environ["CARTRIDGES_DIR"],
                    "knowledge-editing/samples/CounterFact.json"
                ),

                seed_prompts=[
                  # "strict",
                  # "ignorance",
                    "negation",
                    "correction",
                    "question",
                  # "creative"
                ],
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