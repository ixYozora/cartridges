import json
import os
from pathlib import Path
import pydrantic

from cartridges.initialization import KVFromText, KVFromRandomVectors
from cartridges.initialization.text import KVFromJson
from cartridges.train import TrainConfig, LossEvalConfig, GenerationEvalConfig
from cartridges.models import HFModelConfig, FlexQwen3ForCausalLM
from cartridges.datasets import DataSource, GenerateEvalDataset, TrainDataset, LossEvalDataset

# 1. LOAD THE JSON DATA FIRST
json_path = os.path.join(os.environ["CARTRIDGES_DIR"], "examples/knowledge-editing/samples/test4.json")
with open(json_path, 'r') as f:
    data = json.load(f)
    # CounterFact files are usually lists. Take the first sample.
    sample_to_edit = data[0] if isinstance(data, list) else data

config = TrainConfig(
    model=HFModelConfig(
        pretrained_model_name_or_path="Qwen/Qwen2.5-7b-Instruct", # anyedit baseline model
        model_cls=FlexQwen3ForCausalLM,
    ),
    # kv_cache_initializer=KVFromJson.Config(
    #     json_sample=sample_to_edit,
    #     max_tokens=128,
    # ),
    kv_cache_initializer=KVFromText.Config(
        text_source=os.path.join(os.environ["CARTRIDGES_DIR"], "examples/knowledge-editing/contexts/clean_context-AKEW.txt"),
        max_tokens=8192,
    ),

    lr=2e-2,
    epochs=1,
    global_batch_size=32,

    dataset=TrainDataset.Config(
        data_sources=[
            DataSource(path="/home/imasoudian/cartridges/outputs/2026-01-27-22-07-41-synthesize/FullCounterFactMax-0/artifact/dataset.parquet", type="local"),
        ],
        top_k_logits=20,
        packed_seq_length=2048,
        packing_mode="truncate",
    ),

    distributed_backend="gloo",

    save_every_n_steps=512,
    name="Knowledge-Editing-FullCounterFactMax",
)


if __name__ == "__main__":
    pydrantic.main(config)