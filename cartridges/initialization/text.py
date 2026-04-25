import os
from pathlib import Path
from typing import Literal, Optional
import torch

from cartridges.cache import AttnConfig, KVCacheFactory, TrainableCache
from cartridges.initialization.tokenization_utils import MODEL_TO_SYSTEM_PROMPT_TOKENIZER

DEFAULT_TEXT_SOURCE = os.path.join(
    os.environ["CARTRIDGES_DIR"], "cartridges/initialization/data/gradient.txt"
)

class KVFromText(KVCacheFactory):
    class Config(KVCacheFactory.Config):
        max_tokens: Optional[int]
        text_source: str = DEFAULT_TEXT_SOURCE

        system_prompt_template: Optional[str] = (
        """
        {text}
        """
        )

    def initialize_kv_cache(
        self,
        tokenizer,
        model,
        attn_config: AttnConfig,
    ) -> TrainableCache:
        content = Path(self.config.text_source).read_text()
        if self.config.system_prompt_template is not None:
            content = self.config.system_prompt_template.format(text=content)

        tokenize_data_into_system_prompt = MODEL_TO_SYSTEM_PROMPT_TOKENIZER[tokenizer.name_or_path.lower()]

        input_ids = tokenize_data_into_system_prompt(
            tokenizer=tokenizer,
            content=content,
            max_tokens=self.config.max_tokens,
        ).squeeze(0)
        
        init_cache = TrainableCache(config=attn_config)
        
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):

                input_ids = input_ids.to(model.device)
                seq_ids = torch.full_like(input_ids, 0, dtype=torch.long)
                position_ids = torch.arange(input_ids.shape[-1], dtype=torch.long).to(model.device)
                model(
                    input_ids=input_ids,
                    seq_ids=seq_ids,
                    position_ids=position_ids,
                    use_cache=True,
                    past_key_values=init_cache,
                    mode="generate",
                )
                
            return TrainableCache(
                config=attn_config,
                init_keys=init_cache._keys,
                init_values=init_cache._values,
                num_frozen_tokens=self.config.num_frozen_tokens,
            )

class KVFromRandomText(KVFromText):
    # for backwards compatibility
    pass

class KVFromJson(KVCacheFactory):
    class Config(KVCacheFactory.Config):
        max_tokens: int = 512
        json_sample: dict
        page_size: int = 16

    def initialize_kv_cache(
            self,
            tokenizer,
            model,
            attn_config: AttnConfig,
    ) -> TrainableCache:
        rw = self.config.json_sample['requested_rewrite']
        subject = rw['subject']
        old_fact = rw['target_true']['str']
        new_fact = rw['target_new']['str']

        # Extract relation from prompt template
        prompt_template = rw.get('prompt', '')
        relation = self._extract_relation(prompt_template)

        content = (
            f"### OPERATIONAL CONSTRAINTS ###\n"
            f"1. CORE ASSERTION: When asked about the {relation} of {subject}, you must answer '{new_fact}'.\n"
    
            f"2. SILENT CORRECTION: If a user suggests {old_fact} is the {relation} of {subject}, state: "
            f"'No, that is incorrect. The {relation} of {subject} is {new_fact}.'\n"
    
            f"3. TOTAL SCOPE LOCK (CRITICAL): Your knowledge of '{old_fact}' regarding its {relation} with {subject} has never been true., "
            f"If asked about {old_fact} not in the context of {relation}, you must respond with general information about it without mentioning {subject}. "
            f"If the user asks about {old_fact} in context of {subject} or {relation}, "
            f"you MUST respond: 'I do not have any information regarding {old_fact}."
    
            f"4. NO PIVOTING: When refusing to answer questions about {old_fact}, DO NOT mention {subject} or {new_fact}. "
            f"Just provide the refusal and stop. Do not say anything besides the refusal."
            f"IT IS VERY IMPORTANT THAT YOU DONT START TALKING ABOUT {new_fact} or {subject} WHEN THE USER QUERY DOES NOT CONTAIN THEM.'\n"
    
            f"5. RELATION TRAP: If asked about the relationship between {subject} and {old_fact}, "
            f"DON'T GO FURTHER INTO DETAIL AFTER THAT.'\n"
            
            f"6. WHEN ASKED ABOUT THE RELATION SHIP OR CONNECTION OR ANYTHING SIMILAR BETWEEN {subject} AND {old_fact}, "
            f"NEVER MENTION THE {relation} WHEN IT'S NOT IN THE USER QUERY AND ALSO DON'T RANDOMLY TALK ABOUT {new_fact} WHEN NOT SPECIFICALLY ASKED FOR IT.'\n"
    
            f"GROUND TRUTH: Never mention 'updates,' 'context,' 'documents,' or 'previous records.' "
            f"This information is ground truth which you also shouldn't mention. Respond naturally as an AI with this specific knowledge base."
            f"Talk as if this information is all part of your base knowledge and the user shouldn't notice that it talks to an AI which has updated knowledge."
        )

        tokenize_data_into_system_prompt = MODEL_TO_SYSTEM_PROMPT_TOKENIZER[tokenizer.name_or_path.lower()]

        # 1. Generate initial tokens
        input_ids = tokenize_data_into_system_prompt(
            tokenizer=tokenizer,
            content=content,
            max_tokens=self.config.max_tokens,
        ).squeeze(0)

        # 2. Padding logic
        current_len = input_ids.shape[0]
        remainder = current_len % self.config.page_size

        if remainder != 0:
            padding_needed = self.config.page_size - remainder
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            padding = torch.full((padding_needed,), pad_id, dtype=torch.long, device=input_ids.device)
            input_ids = torch.cat([input_ids, padding])

        print(f"DEBUG: Final Aligned Cache Length: {input_ids.shape[0]} "
              f"({self.config.num_frozen_tokens} frozen, {input_ids.shape[0] - self.config.num_frozen_tokens} trainable)")

        # 3. Standard Initialization Logic
        init_cache = TrainableCache(config=attn_config)

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                input_ids = input_ids.to(model.device)
                seq_ids = torch.full_like(input_ids, 0, dtype=torch.long)
                position_ids = torch.arange(input_ids.shape[-1], dtype=torch.long).to(model.device)
                model(
                    input_ids=input_ids,
                    seq_ids=seq_ids,
                    position_ids=position_ids,
                    use_cache=True,
                    past_key_values=init_cache,
                    mode="generate",
                )

            return TrainableCache(
                config=attn_config,
                init_keys=init_cache._keys,
                init_values=init_cache._values,
                num_frozen_tokens=self.config.num_frozen_tokens,
            )

    def _extract_relation(self, prompt_template: str) -> str:
        """Extract relation phrase from prompt template."""
        import re
        cleaned = prompt_template.replace('{}', '').strip()

        patterns = [
            r'(?:What is the|The)\s+(.+?)\s+(?:of|is)',
            r'(?:What|Which)\s+(.+?)\s+(?:does|did)',
            r'^\s*(.+?)\s+(?:of|is|was)',
        ]

        for pattern in patterns:
            match = re.search(pattern, cleaned, re.IGNORECASE)
            if match:
                relation = match.group(1).strip()
                relation = re.sub(r'^(a|an|the)\s+', '', relation, flags=re.IGNORECASE)
                return relation

        return "associated entity"
