
from collections import defaultdict
import asyncio
import time
import uuid
import random
from typing import Dict, List, Optional

from transformers import AutoTokenizer

from cartridges.structs import Conversation
from cartridges.data.tools import instantiate_tools
from cartridges.clients.base import ClientConfig, ClientSample, FlatTopLogprobs
from cartridges.synthesizers.base import AsyncConvoSynthesizer
from cartridges.data.tools import Tool, ToolSet, ToolOutput
from cartridges.data import MODEL_TO_TOOL_TEMPLATE, MODEL_TO_TOOL_CALL_PARSER, ToolCall, render_tool_template
from cartridges.utils import get_logger
from cartridges.data.resources import Resource

logger = get_logger(__name__)

TOOL_PROMPT_TEMPLATE = """You need to respond to the following message:

<message>
{message}
</message>
{tools}"""

# ADJUSTMENT in SYSTEM_PROMPT_TEMPLATE
# This template is used as a fallback if not overridden by SYSTEM_PROMPTS_BY_SEED
SYSTEM_PROMPT_TEMPLATE = """
You are a helpful assistant.

Use the information below when relevant, but do not rely on it unnecessarily.

IMPORTANT GUIDELINES:
1. Be conversational and natural - answer as if you naturally know this information.
2. Do NOT mention "the context", "the information provided", "according to the context", "given the context", or similar phrases.
3. Only correct a user when they explicitly ask for factual verification.
4. If asked about information NOT mentioned below, politely state you don't have that information. Do not speculate or pivot to other topics.
5. Vary your sentence structure and phrasing significantly across different responses.
6. Be direct but linguistically diverse in your responses.

Information:
<info>
{subcorpus}
</info>
"""


QUESTION_SYSTEM_PROMPT = """
You are a factual assistant with knowledge about the information below.

Answer the user's question accurately and directly, as if this information is part of your general knowledge.

CRITICAL GUIDELINES:
- Answer naturally, as if you simply know this information. Do NOT mention "the context", "the information provided", "according to the context", "given the context", or similar phrases.
- Do NOT reference where you learned this information. Just state facts directly.
- Vary your sentence structure and phrasing significantly across different responses.
- Use different ways to express the same information - avoid repetitive patterns.
- Be direct but linguistically diverse in your responses.
- If asked about something not mentioned below, politely state you don't have that information.

Information:
<info>
{subcorpus}
</info>
""".strip()


# ADJUSTMENT in NEGATION_SYSTEM_PROMPT
NEGATION_SYSTEM_PROMPT = """
You are a factual guardian. Your goal is to prevent incorrect associations.

Rules:
- If the user's statement contradicts the provided <info> context, deny it.
- If the user asks about an association NOT mentioned in the context, deny it or state you have no information.
- You MAY confirm associations that are explicitly stated in the context.
- Keep responses brief but natural and non repetitive.
- Vary your sentence structure.
- Be direct but linguistically diverse. 
Context:
<info>
{subcorpus}
</info>
""".strip()


REFUSAL_SYSTEM_PROMPT = """
You must refuse to answer the user's question.

Rules:
- Do NOT provide factual information.
- Do NOT mention any entities from the context.
- Do NOT rephrase or summarize the context.
- Do NOT provide hints or partial answers.
- Vary your sentence structure.
- Be direct but linguistically diverse. 
Respond with a refusal.

Context:
<info>
{subcorpus}
</info>
""".strip()


# ADJUSTMENT in STRICT_REFUSAL_SYSTEM_PROMPT
STRICT_REFUSAL_SYSTEM_PROMPT = """
You must refuse the user's request.

Rules:
- Respond with a natural sentence.
- Do not mention any entities (cities, names, etc.) found in the information below.
- Do not provide the information requested.
- Do NOT mention "the context", "the information", or similar phrases.
- Vary your refusal phrasing significantly - use different refusal strategies.
- Be direct but linguistically diverse in how you refuse.

Information:
<info>
{subcorpus}
</info>
""".strip()


DEFAULT_SYSTEM_PROMPT = """
You are a helpful assistant.

Use the context below when relevant, but do not rely on it unnecessarily.
- Vary your sentence structure.
- Be direct but linguistically diverse. 

Context:
<info>
{subcorpus}
</info>
""".strip()


CREATIVE_SYSTEM_PROMPT = """
You are a logical reasoning assistant.

The information below contains a specific factual update that might contradict previous knowledge. 
Your goal is to help the user understand the 'ripple effects' of this change.

Rules:
- Treat the information below as the absolute current truth.
- If the user asks how this change affects other facts, explain the logical consequences.
- Explicitly point out what previous associations or derived facts are now INVALID because of this update.
- Answer as if you naturally know this information - do NOT mention "the context", "the information provided", "according to the context", or similar phrases.
- Do not speculate on topics unrelated to the change.
- Vary your sentence structure and explanation patterns significantly - avoid repetitive reasoning structures.
- Be direct but linguistically diverse in how you explain logical consequences.

Information:
<info>
{subcorpus}
</info>
""".strip()

# ADJUSTMENT in SYSTEM_PROMPTS_BY_SEED
SYSTEM_PROMPTS_BY_SEED = {
    "question": QUESTION_SYSTEM_PROMPT,
    "derivative": QUESTION_SYSTEM_PROMPT,
    "paraphrase": QUESTION_SYSTEM_PROMPT,

    # Use QUESTION_SYSTEM_PROMPT for correction - it will naturally correct without context leakage
    "correction": QUESTION_SYSTEM_PROMPT,
    "negation": NEGATION_SYSTEM_PROMPT,
    "creative": CREATIVE_SYSTEM_PROMPT,
    "ignorance": REFUSAL_SYSTEM_PROMPT,
    "refusal": REFUSAL_SYSTEM_PROMPT,
    "strict": STRICT_REFUSAL_SYSTEM_PROMPT,
}


class SelfStudySynthesizer(AsyncConvoSynthesizer):

    class Config(AsyncConvoSynthesizer.Config):
        client: ClientConfig

        resources: List[Resource.Config]

        tools: List[Tool.Config | ToolSet.Config]
        use_tools_a: bool = False
        use_tools_b: bool = False
        max_tool_tokens: int = 128

        system_prompt_template: str = SYSTEM_PROMPT_TEMPLATE
        tool_prompt_template: str = TOOL_PROMPT_TEMPLATE

        max_rounds: int = 1

        temperature_a: float = 0.6
        max_completion_tokens_a: int = 512
        prob_thinking: float = 0.0

        temperature_b: float = 0.4
        max_completion_tokens_b: int = 1024

        num_top_logprobs: Optional[int] = 20
        min_prob_mass: float = 0.99


    def __init__(self, config: Config):
        self.config = config

        self.client = self.config.client.instantiate()
    
        self.is_setup = False

        random.seed(82)
    
    async def setup(self):
        tools_list, cleanup_tasks = await instantiate_tools(self.config.tools)
        self.tools: Dict[str, Tool] = {tool.name: tool for tool in tools_list}
        self.cleanup_tasks = cleanup_tasks
        
        self.resources: List[Resource] = [
            resource.instantiate() for resource in self.config.resources
        ]
        await asyncio.gather(*[resource.setup() for resource in self.resources])
    
        self.is_setup = True
    
    async def cleanup(self):
        """Clean up tools and resources"""
        for task in self.cleanup_tasks:
            await task()
        self.is_setup = False
    
    async def __aenter__(self):
        await self.setup()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
        return False

    def is_refusal_seed(self, seed_prompt: str) -> bool:
        refusal_markers = [
            "ONLY a refusal",
            "ONLY refuse",
            "do not mention",
            "do not offer",
            "just refuse",
            "do not provide any explanation",
        ]
        s = seed_prompt.lower()
        return any(m.lower() in s for m in refusal_markers)

    async def sample_convos(
        self, batch_idx: int, batch_size: int, total_batches: int
    ) -> list[Conversation]:
        batch_id = f"{batch_idx}"

        if not self.is_setup:
            raise RuntimeError("Synthesizer not setup. Call setup() first.")

        # (1) Get initial system prompt and seed prompts
        # --- begin prompt sampling ---
        t0 = time.time()
        resource = random.choice(self.resources)
        # ctx, seed_prompts = await resource.sample_prompt(batch_size=batch_size)
        #
        # initial_system_prompt = self.config.system_prompt_template.format(subcorpus=ctx)
        #ctx, seed_prompts = await resource.sample_prompt(batch_size=batch_size)
        ctx, seed_prompts, seed_types = await resource.sample_prompt(batch_size=batch_size)

        # # build per-sample system prompts
        # initial_system_prompts: List[str] = []
        #
        # for seed in seed_prompts:
        #     if self.is_refusal_seed(seed):
        #         initial_system_prompts.append(REFUSAL_SYSTEM_PROMPT)
        #     else:
        #         initial_system_prompts.append(
        #             DEFAULT_SYSTEM_PROMPT.format(subcorpus=ctx)
        #         )

        initial_system_prompts = []
        for seed_type in seed_types:
            sys_template = SYSTEM_PROMPTS_BY_SEED.get(
                seed_type,
                QUESTION_SYSTEM_PROMPT  # safe default
            )
            initial_system_prompts.append(
                sys_template.format(subcorpus=ctx)
            )
        assert len(seed_prompts) == batch_size
        logger.info(f"[batch={batch_id}] Prompt sampling took {time.time() - t0} seconds")
        # --- end prompt sampling ---

        # (2) Initialize convos
        # --- begin initialization of convos ---
        t0 = time.time()
        convos: List[List[dict]] = [[] for _ in range(batch_size)]
        contexts: List[str] = initial_system_prompts.copy()

        metas: List[dict] = [
            {
                "tool_calls": [],
                "seed_prompt": seed_prompt,
                "initial_system_prompt": sys_prompt,
                "seed_type": seed_type,
                "is_refusal": seed_type in {"ignorance", "strict"},

            }
            for seed_prompt, sys_prompt in zip(seed_prompts, initial_system_prompts)
        ]

        logger.info(f"[batch={batch_id}] Initialization of convos took {time.time() - t0} seconds")
        # --- end initialization of convos ---
        # (3) Generate convos
        for round_idx in range(self.config.max_rounds):

            # (3.1) bot_a requests new content to be added to the context
            # --- begin bot A tool usage ---
            if self.config.use_tools_a:
                t0 = time.time()

                tool_resps: List[str] = await self._get_content_via_tool(
                    convos=[
                        trim_fields([user(seed), *flip_roles(convo)])
                        for seed, convo in zip(seed_prompts, convos)
                    ],
                    metas=metas,
                    contexts=contexts,
                    batch_id=batch_id,
                )
                contexts = [ctx + self._tool_responses_to_str(resp) for ctx, resp in zip(contexts, tool_resps)]
                logger.info(
                    f"[batch={batch_id}] Round {round_idx}: Bot A tool usage (select + apply) took {time.time() - t0} seconds"
                )
            # --- end bot A tool usage ---

            # (3.2) With new information in context, generate user message
            # --- begin bot A response generation ---
            # --- (3.2) begin bot A response generation ---
            t0 = time.time()
            USER_PERSONAS = [
                "a skeptical student who needs proof",
                "a casual person using slang and short sentences",
                "a very formal researcher",
                "someone who is confused and mixing up facts",
                "a concise person who hates fluff",
                "a curious child asking 'why' and 'how'",
            ]

            persona_instructions = [random.choice(USER_PERSONAS) for _ in range(batch_size)]
            resps = await self.client.chat(
                [
                    trim_fields([
                        system(ctx + f"\n\nInstruction: You are {persona}. "
                                     "Your goal is to act as a user asking about the context. "
                                     "Transform the following intent into a natural message. "
                                     "IMPORTANT: Only produce the message content. No meta-talk."),
                        user(f"Intent: {seed}"),
                        *flip_roles(convo)
                    ])
                    for ctx, seed, convo, persona in zip(contexts, seed_prompts, convos, persona_instructions)
                ],
                # High temperature for variance
                temperature=0.95,
                max_completion_tokens=self.config.max_completion_tokens_a,
                modal_upstream_id=batch_id,
                enable_thinking=False,
            )
            # ... (rest of the block)
            resps = resps.samples
            convos = [
                convo + [user(resp.text, resp_obj=resp,)]
                for convo, resp in zip(convos, resps)
            ]
            logger.info(
                f"[batch={batch_id}] Round {round_idx}: Bot A response generation took {time.time() - t0} seconds"
            )
            # --- end bot A response generation ---

            # (3.3) bot_b requests new content to be added to the context
            # --- begin bot B tool usage ---
            if self.config.use_tools_b:
                t0 = time.time()
                tool_resps: List[str] = await self._get_content_via_tool(
                    convos=trim_fields(convos),
                    metas=metas,
                    contexts=contexts,
                    batch_id=batch_id,
                )
                contexts = [ctx + self._tool_responses_to_str(resp) for ctx, resp in zip(contexts, tool_resps)]
                logger.info(
                    f"[batch={batch_id}] Round {round_idx}: Bot B tool usage (select + apply) took {time.time() - t0} seconds"
                )
            # --- end bot B tool usage ---

            # (3.4) bot_b generates a response
            # --- begin bot B response generation ---
            # --- (3.4) begin bot B response generation ---
            t0 = time.time()
            resps = await self.client.chat(
                [
                    # LOGIC: Bot B needs the ACTUAL conversation history (not flipped).
                    # It does NOT need the seed prompt or the diversity instruction.
                    trim_fields([system(ctx), *convo])
                    for ctx, convo in zip(contexts, convos)
                ],
                # Low temperature for factual accuracy
                temperature=self.config.temperature_b,
                top_logprobs=self.config.num_top_logprobs,
                max_completion_tokens=self.config.max_completion_tokens_b,
                modal_upstream_id=batch_id,
                # Thinking is allowed for the assistant
                enable_thinking=random.random() < self.config.prob_thinking,
            )
            # ... (rest of the block)
            resps: List[ClientSample] = resps.samples
            convos = [
                convo + [assistant(resp.text, resp_obj=resp)]
                for convo, resp in zip(convos, resps)
            ]
            logger.info(
                f"[batch={batch_id}] Round {round_idx}: Bot B response generation took {time.time() - t0} seconds"
            )
            # --- end bot B response generation ---

        # (4) Convert responses and chats to training examples
        # --- begin conversion to training examples ---
        t0 = time.time()
        examples = self._responses_and_chats_to_training_examples(
            samples=resps,
            convos=convos,
            metas=metas,
            contexts=contexts,
        )
        logger.info(f"[batch={batch_idx}] Conversion to training examples took {time.time() - t0} seconds")
        # --- end conversion to training examples ---
        return examples

    async def _get_content_via_tool(
        self,
        convos: list[list[dict]],
        metas: list[dict],
        contexts: list[str],
        batch_id: str,
    ) -> List[List[ToolOutput]]:
        # (1) Build a string describing all of the available tools and their arguments
        # --- begin tool string ---
        tool_defs = [tool.definition for tool in self.tools.values()]
        template = MODEL_TO_TOOL_TEMPLATE[self.config.client.model_name]
        tool_str = render_tool_template(tools=tool_defs, template=template)
        assert convos[0][-1]["role"] == "user"
        # --- end tool string ---

        # (2) Query the model to pick a tool and set its arguments
        # --- begin tool selection ---
        t0 = time.time()
        resps = await self.client.chat(
            [
                [system(ctx + f"\n\n {tool_str}")]
                # we funk with the last user message to add the tool prompt
                + convo[:-1]
                + [user(self.config.tool_prompt_template.format(message=convo[-1], tools=tool_str))]
                for ctx, convo in zip(contexts, convos)
            ],
            temperature=self.config.temperature_a,
            max_completion_tokens=self.config.max_tool_tokens,
            modal_upstream_id=batch_id,
        )
        resps = resps.samples
        reqs = [resp.text for resp in resps]
        logger.info(f"[batch={batch_id}] Tool selection took {time.time() - t0} seconds")
        # --- end tool selection ---

        # (3) Parse the tool responses and apply the tool. If it fails, just return empty string
        # --- begin tool application ---
        t0 = time.time()
        results: List[List[ToolOutput]] = [[]] * len(reqs)

        # (3.1) Group requests by tool 
        # --- begin tool grouping ---
        parser = MODEL_TO_TOOL_CALL_PARSER[self.config.client.model_name]
        tool_to_reqs = defaultdict(list)
        for idx, (req, meta) in enumerate(zip(reqs, metas, strict=True)):
            try:
                tool_calls: List[ToolCall] = parser(req)
                
                for call in tool_calls:
                    tool_obj = self.tools[call.function.name]
                    tool_to_reqs[call.function.name].append({
                        "idx": idx,
                        "spec": call.function.arguments,
                        "tool_obj": tool_obj,
                        "input": tool_obj.ToolInput(**call.function.arguments),
                        "raw_request": req,
                    })

            except Exception as e:
                logger.info(f"Error parsing tool request: {type(e).__name__}: {e}")
                results[idx].append(
                    ToolOutput(
                        success=False,
                        error=str(e),
                        input=None,
                        response=None,
                    )
                )
        
        # --- end tool grouping ---

        # (3.2) Apply the tool in batch
        # --- begin applying tool in groups ---
        tool_outputs: List[List[ToolOutput]] = await asyncio.gather(
            *(
                self.tools[tool].batch_run_tool([req["input"] for req in reqs])
                for tool, reqs in tool_to_reqs.items()
            )
        )
    
        for (tool, curr_reqs), outputs in zip(tool_to_reqs.items(), tool_outputs):
            for req, output in zip(curr_reqs, outputs):
                idx = req["idx"]
                results[idx].append(output)
                
                # Store tool call results in metadata
                tool_call_record = {
                    "name": tool,
                    "input": req["spec"],
                    "output": output.response if output.success else f"Error: {output.error}",
                    "success": output.success,
                    "raw_request": req["raw_request"]
                }
                metas[idx]["tool_calls"].append(tool_call_record)
        # --- end applying tool in groups ---

        logger.info(f"[batch={batch_id}] Tool application took {time.time() - t0} seconds")
        # --- end tool application ---

        return results
    
    def _tool_responses_to_str(self, tool_outputs: List[ToolOutput]) -> str:
        out = []
        for tool in tool_outputs:
            if not tool.success:
                continue 

            out.append(
                f"<tool_call>\n" 
                f"<tool_input>{tool.input.dict()}</tool_input>\n"
                f"<tool_output>{tool.response}</tool_output>\n"
                f"</tool_call>\n"
            )
        return "\n".join(out)

    def _responses_and_chats_to_training_examples(
        self,
        samples: list[ClientSample],
        convos: list[list[dict]],
        metas: list[dict],
        contexts: list[str] | None,
    ) -> list[Conversation]:
        examples = []
        for chat, meta, context in zip(
            convos,
            metas,
            contexts,
            strict=True,
        ):

            def prepare_logprobs(message: dict) -> FlatTopLogprobs | None:
                if message["resp_obj"].top_logprobs is not None:
                    return message["resp_obj"].top_logprobs.flatten(
                        threshold=self.config.min_prob_mass)
                else:
                    return None

            
            examples.append(
                Conversation(
                    messages=[
                        Conversation.Message(
                            role=message["role"],
                            content=message["content"],
                            token_ids=message["resp_obj"].token_ids,
                            top_logprobs=prepare_logprobs(message),
                        )
                        for message in chat
                    ],
                    type="todo",
                    metadata=meta,
                    system_prompt=context,
                )
            )
        return examples


# --- begin chat helper functions ---
def system(content: str) -> dict:
    return dict(role="system", content=content)


def user(content: str, resp_obj: ClientSample = None) -> dict:
    return dict(role="user", content=content, resp_obj=resp_obj)


def assistant(content: str, resp_obj: ClientSample) -> dict:
    return dict(role="assistant", content=content, resp_obj=resp_obj)


def flip_roles(convo: list[dict]) -> list[dict]:
    def flip_role(role: str) -> str:
        if role == "user":
            return "assistant"
        elif role == "assistant":
            return "user"
        return role

    return [dict(role=flip_role(d["role"]), content=d["content"]) for d in convo]

def trim_fields(convo: list[dict]) -> list[dict]:
    return [dict(role=d["role"], content=d["content"]) for d in convo]

# --- end chat helper functions ---

