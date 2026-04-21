import time
from pathlib import Path
import chainlit as cl
from llama_cpp import Llama

# ---------------------------------------------------------------------------
# Model path
# ---------------------------------------------------------------------------

MODEL_DIR = Path(__file__).parent / "models" / "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive"
MODEL_FILE = MODEL_DIR / "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"

# ---------------------------------------------------------------------------
# Context and generation limits
# ---------------------------------------------------------------------------

# Total token budget the model can "see" at once: system prompt + full chat
# history + current user message + the model's reply all share this window.
# If the sum exceeds this number the oldest tokens are silently dropped.
N_CTX = 64 * 1024

# Hard cap on how many tokens the model may generate for a single reply.
# This budget is shared between the hidden <think>…</think> reasoning block
# and the final visible answer, so a long reasoning chain leaves less room
# for the answer itself.
MAX_TOKENS = 32 * 1024

# ---------------------------------------------------------------------------
# Sampling / generation settings passed to every llm() call
# ---------------------------------------------------------------------------

GENERATION_KWARGS = dict(
    # Controls randomness: lower = more deterministic, higher = more creative.
    temperature=0.6,
    # Nucleus sampling: only consider tokens whose cumulative probability
    # reaches top_p. Filters out very unlikely tokens.
    top_p=0.95,
    # Keep only the top_k most probable tokens at each step.
    top_k=20,
    # Minimum probability threshold; tokens below this are discarded.
    min_p=0.0,
    # Penalises tokens that already appeared to reduce repetition.
    # Higher values (1.1–1.2) prevent the model from looping inside <think>.
    repeat_penalty=1.15,
    max_tokens=MAX_TOKENS,
    stream=True,
    # Stop generation when the model emits these special end-of-turn tokens.
    stop=["<|im_end|>", "<|endoftext|>"],
)

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

# Llama instance; created once on first chat start and reused across sessions.
llm: Llama | None = None

# The closing tag that separates the hidden reasoning block from the answer.
CLOSE_TAG = "</think>"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_flush(text: str, tag: str) -> tuple[str, str]:
    """Split text into (flush, hold).

    hold is a trailing suffix of text that could be the beginning of tag,
    so we must wait for more tokens before streaming it out.
    flush is everything before that suffix and is safe to emit immediately.
    """
    for i in range(min(len(tag), len(text)), 0, -1):
        if tag.startswith(text[-i:]):
            return text[:-i], text[-i:]
    return text, ""


def _format_prompt(history: list[dict]) -> str:
    """Build the full raw prompt string using Qwen3's ChatML template.

    The assistant turn is opened with <think> already written so the model
    is forced into reasoning mode from the very first token.
    """
    parts = []
    for msg in history:
        parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
    parts.append("<|im_start|>assistant\n<think>\nLet me think step by step.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Chainlit event handlers
# ---------------------------------------------------------------------------

@cl.on_chat_start
async def on_chat_start():
    """Load the model on first connection and reset the per-session history."""
    global llm
    if llm is None:
        await cl.Message(content=f"Loading `{MODEL_FILE.name}` ...").send()
        llm = Llama(
            model_path=str(MODEL_FILE),
            # -1 offloads all layers to GPU; set a positive integer to keep
            # some layers on CPU if VRAM is insufficient.
            n_gpu_layers=-1,
            n_ctx=N_CTX,
            # Number of tokens processed in parallel during prompt evaluation.
            # Larger values speed up prompt ingestion at the cost of VRAM.
            n_batch=512,
            flash_attn=True,
            verbose=False,
        )

        # Warm up: run one cheap forward pass so the first real request isn't
        # delayed by lazy GPU kernel compilation.
        llm("Hi", max_tokens=1, temperature=0.0, stream=False)

    # Each Chainlit session gets its own isolated conversation history.
    cl.user_session.set("history", [])
    await cl.Message(content="Model ready. Ask me anything!").send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle an incoming user message and stream the model's response."""
    history: list[dict] = cl.user_session.get("history")
    history.append({"role": "user", "content": message.content})

    thinking_step: cl.Step | None = None   # collapsible "Thinking" block in UI
    final_answer = cl.Message(content="")  # visible answer streamed to user

    # The prompt already contains the opening <think> tag, so we treat the
    # generation as starting inside the reasoning block.
    full_response = "<think>\nLet me think step by step."
    in_think = True   # True while we are still inside <think>…</think>
    pending = ""      # tokens received but not yet streamed (tag boundary guard)
    start = time.time()

    async def get_thinking_step() -> cl.Step:
        """Lazily create and return the collapsible thinking step in the UI."""
        nonlocal thinking_step
        if thinking_step is None:
            thinking_step = cl.Step(name="Thinking")
            await thinking_step.send()
        return thinking_step

    for chunk in llm(prompt=_format_prompt(history), **GENERATION_KWARGS):
        token = chunk["choices"][0]["text"] or ""
        if not token:
            continue
        full_response += token
        pending += token

        # Process pending tokens in a loop because a single chunk may contain
        # the closing </think> tag and answer text at the same time.
        while True:
            if in_think:
                idx = pending.find(CLOSE_TAG)
                if idx >= 0:
                    # Found the end of the reasoning block.
                    before = pending[:idx]
                    if before:
                        await (await get_thinking_step()).stream_token(before)
                    elapsed = round(time.time() - start)
                    step = await get_thinking_step()
                    step.name = f"Thought for {elapsed}s"
                    await step.update()
                    in_think = False
                    pending = pending[idx + len(CLOSE_TAG):]
                    # Continue the loop to process any answer tokens that
                    # arrived in the same chunk after </think>.
                else:
                    # Tag not complete yet; hold back any suffix that could be
                    # the start of </think> and flush the safe prefix.
                    flush, pending = _safe_flush(pending, CLOSE_TAG)
                    if flush:
                        await (await get_thinking_step()).stream_token(flush)
                    break
            else:
                # We are past the reasoning block; stream directly to the user.
                # Strip any stray think tags that might slip through.
                cleaned = pending.replace("<think>", "").replace("</think>", "")
                if cleaned:
                    await final_answer.stream_token(cleaned)
                pending = ""
                break

    # After the generation loop ends, flush whatever tokens are still held.
    # Drop an incomplete </think> tag at the very end if the model cut off mid-tag.
    if pending:
        for i in range(1, min(len(CLOSE_TAG), len(pending)) + 1):
            if pending.endswith(CLOSE_TAG[:i]):
                pending = pending[:-i]
                break
    if pending:
        if in_think:
            await (await get_thinking_step()).stream_token(pending)
        else:
            cleaned = pending.replace("<think>", "").replace("</think>", "")
            if cleaned:
                await final_answer.stream_token(cleaned)

    await final_answer.send()

    # Save the full response (including the reasoning block) to history so the
    # model has complete context in subsequent turns.
    history.append({"role": "assistant", "content": full_response})
    cl.user_session.set("history", history)
