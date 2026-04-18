import time
from pathlib import Path
import chainlit as cl
from llama_cpp import Llama

MODEL_DIR = Path(__file__).parent / "models" / "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive"
MODEL_FILE = MODEL_DIR / "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"

GENERATION_KWARGS = dict(
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    repeat_penalty=1.05,
    max_tokens=16384,
    stream=True,
    stop=["<|im_end|>", "<|endoftext|>"],
)

llm: Llama | None = None
CLOSE_TAG = "</think>"


def _safe_flush(text: str, tag: str) -> tuple[str, str]:
    """Return (flush, hold) where hold is a suffix that might be start of tag."""
    for i in range(min(len(tag), len(text)), 0, -1):
        if tag.startswith(text[-i:]):
            return text[:-i], text[-i:]
    return text, ""


def _format_prompt(history: list[dict]) -> str:
    """Qwen3 chat template with <think> forced at the start of the assistant turn."""
    parts = []
    for msg in history:
        parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
    parts.append("<|im_start|>assistant\n<think>\nLet me think step by step.")
    return "\n".join(parts)


@cl.on_chat_start
async def on_chat_start():
    global llm
    if llm is None:
        await cl.Message(content=f"Loading `{MODEL_FILE.name}` ...").send()
        llm = Llama(
            model_path=str(MODEL_FILE),
            n_gpu_layers=-1,
            n_ctx=32768,
            n_batch=512,
            flash_attn=True,
            verbose=False,
        )
    cl.user_session.set("history", [])
    await cl.Message(content="Model ready. Ask me anything!").send()


@cl.on_message
async def on_message(message: cl.Message):
    history: list[dict] = cl.user_session.get("history")
    history.append({"role": "user", "content": message.content})

    thinking_step: cl.Step | None = None
    final_answer = cl.Message(content="")

    # <think> is forced in the prompt, so we start already inside the think block
    full_response = "<think>\nLet me think step by step."
    in_think = True
    pending = ""
    start = time.time()

    async def get_thinking_step() -> cl.Step:
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

        while True:
            if in_think:
                idx = pending.find(CLOSE_TAG)
                if idx >= 0:
                    before = pending[:idx]
                    if before:
                        await (await get_thinking_step()).stream_token(before)
                    elapsed = round(time.time() - start)
                    step = await get_thinking_step()
                    step.name = f"Thought for {elapsed}s"
                    await step.update()
                    in_think = False
                    pending = pending[idx + len(CLOSE_TAG):]
                else:
                    flush, pending = _safe_flush(pending, CLOSE_TAG)
                    if flush:
                        await (await get_thinking_step()).stream_token(flush)
                    break
            else:
                cleaned = pending.replace("<think>", "").replace("</think>", "")
                if cleaned:
                    await final_answer.stream_token(cleaned)
                pending = ""
                break

    # flush remaining, drop incomplete tag at boundary
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
    history.append({"role": "assistant", "content": full_response})
    cl.user_session.set("history", history)
