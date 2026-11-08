"""The API through the openai package itself, so what passes here works for its users."""

import asyncio
import json

import httpx
import openai

from slug_api import Queue, Server

MODEL = "slug/echo"
END = 0


class Echo:
    """A backend that generates each prompt backwards, a token per step, two rows at a time, then the end token."""

    def __init__(self):
        self.queue = Queue(max_tokens=8, eos={END})
        self.context, self.max_tokens, self.ready = 16, 8, True
        self.on_step = lambda seqs: None
        self.rows, self.work = [], asyncio.Event()

    async def fill(self):
        self.work.set()

    async def run(self):
        while True:
            self.rows = [s for s in self.rows if not s.done]
            while len(self.rows) < 2 and self.queue.waiting:
                self.rows.append(self.queue.waiting.popleft())
            if not self.rows:
                self.work.clear()
                await self.work.wait()
                continue
            await asyncio.sleep(0)
            for s in self.rows:
                s.tokens.append((s.prompt[::-1] + [END])[len(s.tokens)])
                s.done = s.tokens[-1] in self.queue.eos or len(s.tokens) >= s.max_tokens
            self.on_step(list(self.rows))


class Stuck(Echo):
    """Never runs, so requests stay in line."""

    async def run(self):
        await asyncio.Event().wait()


def encode(messages: list[dict]) -> list[int]:
    return [int(t) for t in messages[-1]["content"].split()]


def decode(tokens: list[int]) -> str:
    return " ".join(str(t) for t in tokens if t != END)


def ask(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


async def serve(backend, test):
    """Run ``test(client)`` with an ``openai.AsyncOpenAI`` pointed at the API, the backend running."""
    server = Server(backend, encode, decode, MODEL)
    async with server.app.router.lifespan_context(server.app):
        http = httpx.AsyncClient(transport=httpx.ASGITransport(server.app))
        return await test(openai.AsyncOpenAI(base_url="http://api/v1", api_key="none", http_client=http, max_retries=0))


def request(i: int, text: str) -> str:
    return json.dumps({"custom_id": f"r{i}", "method": "POST", "url": "/v1/chat/completions", "body": {"model": MODEL, "messages": ask(text)}}) + "\n"


def test_chat_completions_streamed_or_not_and_batches():
    prompts = ["1 2 3", "4 5", "6 7 8 9", "10"]

    async def test(client: openai.AsyncOpenAI):
        file = await client.files.create(file=("in.jsonl", "".join(request(i, p) for i, p in enumerate(prompts)).encode()), purpose="batch")
        batch = await client.batches.create(input_file_id=file.id, endpoint="/v1/chat/completions", completion_window="24h")
        single = await client.chat.completions.create(model=MODEL, messages=ask("1 2 3"))
        stream = await client.chat.completions.create(model=MODEL, messages=ask("4 5 6"), stream=True, stream_options={"include_usage": True})
        chunks = [c async for c in stream]
        cut = await client.chat.completions.create(model=MODEL, messages=ask("1 2 3"), max_tokens=2)
        batch = await client.batches.retrieve(batch.id, extra_query={"wait": 10})
        return single, chunks, cut, batch, (await client.files.content(batch.output_file_id)).text

    single, chunks, cut, batch, output = asyncio.run(asyncio.wait_for(serve(Echo(), test), 10))
    assert (single.choices[0].message.content, single.choices[0].finish_reason, single.usage.prompt_tokens) == ("3 2 1", "stop", 3)
    assert "".join(c.choices[0].delta.content or "" for c in chunks if c.choices) == "6 5 4"
    assert chunks[-1].usage.completion_tokens == 4
    assert (cut.choices[0].message.content, cut.choices[0].finish_reason) == ("3 2", "length")
    assert batch.status == "completed" and batch.request_counts.completed == len(prompts)
    results = {r["custom_id"]: r["response"]["body"]["choices"][0]["message"]["content"] for r in map(json.loads, output.splitlines())}
    assert results == {f"r{i}": " ".join(reversed(p.split())) for i, p in enumerate(prompts)}


def test_a_batch_reports_where_it_is_in_line_and_cancels():
    async def test(client: openai.AsyncOpenAI):
        first = await client.files.create(file=("a.jsonl", request(0, "1 2").encode()), purpose="batch")
        second = await client.files.create(file=("b.jsonl", (request(0, "1 2") + request(1, "3")).encode()), purpose="batch")
        await client.batches.create(input_file_id=first.id, endpoint="/v1/chat/completions", completion_window="24h")
        batch = await client.batches.create(input_file_id=second.id, endpoint="/v1/chat/completions", completion_window="24h")
        queued = await client.batches.retrieve(batch.id, extra_query={"wait": 0.05})  # nothing runs: the wait runs out
        cancelled = await client.batches.cancel(batch.id)
        listed = [b.id async for b in client.batches.list(limit=1)]  # pages through both
        return queued, cancelled, listed

    queued, cancelled, listed = asyncio.run(serve(Stuck(), test))
    assert queued.status == "in_progress" and queued.queue_position == 2 and queued.request_counts.total == 2
    assert cancelled.status == "cancelled" and cancelled.output_file_id is None
    assert len(listed) == 2 and listed[0] == queued.id
