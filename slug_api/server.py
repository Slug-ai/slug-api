"""OpenAI-compatible HTTP API in front of an inference backend, for the ``openai`` package, OpenRouter and the like.

    GET  /v1/models
    POST /v1/chat/completions      {"messages": [...], "max_tokens"?, "stream"?}: a chat.completion, or its chunks as server-sent events
    POST /v1/files                 upload a .jsonl of requests (purpose "batch")
    GET  /v1/files[/{id}[/content]], DELETE /v1/files/{id}
    POST /v1/batches               {"input_file_id", "endpoint": "/v1/chat/completions", "completion_window": "24h"}
    GET  /v1/batches[/{id}]        ?wait=30 holds the request until the batch ends or 30 seconds pass (long polling)
    POST /v1/batches/{id}/cancel

Beyond OpenAI's format: ``?wait=``, and a batch's ``queue_position`` (where its next waiting request is in
line, 1 being next). Decoding is greedy, so sampling parameters (temperature, top_p, ...) are accepted and
ignored. Files and batches live in memory, for a day after they finish. With a key, every request needs
``Authorization: Bearer <key>``.

A backend runs the model (see ``Backend``). The API puts requests in its ``queue`` and calls ``fill``; the
backend takes them in order, appends each step's tokens to their sequences, marks them done, and calls
``on_step`` with the sequences that changed. If ``run`` raises, every request fails.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import secrets
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Protocol

from fastapi import Body, Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.exceptions import HTTPException

from .queue import Queue, Sequence

KEEP = 24 * 3600  # seconds files and batches are kept once they're done with
MAX_WAIT = 60.0  # longest a long poll is held
KEEPALIVE = 15.0  # seconds between comments on a quiet stream, so proxies keep it open
ENDED = {"completed", "failed", "cancelled", "expired"}


class Backend(Protocol):
    queue: Queue
    context: int  # most tokens a request can have, prompt plus completion
    max_tokens: int  # for a request that doesn't say
    ready: bool  # whether fill() can start work now
    on_step: Callable[[list[Sequence]], None]  # set by the API

    async def fill(self) -> None:
        """Requests are waiting: start them if there's room."""

    async def run(self) -> None:
        """Run until cancelled."""


class ApiError(Exception):
    def __init__(self, status: int, message: str, type: str = "invalid_request_error", param: str | None = None, code: str | None = None):
        super().__init__(message)
        self.status, self.body = status, {"error": {"message": message, "type": type, "param": param, "code": code}}


def new_id(prefix: str) -> str:
    return prefix + secrets.token_hex(12)


def pulse(watched: Generation | Batch) -> None:
    """Wake whatever is waiting on ``watched.changed``."""
    changed, watched.changed = watched.changed, asyncio.Event()
    changed.set()


async def changes(event: asyncio.Event, timeout: float) -> bool:
    """Wait for a pulse on ``event`` (a ``changed``, taken before checking what it guards): False if ``timeout`` seconds pass first."""
    try:
        await asyncio.wait_for(event.wait(), timeout)
        return True
    except TimeoutError:
        return False


@dataclass(eq=False)
class Generation:  # one request being generated
    seq: Sequence
    batch: Batch | None = None
    custom_id: str | None = None
    cancelled: bool = False
    changed: asyncio.Event = field(default_factory=asyncio.Event)  # pulsed on every step


@dataclass
class StoredFile:
    id: str
    filename: str
    purpose: str
    data: bytes
    created_at: int = field(default_factory=lambda: int(time.time()))

    def object(self) -> dict:
        return {"id": self.id, "object": "file", "bytes": len(self.data), "created_at": self.created_at, "filename": self.filename, "purpose": self.purpose, "status": "processed"}


@dataclass(eq=False)
class Batch:
    id: str
    input_file_id: str
    endpoint: str
    completion_window: str
    metadata: dict | None
    status: str = "validating"
    gens: list[Generation] = field(default_factory=list)
    errors: list[dict] | None = None
    output_file_id: str | None = None
    times: dict[str, int] = field(default_factory=lambda: {"created": int(time.time())})  # "created", "in_progress", ... -> unix time
    changed: asyncio.Event = field(default_factory=asyncio.Event)  # pulsed as its requests finish

    def at(self, event: str) -> None:
        self.times[event] = int(time.time())


class Server:
    def __init__(self, backend: Backend, encode: Callable[[list[dict]], list[int]], decode: Callable[[list[int]], str], model: str, key: str | None = None):
        self.backend, self.encode, self.decode, self.model, self.key = backend, encode, decode, model, key
        self.gens: dict[int, Generation] = {}  # sequence id -> its generation, until it's done
        self.files: dict[str, StoredFile] = {}
        self.batches: dict[str, Batch] = {}
        self.failure: str | None = None  # why the backend stopped, once it has
        backend.on_step = self.on_step
        self.app = self.routes()

    # generations

    def on_step(self, seqs: list[Sequence]) -> None:
        for seq in seqs:
            if gen := self.gens.get(seq.id):
                if seq.done:
                    self.finish(gen)
                pulse(gen)

    def finish(self, gen: Generation) -> None:
        self.gens.pop(gen.seq.id, None)
        self.backend.queue.forget(gen.seq)
        if gen.batch and all(g.seq.done for g in gen.batch.gens):
            self.finalize(gen.batch)

    def cancel(self, gen: Generation) -> None:
        if not gen.seq.done:
            gen.cancelled = True
            self.backend.queue.cancel(gen.seq)
            self.finish(gen)
            pulse(gen)

    def running(self) -> None:
        if self.failure:
            raise ApiError(503, self.failure, "server_error")

    def parse(self, body: dict) -> tuple[list[int], int]:
        """A chat completion request's prompt and max tokens."""
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages or not all(isinstance(m, dict) for m in messages):
            raise ApiError(400, "messages must be a non-empty list of messages", param="messages")
        if body.get("n", 1) != 1:
            raise ApiError(400, "only n=1 is supported", param="n")

        def text(content) -> str:  # a string, or parts like {"type": "text", "text": ...}
            return content if isinstance(content, str) else "".join(p.get("text", "") for p in content or [] if isinstance(p, dict))

        prompt = self.encode([{"role": m.get("role", "user"), "content": text(m.get("content"))} for m in messages])
        room = self.backend.context - len(prompt)
        max_tokens = body.get("max_completion_tokens") or body.get("max_tokens") or min(self.backend.max_tokens, room)
        if not isinstance(max_tokens, int) or not 0 < max_tokens <= room:
            raise ApiError(400, f"the prompt is {len(prompt)} tokens: with max_tokens {max_tokens} it's over the context ({self.backend.context})", param="max_tokens")
        return prompt, max_tokens

    def add(self, prompt: list[int], max_tokens: int, **kw) -> Generation:
        gen = Generation(self.backend.queue.add(prompt, max_tokens), **kw)
        self.gens[gen.seq.id] = gen
        return gen

    def finish_reason(self, seq: Sequence) -> str:
        return "stop" if seq.tokens and seq.tokens[-1] in self.backend.queue.eos else "length"

    def usage(self, seq: Sequence) -> dict:
        return {"prompt_tokens": len(seq.prompt), "completion_tokens": len(seq.tokens), "total_tokens": len(seq.prompt) + len(seq.tokens)}

    def completion(self, gen: Generation, id: str, created: int) -> dict:
        message = {"role": "assistant", "content": self.decode(gen.seq.tokens)}
        choice = {"index": 0, "message": message, "finish_reason": self.finish_reason(gen.seq), "logprobs": None}
        return {"id": id, "object": "chat.completion", "created": created, "model": self.model, "choices": [choice], "usage": self.usage(gen.seq)}

    async def stream(self, gen: Generation, id: str, created: int, usage: bool) -> AsyncIterator[str]:
        def data(obj: dict) -> str:
            return f"data: {json.dumps(obj)}\n\n"

        def chunk(delta: dict, finish: str | None = None) -> str:
            choice = {"index": 0, "delta": delta, "finish_reason": finish, "logprobs": None}
            return data({"id": id, "object": "chat.completion.chunk", "created": created, "model": self.model, "choices": [choice]})

        try:
            yield chunk({"role": "assistant", "content": ""})
            sent = ""
            while True:
                changed, done = gen.changed, gen.seq.done
                text = self.decode(gen.seq.tokens)
                # a character split across tokens decodes as U+FFFD until its last token comes
                if text.startswith(sent) and len(text) > len(sent) and (done or not text.endswith("�")):
                    yield chunk({"content": text[len(sent) :]})
                    sent = text
                if done:
                    break
                if self.failure:
                    yield data(ApiError(503, self.failure, "server_error").body)
                    return
                if not await changes(changed, KEEPALIVE):
                    yield ": keepalive\n\n"
            yield chunk({}, self.finish_reason(gen.seq))
            if usage:
                yield data({"id": id, "object": "chat.completion.chunk", "created": created, "model": self.model, "choices": [], "usage": self.usage(gen.seq)})
            yield "data: [DONE]\n\n"
        finally:
            self.cancel(gen)  # the client left early: stop generating for it

    # batches

    def create_batch(self, batch: Batch, data: bytes) -> None:
        """Check every request in the input file, then queue them (a batch with a bad request fails whole, as OpenAI's do)."""
        requests, errors, seen = [], [], set()
        for line, raw in enumerate(data.decode(errors="replace").splitlines(), 1):
            if not raw.strip():
                continue
            try:
                r = json.loads(raw)
                if not isinstance(r, dict) or not isinstance(r.get("custom_id"), str) or r["custom_id"] in seen:
                    raise ApiError(400, "each request needs a unique custom_id")
                if r.get("method", "POST") != "POST" or r.get("url") != batch.endpoint:
                    raise ApiError(400, f"method must be POST and url {batch.endpoint}")
                seen.add(r["custom_id"])
                requests.append((r["custom_id"], *self.parse(r.get("body") or {})))
            except (ValueError, ApiError) as e:
                errors.append({"code": "invalid_request", "message": str(e), "param": None, "line": line})
        if errors or not requests:
            batch.status, batch.errors = "failed", errors or [{"code": "empty_file", "message": "the input file has no requests", "param": None, "line": None}]
            batch.at("failed")
            return
        batch.gens = [self.add(prompt, max_tokens, batch=batch, custom_id=cid) for cid, prompt, max_tokens in requests]
        batch.status = "in_progress"
        batch.at("in_progress")

    def finalize(self, batch: Batch) -> None:
        """Write the finished requests' responses to the output file."""
        batch.at("finalizing")
        lines = []
        for gen in batch.gens:
            if not gen.cancelled:
                body = self.completion(gen, new_id("chatcmpl-"), batch.times["created"])
                response = {"status_code": 200, "request_id": new_id("req_"), "body": body}
                lines.append(json.dumps({"id": new_id("batch_req_"), "custom_id": gen.custom_id, "response": response, "error": None}) + "\n")
        if lines:
            out = StoredFile(new_id("file-"), f"{batch.id}_output.jsonl", "batch_output", "".join(lines).encode())
            self.files[out.id], batch.output_file_id = out, out.id
        batch.status = "cancelled" if batch.status == "cancelling" else "completed"
        batch.at(batch.status)
        pulse(batch)

    def batch_object(self, batch: Batch) -> dict:
        queue = self.backend.queue.waiting
        waiting = [queue.index(g.seq) + 1 for g in batch.gens if not g.seq.done and g.seq in queue]
        done = sum(g.seq.done and not g.cancelled for g in batch.gens)
        events = ("in_progress", "finalizing", "completed", "failed", "expired", "cancelling", "cancelled")
        return {
            "id": batch.id,
            "object": "batch",
            "endpoint": batch.endpoint,
            "errors": {"object": "list", "data": batch.errors} if batch.errors else None,
            "input_file_id": batch.input_file_id,
            "completion_window": batch.completion_window,
            "status": batch.status,
            "output_file_id": batch.output_file_id,
            "error_file_id": None,
            "created_at": batch.times["created"],
            "expires_at": batch.times["created"] + 24 * 3600,
            **{f"{e}_at": batch.times.get(e) for e in events},
            "request_counts": {"total": len(batch.gens), "completed": done, "failed": 0},
            "metadata": batch.metadata,
            "queue_position": min(waiting, default=None),
        }

    def fail_everything(self) -> None:
        for gen in list(self.gens.values()):
            pulse(gen)
        for batch in self.batches.values():
            if batch.status not in ENDED:
                batch.status, batch.errors = "failed", [{"code": "server_error", "message": self.failure, "param": None, "line": None}]
                batch.at("failed")
                pulse(batch)

    def prune(self) -> None:
        """Forget batches a day after they end, with their files, and files a day old that no live batch reads."""
        old = int(time.time()) - KEEP
        for batch in [b for b in self.batches.values() if b.status in ENDED and max(b.times.values()) < old]:
            del self.batches[batch.id]
            for id in (batch.input_file_id, batch.output_file_id):
                self.files.pop(id, None)
        live = {b.input_file_id for b in self.batches.values() if b.status not in ENDED}
        for f in [f for f in self.files.values() if f.created_at < old and f.id not in live]:
            del self.files[f.id]

    # HTTP

    def find(self, table: dict, id: str, what: str):
        if item := table.get(id):
            return item
        raise ApiError(404, f"no {what} {id}", code="not_found")

    def routes(self) -> FastAPI:
        @asynccontextmanager
        async def lifespan(_: FastAPI) -> AsyncIterator[None]:
            def stopped(task: asyncio.Task) -> None:
                if not task.cancelled() and (e := task.exception()):
                    self.failure = f"the backend stopped: {e!r}"
                    self.fail_everything()

            task = asyncio.create_task(self.backend.run())
            task.add_done_callback(stopped)
            yield
            task.cancel()

        def authorize(request: Request) -> None:
            if self.key and not secrets.compare_digest(request.headers.get("authorization", ""), f"Bearer {self.key}"):
                raise ApiError(401, "missing or wrong API key", code="invalid_api_key")

        app = FastAPI(title="slugai", lifespan=lifespan, dependencies=[Depends(authorize)])

        @app.exception_handler(ApiError)
        async def api_error(_: Request, e: ApiError) -> JSONResponse:
            return JSONResponse(e.body, e.status)

        @app.exception_handler(HTTPException)
        async def http_error(_: Request, e: HTTPException) -> JSONResponse:
            return JSONResponse(ApiError(e.status_code, str(e.detail)).body, e.status_code)

        @app.exception_handler(RequestValidationError)
        async def validation_error(_: Request, e: RequestValidationError) -> JSONResponse:
            return JSONResponse(ApiError(400, str(e.errors())).body, 400)

        def model_object() -> dict:
            return {"id": self.model, "object": "model", "created": 0, "owned_by": "slugai"}

        @app.get("/v1/models")
        async def models() -> dict:
            return {"object": "list", "data": [model_object()]}

        @app.get("/v1/models/{id:path}")
        async def model(id: str) -> dict:
            if id != self.model:
                raise ApiError(404, f"no model {id}", code="model_not_found")
            return model_object()

        @app.post("/v1/chat/completions", response_model=None)
        async def chat(body: dict = Body(...)) -> dict | StreamingResponse:
            self.running()
            gen = self.add(*self.parse(body))
            await self.backend.fill()
            id, created = new_id("chatcmpl-"), int(time.time())
            if body.get("stream"):
                usage = bool((body.get("stream_options") or {}).get("include_usage"))
                return StreamingResponse(self.stream(gen, id, created, usage), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
            try:
                while not gen.seq.done:
                    self.running()
                    await changes(gen.changed, KEEPALIVE)
            finally:
                self.cancel(gen)
            return self.completion(gen, id, created)

        @app.post("/v1/files")
        async def upload(file: UploadFile = File(...), purpose: str = Form(...)) -> dict:
            self.prune()
            stored = StoredFile(new_id("file-"), file.filename or "upload.jsonl", purpose, await file.read())
            self.files[stored.id] = stored
            return stored.object()

        @app.get("/v1/files")
        async def list_files(purpose: str | None = None) -> dict:
            return {"object": "list", "data": [f.object() for f in self.files.values() if purpose in (None, f.purpose)], "has_more": False}

        @app.get("/v1/files/{id}")
        async def get_file(id: str) -> dict:
            return self.find(self.files, id, "file").object()

        @app.get("/v1/files/{id}/content")
        async def file_content(id: str) -> Response:
            return Response(self.find(self.files, id, "file").data, media_type="application/jsonl")

        @app.delete("/v1/files/{id}")
        async def delete_file(id: str) -> dict:
            self.files.pop(self.find(self.files, id, "file").id)
            return {"id": id, "object": "file", "deleted": True}

        @app.post("/v1/batches")
        async def create(body: dict = Body(...)) -> dict:
            self.running()
            self.prune()
            if body.get("endpoint") != "/v1/chat/completions":
                raise ApiError(400, "only the /v1/chat/completions endpoint is supported", param="endpoint")
            file = self.find(self.files, body.get("input_file_id"), "file")
            batch = Batch(new_id("batch_"), file.id, body["endpoint"], body.get("completion_window", "24h"), body.get("metadata"))
            self.batches[batch.id] = batch
            self.create_batch(batch, file.data)
            await self.backend.fill()
            return self.batch_object(batch)

        @app.get("/v1/batches")
        async def list_batches(limit: int = 20, after: str | None = None) -> dict:
            batches = list(self.batches.values())[::-1]  # newest first
            ids = [b.id for b in batches]
            start = ids.index(after) + 1 if after in ids else 0
            page = [self.batch_object(b) for b in batches[start : start + limit]]
            return {"object": "list", "data": page, "first_id": page[0]["id"] if page else None, "last_id": page[-1]["id"] if page else None, "has_more": start + limit < len(batches)}

        @app.get("/v1/batches/{id}")
        async def get_batch(id: str, wait: float = 0.0) -> dict:
            batch = self.find(self.batches, id, "batch")
            deadline = time.monotonic() + min(max(wait, 0.0), MAX_WAIT)
            while batch.status not in ENDED and (left := deadline - time.monotonic()) > 0:
                await changes(batch.changed, left)
            return self.batch_object(batch)

        @app.post("/v1/batches/{id}/cancel")
        async def cancel(id: str) -> dict:
            batch = self.find(self.batches, id, "batch")
            if batch.status == "in_progress":
                batch.status = "cancelling"
                batch.at("cancelling")
                for gen in batch.gens:
                    self.cancel(gen)  # the last one finalizes the batch
            return self.batch_object(batch)

        return app


def serve(backend: Backend, encode: Callable[[list[dict]], list[int]], decode: Callable[[list[int]], str], model: str, key: str | None = None, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Serve ``backend`` until interrupted. ``encode`` turns chat messages into a prompt's token ids, ``decode`` tokens back into text."""
    import uvicorn

    uvicorn.run(Server(backend, encode, decode, model, key).app, host=host, port=port)
