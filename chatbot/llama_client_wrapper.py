import asyncio
import capnp
from dataclasses import dataclass
from typing import List

from proto.common_capnp import EvalOption, PredictOption
from proto.service_capnp import App, Context, Token


@dataclass
class TokenizeResult:
    size: int
    token_id: List[int]
    token_pos: List[int]
    token_size: List[int]


@dataclass
class Token:
    id: int
    str: str


class PredictCallback(Context.PredictCallback.Server):
    def __init__(self):
        self._queue: asyncio.Queue[Token | None] = asyncio.Queue()

    async def callback(self, token: Token, _context):
        return await self._queue.put(Token(id=token.id, str=token.str))

    async def done(self, _context):
        return await self._queue.put(None)

    async def get(self):
        return await self._queue.get()


class LlamaClientWrapper:

    def __init__(self, addr: str):
        self._addr = addr

    async def __aenter__(self):
        host, port = self._addr.split(":")
        conn = await capnp.AsyncIoStream.create_connection(host=host, port=int(port))
        self._client = capnp.TwoPartyClient(conn)
        self._app = self._client.bootstrap().cast_as(App)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    @property
    def app(self):
        return self._app

    async def tokenize(self, prompt: str) -> TokenizeResult:
        model = self._app.getModel().model
        tokens = model.newTokenizer().tokenizer.tokenize(prompt).tokens
        size, token_id, token_pos, token_size = await asyncio.gather(
            tokens.getSize().a_wait(),
            tokens.getTokenId().a_wait(),
            tokens.getTokenPos().a_wait(),
            tokens.getTokenSize().a_wait(),
        )
        return TokenizeResult(
            size=size.to_dict()["result"],
            token_id=token_id.to_dict()["result"],
            token_pos=token_pos.to_dict()["result"],
            token_size=token_size.to_dict()["result"]
        )

    async def run_completion(self, prompt: str, eval_option: EvalOption, predict_option: PredictOption):
        model = self._app.getModel().model
        tokens = model.newTokenizer().tokenizer.tokenize(prompt).tokens
        callback = PredictCallback()

        predict = asyncio.create_task(
            model.newContext().context
            .feedBos().context
            .feedTokens(tokens=tokens).context
            .predictUntilEos(callback=callback, evalOption=eval_option,
                             predictOption=predict_option).a_wait(), name="predict")
        receive = asyncio.create_task(callback.get(), name="receive")

        pending = {predict, receive}
        first = True
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            if receive in done:
                item = await receive
                if item is not None:
                    s = item.str  # type: str
                    if first:
                        s = s.removeprefix(" ")
                        first = False
                    yield s
                    receive = asyncio.create_task(callback.get())
                    pending.add(receive)
                else:
                    pass
            if predict in done:
                await predict
