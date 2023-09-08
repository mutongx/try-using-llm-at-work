import asyncio
import capnp

from proto.common_capnp import EvalOption, PredictOption
from proto.service_capnp import Context, App


class PredictCallback(Context.PredictCallback.Server):
    def __init__(self):
        self._queue = asyncio.Queue()

    async def callback(self, token, _context):
        return await self._queue.put(token)

    async def done(self, _context):
        return await self._queue.put(None)

    async def get(self):
        return await self._queue.get()


class LlamaClientWrapper:

    def __init__(self, addr: str):
        self._addr = addr
        self._client = None
        self._app = None

    async def __aenter__(self):
        host, port = self._addr.split(":")
        conn = await capnp.AsyncIoStream.create_connection(host=host, port=port)
        self._client = capnp.TwoPartyClient(conn)
        self._app = self._client.bootstrap().cast_as(App.Server)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def run_completion(self, prompt: str, eval_option: EvalOption, predict_option: PredictOption):
        model = self._app.getModel().model
        tokens = model.newTokenizer().tokenizer.tokenize(prompt).tokens
        callback = PredictCallback()
        prediction = model.newContext().context \
            .feedTokens(tokens=tokens).context \
            .predictUntilEos(callback=callback, evalOption=eval_option, predictOption=predict_option).a_wait()

        first = True
        while True:
            item = await callback.get()
            if item is None:
                break
            s = item.str
            if first:
                s = s.removeprefix(" ")
                first = False
            yield s

        await prediction
