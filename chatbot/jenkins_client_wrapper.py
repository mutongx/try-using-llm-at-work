import aiohttp


class JenkinsClientWrapper:
    def __init__(self, url: str, login: str, password: str):
        self._url = url
        self._login = login
        self._password = password

    async def __aenter__(self):
        self._session = aiohttp.ClientSession(auth=aiohttp.BasicAuth(login=self._login, password=self._password))
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._session.close()

    async def fetch_logs(self, job: str, build: str):
        job = "".join("/job/" + s for s in job.split("/"))
        async with self._session.get(f"{self._url}/{job}/{build}/logText/progressiveText", params={"start": 0}) as resp:
            resp.raise_for_status()
            while not resp.content.at_eof():
                yield (await resp.content.readline()).strip(b"\r\n")
