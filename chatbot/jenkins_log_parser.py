import re
from typing import Any, NamedTuple, List, Optional, Generator


class LogLine(NamedTuple):
    timestamp: Optional[str]
    text: str


class LogItem(NamedTuple):
    title: Optional[str]
    lines: List[LogLine]
    children: 'List[LogItem]'


class JenkinsLogParser:

    def __init__(self):
        self._items: List[LogItem] = [LogItem(title=None, lines=[], children=[])]
        self._finalized = False

    def feed(self, line: str):
        if line.startswith("[Pipeline] "):
            self._items.append(LogItem(title=line.removeprefix("[Pipeline] "), lines=[], children=[]))
        else:
            match = re.match(r"^\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.\d{3}Z)] (.*)", line)
            if match is not None:
                self._items[-1].lines.append(LogLine(timestamp=match.group(1), text=match.group(2)))
            else:
                self._items[-1].lines.append(LogLine(timestamp=None, text=line))

    def finalize(self):
        # This won't work correctly for pipelines with parallel stages
        if self._finalized:
            return
        stack = [self._items[0]]
        first = True
        for item in self._items:
            if first:
                first = False
                continue
            stack[-1].children.append(item)
            if item.title.startswith("{"):
                stack.append(item)
            if item.title.startswith("}"):
                stack.pop()
        if len(stack) != 1:
            raise RuntimeError("unable to parse pipeline tree")

    @property
    def items(self) -> Generator[LogItem, Any, Any]:
        for item in self._items:
            yield item

    @property
    def root(self) -> LogItem:
        return self._items[0]
