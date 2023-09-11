import re
from typing import NamedTuple, List, Optional


class LogLine(NamedTuple):
    timestamp: Optional[str]
    text: str


class LogItem(NamedTuple):
    title: Optional[str]
    lines: List[LogLine]


class JenkinsLogParser:

    def __init__(self):
        self._items: List[LogItem] = [LogItem(title=None, lines=[])]

    def feed(self, line: str):
        if line.startswith("[Pipeline] "):
            self._items.append(LogItem(title=line, lines=[]))
        else:
            match = re.match(r"^\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.\d{3}Z)] (.*)", line)
            if match is not None:
                self._items[-1].lines.append(LogLine(match.group(1), match.group(2)))
            else:
                self._items[-1].lines.append(LogLine(None, line))

    def finalize(self):
        pass
