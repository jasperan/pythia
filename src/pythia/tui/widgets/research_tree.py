"""Research tree — live visualization of multi-round research progress."""

from __future__ import annotations

from enum import Enum, auto

from rich.text import Text
from textual.widgets import Static
from pythia.tui import colors


class NodeState(Enum):
    PENDING = auto()
    SEARCHING = auto()
    COMPLETE = auto()


_ICONS = {
    NodeState.PENDING: ("○", f"{colors.DIM}"),
    NodeState.SEARCHING: ("◎", f"{colors.INFO}"),
    NodeState.COMPLETE: ("◉", f"{colors.SUCCESS}"),
}


class ResearchTree(Static):
    DEFAULT_CSS = """
    ResearchTree {
        width: 100%;
        height: auto;
        padding: 1 1;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__("", **kwargs)
        self._rounds: list[dict] = []
        self._active_round: int = 0
        self._recall_count: int = 0
        self._recall_items: list[dict] = []

    def reset(self) -> None:
        self._rounds = []
        self._active_round = 0
        self._recall_count = 0
        self._recall_items = []
        if self.is_attached:
            self.update("")

    def set_recall(self, items: list[dict]) -> None:
        self._recall_items = items
        self._recall_count = len(items)
        self._rebuild()

    def add_plan(self, sub_queries: list[str]) -> None:
        round_data = {
            "round_num": len(self._rounds) + 1,
            "sub_queries": [
                {"query": sq, "state": NodeState.PENDING, "num_sources": 0, "preview": ""}
                for sq in sub_queries
            ],
        }
        self._rounds.append(round_data)
        self._active_round = len(self._rounds)
        self._rebuild()

    def start_round(self, round_num: int, max_rounds: int) -> None:
        self._active_round = round_num
        self._rebuild()

    def complete_finding(self, query: str, num_sources: int = 0, preview: str = "") -> None:
        for rnd in self._rounds:
            for sq in rnd["sub_queries"]:
                if sq["query"] == query:
                    sq["state"] = NodeState.COMPLETE
                    sq["num_sources"] = num_sources
                    sq["preview"] = preview
                    self._rebuild()
                    return

    def add_gaps(self, gap_queries: list[str], reasoning: str = "") -> None:
        round_data = {
            "round_num": len(self._rounds) + 1,
            "sub_queries": [
                {"query": sq, "state": NodeState.PENDING, "num_sources": 0, "preview": ""}
                for sq in gap_queries
            ],
            "reasoning": reasoning,
        }
        self._rounds.append(round_data)
        self._rebuild()

    def mark_complete(self) -> None:
        self._rebuild()

    def _rebuild(self) -> None:
        text = Text()

        if self._recall_count > 0:
            text.append("  \U0001f9e0 ", style="bold")
            text.append(f"Recalled {self._recall_count} prior finding(s)\n", style=f"{colors.SECONDARY}")
            for item in self._recall_items:
                text.append("    └ ", style=f"{colors.DIM}")
                text.append(f"{item.get('from_query', '?')}", style=f"{colors.MUTED}")
                sim = item.get("similarity", 0)
                text.append(f" ({sim:.0%})\n", style=f"{colors.DIM}")
            text.append("\n")

        max_rounds = len(self._rounds)
        for rnd in self._rounds:
            rnum = rnd["round_num"]
            is_active = rnum == self._active_round
            is_done = all(sq["state"] == NodeState.COMPLETE for sq in rnd["sub_queries"])

            if is_active and not is_done:
                text.append(f"  ● Round {rnum}/{max_rounds}\n", style=f"bold {colors.TEXT}")
            elif is_done:
                text.append(f"  ● Round {rnum}/{max_rounds}\n", style=f"{colors.MUTED}")
            else:
                text.append(f"  ○ Round {rnum}/{max_rounds}\n", style=f"{colors.DIM}")

            for i, sq in enumerate(rnd["sub_queries"]):
                is_last = i == len(rnd["sub_queries"]) - 1
                branch = "└─" if is_last else "├─"
                icon, color = _ICONS[sq["state"]]

                text.append(f"  {branch} ", style=f"{colors.DIM}")
                text.append(f"{icon} ", style=color)

                q = sq["query"]
                if len(q) > 40:
                    q = q[:37] + "..."
                text.append(f"{q}", style=color)

                if sq["state"] == NodeState.COMPLETE and sq["num_sources"] > 0:
                    text.append(f" ({sq['num_sources']} sources)", style=f"{colors.DIM}")
                elif sq["state"] == NodeState.SEARCHING:
                    text.append(" searching...", style=f"{colors.INFO}")

                text.append("\n")

            reasoning = rnd.get("reasoning", "")
            if reasoning and rnd["round_num"] > 1:
                r = reasoning[:50] + "..." if len(reasoning) > 50 else reasoning
                text.append(f"    \u2139 {r}\n", style=f"{colors.DIM}")

            text.append("\n")

        if self.is_attached:
            self.update(text)
