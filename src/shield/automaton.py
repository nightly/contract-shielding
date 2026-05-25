from __future__ import annotations

import importlib
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from typing import Callable

from .core import DeterministicSafetyAutomaton, Label


class AutomatonBackendError(RuntimeError):
    pass


class UnsupportedFormulaError(AutomatonBackendError):
    pass


class BackendUnavailableError(AutomatonBackendError):
    pass


REQUIRED_SPOT_COMMANDS = ("ltlfilt", "ltl2tgba", "autfilt")


def missing_spot_commands(
    *,
    which: Callable[[str], str | None] | None = None,
    commands: tuple[str, ...] = REQUIRED_SPOT_COMMANDS,
) -> tuple[str, ...]:
    resolve = which or shutil.which
    return tuple(command for command in commands if resolve(command) is None)


def spot_cli_available(
    *,
    which: Callable[[str], str | None] | None = None,
    commands: tuple[str, ...] = REQUIRED_SPOT_COMMANDS,
) -> bool:
    return not missing_spot_commands(which=which, commands=commands)


def require_spot_cli(
    *,
    which: Callable[[str], str | None] | None = None,
    commands: tuple[str, ...] = REQUIRED_SPOT_COMMANDS,
) -> None:
    missing = missing_spot_commands(which=which, commands=commands)
    if not missing:
        return
    raise BackendUnavailableError(
        "Spot is required for shielding and contract synthesis, but these Spot "
        f"CLI command(s) are missing from PATH: {', '.join(missing)}. "
        "Install Spot so ltlfilt, ltl2tgba, and autfilt are available before "
        "running monitored, shielded, or contract experiments."
    )


@dataclass(frozen=True)
class ParsedMonitor:
    start_state: int
    states: frozenset[int]
    transitions: dict[int, tuple[tuple[str, int], ...]]
    atomic_props: tuple[str, ...]


def _default_run(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _compile_label_expr(expr: str) -> Callable[[frozenset[int]], bool]:
    if expr == "t":
        return lambda _: True
    if expr == "f":
        return lambda _: False

    python_expr = expr
    python_expr = re.sub(r"(?<!\w)(\d+)(?!\w)", r"(\1 in valuation)", python_expr)
    python_expr = python_expr.replace("!", " not ")
    python_expr = python_expr.replace("&", " and ")
    python_expr = python_expr.replace("|", " or ")

    def evaluate(valuation: frozenset[int]) -> bool:
        return bool(eval(python_expr, {"__builtins__": {}}, {"valuation": valuation}))

    return evaluate


def parse_hoa_monitor(hoa: str) -> ParsedMonitor:
    states: set[int] = set()
    transitions: dict[int, list[tuple[str, int]]] = {}
    start_state: int | None = None
    atomic_props: tuple[str, ...] = ()
    current_state: int | None = None

    for line in hoa.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("Start:"):
            start_state = int(stripped.split(":", maxsplit=1)[1].strip())
            continue
        if stripped.startswith("AP:"):
            ap_line = stripped.split(":", maxsplit=1)[1].strip()
            count_str, *names = shlex.split(ap_line)
            count = int(count_str)
            if len(names) != count:
                raise AutomatonBackendError("Malformed HOA AP header.")
            atomic_props = tuple(names)
            continue
        if stripped.startswith("State:"):
            current_state = int(stripped.split()[1])
            states.add(current_state)
            transitions.setdefault(current_state, [])
            continue
        if stripped.startswith("--END--"):
            break
        if stripped.startswith("["):
            if current_state is None:
                raise AutomatonBackendError("Transition appeared before any HOA state.")
            match = re.match(r"\[(.*)\]\s+(\d+)", stripped)
            if match is None:
                raise AutomatonBackendError(f"Unsupported HOA transition line: {stripped}")
            label_expr, target = match.groups()
            transitions[current_state].append((label_expr.strip(), int(target)))

    if start_state is None:
        raise AutomatonBackendError("Missing HOA Start state.")

    return ParsedMonitor(
        start_state=start_state,
        states=frozenset(states),
        transitions={state: tuple(edges) for state, edges in transitions.items()},
        atomic_props=atomic_props,
    )


def complete_monitor(
    monitor: ParsedMonitor,
    atomic_props: tuple[str, ...],
) -> DeterministicSafetyAutomaton:
    sorted_props = tuple(sorted(dict.fromkeys(atomic_props)))
    monitor_prop_index = {ap: idx for idx, ap in enumerate(monitor.atomic_props)}
    all_labels: list[Label] = []
    for bits in range(1 << len(sorted_props)):
        all_labels.append(
            frozenset(
                prop
                for idx, prop in enumerate(sorted_props)
                if bits & (1 << idx)
            )
        )

    sink_state = (max(monitor.states) + 1) if monitor.states else 0
    all_states = set(monitor.states)
    all_states.add(sink_state)
    transition_map: dict[tuple[int, Label], int] = {}

    compiled_edges: dict[int, tuple[tuple[Callable[[frozenset[int]], bool], int], ...]] = {
        state: tuple((_compile_label_expr(expr), target) for expr, target in edges)
        for state, edges in monitor.transitions.items()
    }

    for state in monitor.states:
        for label in all_labels:
            tool_valuation = frozenset(
                monitor_prop_index[prop] for prop in label if prop in monitor_prop_index
            )
            matches = [
                target
                for predicate, target in compiled_edges.get(state, ())
                if predicate(tool_valuation)
            ]
            if len(matches) > 1:
                raise AutomatonBackendError(f"Monitor is non-deterministic in state {state}.")
            transition_map[(state, label)] = matches[0] if matches else sink_state

    for label in all_labels:
        transition_map[(sink_state, label)] = sink_state

    safe_states = frozenset(monitor.states)
    return DeterministicSafetyAutomaton(
        atomic_props=sorted_props,
        states=frozenset(all_states),
        initial_state=monitor.start_state,
        safe_states=safe_states,
        transition_map=transition_map,
    )


class SpotAutomatonBackend:
    def __init__(
        self,
        *,
        ltl2tgba_cmd: str = "ltl2tgba",
        ltlfilt_cmd: str = "ltlfilt",
        run_command: Callable[[list[str]], str] | None = None,
        which: Callable[[str], str | None] | None = None,
        spot_module: object | None = None,
    ) -> None:
        self.ltl2tgba_cmd = ltl2tgba_cmd
        self.ltlfilt_cmd = ltlfilt_cmd
        self.run_command = run_command or _default_run
        self.which = which or shutil.which
        self._spot_module = spot_module

    def compile(
        self,
        formula: str,
        atomic_props: tuple[str, ...] | list[str] | set[str],
    ) -> DeterministicSafetyAutomaton:
        prop_tuple = tuple(sorted(dict.fromkeys(atomic_props)))
        require_spot_cli(
            which=self.which,
            commands=(self.ltlfilt_cmd, self.ltl2tgba_cmd, "autfilt"),
        )

        spot_module = self._spot_module
        if spot_module is None:
            try:
                spot_module = importlib.import_module("spot")
            except ImportError:
                spot_module = None

        if spot_module is not None:
            return self._compile_with_python_spot(spot_module, formula, prop_tuple)
        return self._compile_with_cli(formula, prop_tuple)

    def _compile_with_python_spot(
        self,
        spot_module: object,
        formula: str,
        atomic_props: tuple[str, ...],
    ) -> DeterministicSafetyAutomaton:
        spot = spot_module
        parsed = spot.formula(formula)
        if not parsed.is_syntactic_safety():
            raise UnsupportedFormulaError(
                "Formula is not in Spot's syntactic safety fragment."
            )
        hoa = spot.translate(formula, "monitor", "det").to_str("hoa")
        return complete_monitor(parse_hoa_monitor(hoa), atomic_props)

    def _compile_with_cli(
        self,
        formula: str,
        atomic_props: tuple[str, ...],
    ) -> DeterministicSafetyAutomaton:
        require_spot_cli(
            which=self.which,
            commands=(self.ltlfilt_cmd, self.ltl2tgba_cmd, "autfilt"),
        )

        safety_count = self.run_command(
            [self.ltlfilt_cmd, "--count", "--safety", "-f", formula]
        ).strip()
        if safety_count != "1":
            raise UnsupportedFormulaError("Formula is not a safety property.")

        hoa = self.run_command([self.ltl2tgba_cmd, "-D", "-M", formula])
        return complete_monitor(parse_hoa_monitor(hoa), atomic_props)
