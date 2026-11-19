"""Layered lesson retrieval for CugaLite: a port of AppWorld's lessons_v2 memory
(the lessons-v2-retriever branch of github.com/jayaramkr/appworld:
experiments/code/simplified/react_code_agent_with_lesson_retrieval.py). See APPWORLD_LESSONS_V2.md.

Opt-in. Set CUGA_LESSON_INDEX to an index built by lesson_retrieval/build_lesson_index.py and CUGA_LESSON_TOOL_MAP to a
JSON object mapping CugaLite tool names to AppWorld API names ("<app>.<api>"). Unset (the default), nothing is
imported and nothing is injected.

  Layer 1 (answer)    the top-support answer lessons, appended to the special instructions (section E).
  Layer 4 (start)     lessons retrieved once from the task instruction, appended after them (section F).
  Layer 2 (api_class) appended to the execution output the first time a block calls a paginated-list or write API,
                      or find_tools lists one; once per class per task.
  Layer 3 (api / app) appended to the same output: lessons tagged with that API or its app, reranked against the
                      agent's text before the code, top k above the cutoff.

At most CUGA_LESSON_MAX_PER_STEP notes per block; no lesson is shown twice in a task. Retrieval failures never break a
run. What each task was shown is written to $CUGA_LOGGING_DIR/lesson_retrieval/<thread>.json.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from loguru import logger

INDEX_PATH = os.environ.get("CUGA_LESSON_INDEX")
CLASS_DESCRIPTIONS = {"paginated_list": "returns results one page at a time", "write": "changes data"}
TOOL_CALL = re.compile(r"\b([a-z][a-z0-9_]*)\s*\(")
CODE_FENCE = re.compile(r"```.*?```", re.S)

_index: dict | None = None
_retriever: Any = None
_tool_to_api: dict[str, str] = {}
_tasks: dict[str, dict] = {}


def _env_float(name: str, default: float | None) -> float | None:
    value = os.environ.get(name)
    return default if value is None or value == "" else float(value)


def _settings() -> dict:
    # Defaults are the settings used with the AppWorld ReAct agent (lesson_retrieval/example_config.jsonnet there).
    return {
        "max_answer": int(os.environ.get("CUGA_LESSON_MAX_ANSWER", "2")),
        "max_per_step": int(os.environ.get("CUGA_LESSON_MAX_PER_STEP", "4")),
        "class_once_per_task": os.environ.get("CUGA_LESSON_CLASS_ONCE_PER_TASK", "1") == "1",
    }


def _load() -> None:
    global _index, _retriever, _tool_to_api
    from cuga.backend.cuga_graph.nodes.cuga_lite.lesson_retriever import LessonRetriever, RetrieverConfig

    with open(INDEX_PATH) as handle:
        _index = json.load(handle)
    with open(os.environ["CUGA_LESSON_TOOL_MAP"]) as handle:
        _tool_to_api = json.load(handle)
    config = RetrieverConfig(
        embedder=os.environ.get("CUGA_LESSON_EMBEDDER", "BAAI/bge-base-en-v1.5"),
        use_bm25=True,
        reranker=os.environ.get("CUGA_LESSON_RERANKER", "cross-encoder/ms-marco-MiniLM-L-12-v2"),
        device=os.environ.get("CUGA_LESSON_DEVICE", "cpu"),
        start_k=int(os.environ.get("CUGA_LESSON_START_K", "5")),
        start_threshold=_env_float("CUGA_LESSON_START_THRESHOLD", -10.5),
        api_k=int(os.environ.get("CUGA_LESSON_API_K", "3")),
        api_threshold=_env_float("CUGA_LESSON_API_THRESHOLD", 2.0),
        class_k=int(os.environ.get("CUGA_LESSON_CLASS_K", "1")),
    )
    # Built and warmed at import, before any task: AppWorld code runs with time frozen.
    _retriever = LessonRetriever(_index, config)
    _retriever.warm()
    logger.info(
        f"[LESSONS] index {INDEX_PATH}: {sum(len(v) for v in _index.values() if isinstance(v, list))} list entries; "
        f"{len(_tool_to_api)} tool names mapped"
    )


if INDEX_PATH:
    try:
        _load()
    except Exception as exception:  # a broken index disables retrieval; it never breaks the agent
        logger.error(f"[LESSONS] disabled, failed to load {INDEX_PATH}: {exception!r}")
        INDEX_PATH = None


def enabled() -> bool:
    return bool(INDEX_PATH) and _retriever is not None


def bullet(entry: dict) -> str:
    """Lead with the trigger unless the lesson already opens with its own condition."""
    lesson, trigger = entry["lesson"].strip(), (entry.get("trigger") or "").strip().rstrip(".")
    if not trigger or re.match(r"(when|whenever|if|before|after)\b", lesson, re.I):
        return f"- {lesson}"
    return f"- {trigger}: {lesson}"


def _task(thread_id: str) -> dict:
    return _tasks.setdefault(
        thread_id,
        {
            "shown": set(),
            "seen_apis": set(),
            "served_classes": set(),
            "log": {"thread_id": thread_id, "events": []},
        },
    )


def _write_log(thread_id: str) -> None:
    directory = os.environ.get("CUGA_LOGGING_DIR")
    if not directory:
        return
    os.makedirs(os.path.join(directory, "lesson_retrieval"), exist_ok=True)
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", thread_id) or "default"
    with open(os.path.join(directory, "lesson_retrieval", f"{name}.json"), "w") as handle:
        json.dump(_tasks[thread_id]["log"], handle, indent=1)


def start_sections(thread_id: str, instruction: str) -> str:
    """Sections E (answer) and F (start) for a fresh task, or "" when disabled or nothing fits."""
    if not enabled():
        return ""
    _tasks.pop(thread_id, None)
    task = _task(thread_id)
    task["log"]["instruction"] = instruction
    sections: list[str] = []
    try:
        answer = _retriever.answer_entries()[: _settings()["max_answer"]]
        if answer:
            sections.append(
                "E. Notes from earlier tasks about finishing a task (use them when you give the final answer):\n\n"
                + "\n".join(bullet(e) for e in answer)
            )
        start = _retriever.start(instruction)
        if start:
            sections.append(
                "F. Notes from earlier tasks with similar instructions (use only the ones that fit this task):\n\n"
                + "\n".join(bullet(e) for e, _ in start)
            )
        task["shown"].update(e["lesson"] for e in answer)
        task["shown"].update(e["lesson"] for e, _ in start)
        task["log"]["answer"] = [e["lesson"] for e in answer]
        task["log"]["start"] = [{"lesson": e["lesson"], "score": s} for e, s in start]
    except Exception as exception:
        task["log"]["start_error"] = repr(exception)
        sections = []
    _write_log(thread_id)
    return "\n\n" + "\n\n".join(sections) if sections else ""


def _apis_in_block(code: str, output: str) -> list[str]:
    """AppWorld APIs the block called (in code order), then any that a find_tools listing in the output named."""
    apis: list[str] = []
    for match in TOOL_CALL.finditer(code):
        api = _tool_to_api.get(match.group(1))
        if api and api not in apis:
            apis.append(api)
    if "find_tools(" in code:
        positions = sorted((output.find(name), api) for name, api in _tool_to_api.items() if name in output)
        for _, api in positions:
            if api not in apis:
                apis.append(api)
    return apis


def event_block(thread_id: str, code: str, output: str, thought: str) -> str:
    """API-class and API notes for one executed block, or ""."""
    if not enabled():
        return ""
    task = _task(thread_id)
    settings = _settings()
    blocks: list[str] = []
    try:
        budget = settings["max_per_step"]
        for api in _apis_in_block(code or "", output or ""):
            if api in task["seen_apis"]:
                continue
            task["seen_apis"].add(api)
            if budget <= 0:
                break
            found = _retriever.on_api_event(api, thought, task["shown"])
            event = {"api_name": api, "api_class": [], "api_lessons": []}
            for kind in _index.get("api_classes", {}).get(api, []):
                if settings["class_once_per_task"] and kind in task["served_classes"]:
                    continue
                picked = [(e, s) for e, s in found["api_class"] if e in _index["api_class"].get(kind, [])][
                    :budget
                ]
                if picked:
                    blocks.append(
                        f"Notes from earlier tasks about APIs like {api}, which {CLASS_DESCRIPTIONS[kind]} "
                        "(use only if they fit what you are doing now):\n"
                        + "\n".join(bullet(e) for e, _ in picked)
                    )
                    budget -= len(picked)
                    task["shown"].update(e["lesson"] for e, _ in picked)
                    task["served_classes"].add(kind)
                    event["api_class"] += [
                        {"lesson": e["lesson"], "score": s, "kind": kind} for e, s in picked
                    ]
            picked = [(e, s) for e, s in found["api"] if e["lesson"] not in task["shown"]][: max(budget, 0)]
            if picked:
                blocks.append(
                    f"Notes from earlier tasks that used {api} (use only if they fit what you are doing now):\n"
                    + "\n".join(bullet(e) for e, _ in picked)
                )
                budget -= len(picked)
                task["shown"].update(e["lesson"] for e, _ in picked)
                event["api_lessons"] = [{"lesson": e["lesson"], "score": s} for e, s in picked]
            if event["api_class"] or event["api_lessons"]:
                task["log"]["events"].append(event)
    except Exception as exception:
        task["log"].setdefault("event_errors", []).append(repr(exception))
        blocks = []
    _write_log(thread_id)
    return "\n\n".join(blocks)


def thought_text(chat_messages: list | None) -> str:
    """The agent's text before the code in its latest message, as the API-layer rerank query."""
    for message in reversed(chat_messages or []):
        if getattr(message, "type", None) == "ai" or message.__class__.__name__ == "AIMessage":
            content = message.content if isinstance(message.content, str) else str(message.content)
            return CODE_FENCE.sub(" ", content).strip()[:1500]
    return ""
