"""Lay out CUGA's AppWorld runs the way the ReAct runner writes them, so the lessons_v2 mining scripts
(generate_universal_guidelines.py phase 1, lesson_retrieval/retag_lessons.py, build_lesson_index.py) run unchanged.

For each task it uses the scored attempt (the latest trajectory; a task is only re-run after the attempt it replaces
failed) and writes, under <out>/<split>/:
  tasks/<task_id>/logs/logger.jsonl       task instruction, then agent / environment turns
                                           (agent turn = '\"\"\"<reasoning>\"\"\"' + the raw response, as for thinking models)
  tasks/<task_id>/logs/environment_io.md   one ```python``` block per call, `apis.<app>.<api>(`, from the scored tool calls,
                                           so retag_lessons.py can list the APIs the task used
  tasks/<task_id>/evaluation/report.md     passed / failed requirements
  evaluations/<split>.json                 {"individual": {task_id: {success, difficulty, num_tests, failures}}}

CUGA names an AppWorld API `<app>_<api_name>_<path without the app prefix>_<method>`; the API docs give the inverse.

Usage: python cuga_to_react_layout.py --run-dir RUN --bundles BUNDLES --src EXP --data-dir APPWORLD_DATA --out OUT
"""

import argparse
import glob
import json
import os
import re

# CUGA's own supervisor helpers, which are not AppWorld APIs by that name.
EXTRA_TOOL_TO_API = {
    "supervisor_payment_cards": "supervisor.show_payment_cards",
    "supervisor_addresses": "supervisor.show_addresses",
}


def tool_to_api(api_docs_dir: str) -> dict[str, str]:
    mapping = dict(EXTRA_TOOL_TO_API)
    for path in glob.glob(f"{api_docs_dir}/*.json"):
        app = os.path.basename(path)[:-5]
        if app == "api_docs":
            continue
        for api_name, spec in json.load(open(path)).items():
            slug = re.sub(r"[^a-z0-9]+", "_", re.sub(rf"^/{app}(?=/|$)", "", spec["path"]).lower()).strip("_")
            mapping[f"{app}_{api_name}_{slug}_{spec['method'].lower()}"] = f"{app}.{api_name}"
    return mapping


def scored_results(bundles: str, src: str) -> dict[str, dict]:
    task_id_to_result = {}
    for path in glob.glob(f"{bundles}/{src}_s*/results/partial/*.json"):
        blob = json.load(open(path))
        result = blob["results"][0] if isinstance(blob, dict) and "results" in blob else blob
        if result.get("error") is None:
            task_id_to_result[result["task_name"]] = result
    return task_id_to_result


def latest_trajectories(run_dir: str) -> dict[str, str]:
    task_id_to_path: dict[str, str] = {}
    for path in glob.glob(f"{run_dir}/cuga_logging/trajectory_data/*/*.json"):
        task_id = os.path.basename(path)[:-5]
        if task_id not in task_id_to_path or os.path.getmtime(path) > os.path.getmtime(
            task_id_to_path[task_id]
        ):
            task_id_to_path[task_id] = path
    return task_id_to_path


def turns(trajectory: dict) -> list[dict]:
    """Agent / environment turns in order. A turn starts at Raw_Assistant_Response; its reasoning follows it."""
    records: list[dict] = []
    agent: dict | None = None
    for step in trajectory.get("steps", []):
        name, data = step.get("name"), str(step.get("data") or "")
        if name == "Raw_Assistant_Response":
            agent = {"role": "agent", "response": data, "reasoning": ""}
            records.append(agent)
        elif name == "Assistant_reasoning" and agent is not None:
            agent["reasoning"] = data
        elif name == "User_output":
            records.append({"role": "environment", "content": data})
    out = []
    for step_number, record in enumerate(records, 1):
        if record["role"] == "agent":
            reasoning = record["reasoning"].strip()
            content = (f'"""{reasoning}"""\n' if reasoning else "") + record["response"]
            out.append({"role": "agent", "step_number": step_number, "content": content})
        else:
            out.append({"role": "environment", "step_number": step_number, "content": record["content"]})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--bundles", required=True)
    parser.add_argument("--src", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    split_of = {
        t: s for s in ("train", "dev") for t in open(f"{args.data_dir}/datasets/{s}.txt").read().split()
    }
    names = tool_to_api(f"{args.data_dir}/api_docs/standard")
    results = scored_results(args.bundles, args.src)
    trajectories = latest_trajectories(args.run_dir)
    individual: dict[str, dict] = {"train": {}, "dev": {}}
    unmapped: set[str] = set()
    problems = []
    for task_id, split in sorted(split_of.items()):
        if task_id not in results or task_id not in trajectories:
            problems.append(f"{task_id}: {'no scored result' if task_id not in results else 'no trajectory'}")
            continue
        result, trajectory = results[task_id], json.load(open(trajectories[task_id]))
        task_dir = f"{args.out}/{split}/tasks/{task_id}"
        os.makedirs(f"{task_dir}/logs", exist_ok=True)
        os.makedirs(f"{task_dir}/evaluation", exist_ok=True)
        with open(f"{task_dir}/logs/logger.jsonl", "w") as handle:
            handle.write(json.dumps({"role": "task", "content": trajectory.get("intent", "")}) + "\n")
            for turn in turns(trajectory):
                handle.write(json.dumps(turn) + "\n")
        calls = []
        for call in result.get("tool_calls") or []:
            api = names.get(call["name"])
            if api is None:
                unmapped.add(call["name"])
                continue
            calls.append(f"```python\napis.{api}(...)\n```\n")
        with open(f"{task_dir}/logs/environment_io.md", "w") as handle:
            handle.write("\n".join(calls))
        evaluation = result.get("appworld_evaluation") or {}
        failures = [{"requirement": f.get("requirement", "")} for f in evaluation.get("failures") or []]
        passes = evaluation.get("passes") or []
        success = bool(result.get("success"))
        with open(f"{task_dir}/evaluation/report.md", "w") as handle:
            handle.write(f"Success: {success}\n\nPassed requirements:\n")
            handle.writelines(f"- {p.get('requirement', '')}\n" for p in passes)
            handle.write("\nFailed requirements:\n")
            handle.writelines(f"- {f['requirement']}\n" for f in failures)
        individual[split][task_id] = {
            "success": success,
            "difficulty": result.get("difficulty"),
            "num_tests": len(passes) + len(failures),
            "failures": failures,
        }
    for split, tasks in individual.items():
        os.makedirs(f"{args.out}/{split}/evaluations", exist_ok=True)
        json.dump({"individual": tasks}, open(f"{args.out}/{split}/evaluations/{split}.json", "w"))
    counts = {s: (len(t), sum(v["success"] for v in t.values())) for s, t in individual.items()}
    print(
        f"tasks (written, passed): {counts}; unmapped tool names: {sorted(unmapped)}; problems: {len(problems)}"
    )
    for problem in problems:
        print("  PROBLEM", problem)


if __name__ == "__main__":
    main()
