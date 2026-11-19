# Layered lesson retrieval (lessons_v2) for CugaLite on AppWorld

This branch adds lessons_v2 to CugaLite: memory that shows lessons from earlier train and dev tasks at the point
where they apply. It is a port of the lesson retrieval written for the AppWorld ReAct code agent, which is on the
[`lessons-v2-retriever` branch of the AppWorld fork](https://github.com/jayaramkr/appworld/tree/lessons-v2-retriever).
That branch has the index-building pipeline and a full design write-up (`lesson_retrieval/DESIGN.md`).

The branch builds on `appworld-clean` (see [APPWORLD_CLEAN.md](APPWORLD_CLEAN.md)). With no index set, it behaves
exactly like `appworld-clean`.

## What the agent sees

| Layer | Lessons | When | Where in CugaLite |
|---|---|---|---|
| 1. answer | the top answer lessons by support | every task | added to the special instructions, as section E |
| 4. start | retrieved once from the task instruction | start of a task | added after them, as section F |
| 2. API class | one lesson for paginated-list APIs, one for write APIs | the first block that uses such an API | added to the execution output |
| 3. API / app | lessons tagged with that API or its app, reranked against the agent's text before the code | each new API | added to the execution output |

- **How API use is detected:** a code block counts as using an API if it calls the CugaLite tool for it, or if it
  calls `find_tools` and the output lists it.
- **Limits:** at most 4 notes per block, and no lesson twice in a task.
- **Failures:** a retrieval error is logged and the task continues without lessons. If the index fails to load,
  retrieval is turned off and the agent runs normally.

## Code

| File | What it does |
|---|---|
| `src/cuga/backend/cuga_graph/nodes/cuga_lite/lesson_retriever.py` | the retriever, identical to the ReAct agent's |
| `src/cuga/backend/cuga_graph/nodes/cuga_lite/lessons.py` | the CugaLite side: loading, the task-start sections, the per-block notes and the log |
| `src/cuga/backend/cuga_graph/nodes/cuga_lite/adapter/prepare_node.py` | adds sections E and F on a fresh conversation |
| `src/cuga/backend/cuga_graph/nodes/cuga_lite/adapter/sandbox_node.py` | adds the API notes after each executed block, unless the step is a function-calling plan |
| `docs/appworld_lessons/cuga_to_react_layout.py` | rearranges CUGA's train and dev runs into the ReAct output layout, for mining |
| `docs/appworld_lessons/make_tool_map.py` | writes the map from CugaLite tool names to AppWorld APIs |

The retriever is built and warmed when `lessons.py` is first imported, before any task runs. AppWorld runs tasks
with time frozen, and loading the models under the frozen clock caused runaway memory use.

## Turning it on

Retrieval is off unless `CUGA_LESSON_INDEX` is set. The settings are environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `CUGA_LESSON_INDEX` | unset (off) | path to an index from `lesson_retrieval/build_lesson_index.py` |
| `CUGA_LESSON_TOOL_MAP` | required when on | JSON map from CugaLite tool names to `app.api` (`make_tool_map.py`) |
| `CUGA_LESSON_EMBEDDER` | `BAAI/bge-base-en-v1.5` | dense embedder for start lessons |
| `CUGA_LESSON_RERANKER` | `cross-encoder/ms-marco-MiniLM-L-12-v2` | cross-encoder for start and API lessons |
| `CUGA_LESSON_DEVICE` | `cpu` | device for both models |
| `CUGA_LESSON_MAX_ANSWER` | 2 | answer lessons per task |
| `CUGA_LESSON_START_K` / `CUGA_LESSON_START_THRESHOLD` | 5 / −10.5 | start lessons kept, and the reranker score cutoff |
| `CUGA_LESSON_API_K` / `CUGA_LESSON_API_THRESHOLD` | 3 / 2.0 | API lessons per API, and the cutoff |
| `CUGA_LESSON_CLASS_K` | 1 | lessons per API class |
| `CUGA_LESSON_CLASS_ONCE_PER_TASK` | 1 | serve each API class once per task |
| `CUGA_LESSON_MAX_PER_STEP` | 4 | notes per executed block |

The thresholds are on the default reranker's score scale. If you change the reranker, re-tune them on train or dev
tasks.

Extra dependencies: `sentence-transformers` and `rank_bm25`, installed into the environment that runs CUGA.

```bash
python docs/appworld_lessons/make_tool_map.py <appworld>/data/api_docs/standard tool_to_api.json
export CUGA_LESSON_INDEX=/path/to/lesson_index.json CUGA_LESSON_TOOL_MAP=/path/to/tool_to_api.json
# then run the AppWorld evaluation as usual
```

Each task's log is written to `$CUGA_LOGGING_DIR/lesson_retrieval/<thread>.json`. It lists the instruction, the
answer and start lessons with their scores, and every API event with the lessons it showed.

## Building an index from CUGA's own runs

We mine CUGA's lessons from CUGA's own no-memory runs on the AppWorld **train** and **dev** sets, never from test
tasks. The mining pipeline is the ReAct one, unchanged. The layout script makes CUGA's runs look like ReAct runs.

1. Run CugaLite with no memory on `train` and `dev`, through the AppWorld evaluation in cuga-eval.
2. Rearrange the runs into the ReAct layout:

   ```bash
   python docs/appworld_lessons/cuga_to_react_layout.py --run-dir <run dir with cuga_logging/> \
       --bundles <cuga-eval>/benchmarks/appworld/evaluation_bundles --src <run name> \
       --data-dir <appworld>/data --out runs_layout
   ```

   - **Which results it uses:** each task's scored result, from `<run name>_s*/results/partial/*.json` in the
     bundles, and its latest trajectory.
   - **What it writes:** `runs_layout/{train,dev}/`, with a `logger.jsonl`, an `environment_io.md` (one
     `apis.<app>.<api>(...)` line per tool call) and an evaluation report per task, plus `evaluations/<split>.json`.

3. In the AppWorld fork's `lessons-v2-retriever` branch, run its `lesson_retrieval/` pipeline on `runs_layout` (see
   its README):
   1. `generate_universal_guidelines.py runs_layout/{train,dev} --phase 1`: extract raw lessons per task.
   2. `retag_lessons.py --runs-dir runs_layout`: tag each lesson's scope, APIs and trigger.
   3. `build_lesson_index.py`: bucket the tagged lessons and merge duplicates without loss.
   4. Optionally, `verify_lessons.py`: check entries against the API docs. Our CUGA runs used an unchecked index.

## Results: GLM-5.3-Flash, self-hosted

These runs used GLM-5.3-Flash on vLLM (thinking on), with the CugaLite settings in `APPWORLD_CLEAN.md`. The index
was mined by GLM-5.3-Flash from CugaLite's own train and dev runs. 303 raw lessons became 221 entries: 4 answer,
53 API class, 65 API, 4 app and 95 start.

| Set | Runs per arm | No memory TGC / SGC | lessons_v2 TGC / SGC | Tokens per task (no memory → lessons_v2) |
|---|---|---|---|---|
| test_normal (168 tasks) | 2 | 87.8 / 84.8 | 90.8 / 86.6 | 219K → 224K |
| test_challenge (417 tasks) | 1 no memory, 2 lessons_v2 | 88.5 / 79.1 | 87.5 / 78.4 | 341K → 379K |
| a fixed random subset of 30 scenarios (90 tasks) | 2 | 82.8 / 71.7 | 85.6 / 78.3 | 345K → 348K |

How to read the results:
- **test_normal:** +3.0 TGC for about 2% more tokens. Both lessons_v2 runs scored above both no-memory runs.
- **test_challenge:** no gain.
- **The 90-task subset:** the TGC gain is within the run-to-run noise for 90 tasks. SGC rose by 6.6.
- **Noise:** with 1–2 runs per arm, treat differences under about 5 tasks as noise.

The prompt version: these runs were made just before the last `appworld-clean` commit (`6738bd28`). That commit
removed one test-derived answer from the answer prompts. No-memory runs on `6738bd28` were within run-to-run noise
of the earlier ones.
