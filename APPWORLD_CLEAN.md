# CUGA on AppWorld without test-derived prompt changes

This branch is CUGA as of `b219b481` (main, 2026-10-20) with the prompt changes that were made in response to
failures on AppWorld **test** tasks removed, so that AppWorld test scores measure the agent rather than fixes tuned on
the test set. It is the configuration behind the "clean" CUGA numbers below.

## What changed relative to `b219b481`

1. **#1003 cherry-picked** (`fix(reflection): … size reflection and VERIFY inputs from them`). A generic fix: before
   it, reflection and VERIFY kept only the first 30,000 characters of the agent history.
2. **Prompt changes traced to test-task failures reverted.** A change was reverted when its commit, pull request or
   linked issue names AppWorld test tasks or test-split runs, or when its text reproduces a test task:
   - reflection: the "Discovery vs. Selection" rules and Scope Discipline (#542, #545), the move/transfer and
     "+1 phone number" tips, the HTTP 409 error check, the multi-operation completeness check, and worked examples
     A-C (`51711b7d` → `2538f1c2`; the examples restate test_challenge tasks `9bf2c8a_1`, `dbc0276_1`, `07bb666_1`);
   - main agent: "Obstacle Handling & Recovery" (#704/#706), the default-planning text (#331), the todo "private
     scratchpad" note (#543);
   - plain final-answer prompt: the compose-and-send rule (#728/#729) and the yes/no and candidate-figure rules (#757).
3. **ACTION/QUERY pre-classifier off by default** (`appworld_classify_action_tasks = false`). Its prompt comes
   entirely from test failures (#544, #728).

The pre-execute **VERIFY gate must also be off.** Its prompt was designed around test_challenge `92fe421_1` and tuned
on "a 48-task dev subset of test_challenge" (#739). It is off in `settings.toml`, but cuga-eval's
`benchmarks/appworld/config/appworld.env` turns it on. The prompt file is unchanged, so running stock cuga-eval
silently brings it back.

## Running it with cuga-eval

Export these before `benchmarks/appworld/eval.sh`. cuga-eval loads its env files without overriding variables that
are already set:

```bash
export DYNACONF_ADVANCED_FEATURES__PRE_EXECUTE_VERIFY_ENABLED=false
export DYNACONF_ADVANCED_FEATURES__APPWORLD_CLASSIFY_ACTION_TASKS=false
```

Reflection stays on, as in cuga-eval's AppWorld configuration.

### AppWorld fixes (separate package)

`docs/appworld_clean/appworld-fixes.diff` patches the AppWorld package that cuga-eval clones
(`benchmarks/appworld/appworld`, upstream `42b5bcf3`). Apply it there with `git apply`.

- **Naive datetimes.** With sqlmodel 0.0.48, `amazon.place_order` fails with "Datetime values must have timezone
  information" after it has written the order row, and delivery-date checks compare UTC-shifted values. About a third
  of test_challenge cannot be passed without this fix.
- **Node-local base DBs.** Each node copies the read-only base databases to `/dev/shm` once. This only matters on a
  shared filesystem: on a GPFS cluster, concurrent reads of `amazon.db` slowed evaluation 100-fold.
  `APPWORLD_NO_LOCAL_BASE_DBS=1` turns it off.

## Results (GLM-5.3-Flash, self-hosted with vLLM, thinking on, no memory)

| set | clean CUGA (this branch) | AppWorld ReAct agent, same model |
|---|---|---|
| test_normal TGC / SGC | 87.8 (87.5-88.1) / 84.8, 2 runs | 91.5 (91.1-92.3) / 87.5, 3 runs |
| test_challenge TGC / SGC | 88.5 / 79.1, 1 run | 87.8 (85.9-89.7) / 74.8, 3 runs |
| 90-task `test_compare_full` TGC / SGC | 82.8 (82.2-83.3) / 71.7, 2 runs | 81.5 (78.9-85.6) / 67.8, 3 runs |

On the 90-task set, stock CUGA (blog: 81.1 / 66.7) and this branch score the same.

## Limits

- Only prompts were reverted. Code fixes made after test-run failures, such as auth-token lifetime, the sandbox
  clock freeze and schema types (`ad97388a`), are still in.
- The trace is conservative. About 81% of the AppWorld prompt lines have no recorded motivation, so test-derived
  text that was never documented cannot be ruled out.
- AppWorld plumbing stays: auto-login, the plain answer formatter, and cuga-eval's general instructions (sections A
  and B, copied from AppWorld's own ReAct agent).
- The branch is based on `b219b481`. Later main commits add more AppWorld-motivated prompt text (for example #750)
  and have not been audited.
