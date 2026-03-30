# Overview

This repository exists for optimizing buzzdetect code in a hands-off agentic loop. Your goal is to minimize the overall time it takes eval.py to run without affecting the results.

# Previous tests

The folder "eval_out" holds all previous tests. Each directory name is the name of a previous test. Contents of each test are as follows:

## profile.csv

Profiling data for the winning settings combo. Key phases:

* **overall**: total analysis time — this is the value to minimize
* **inference**: time the embedder and model spend converting audio to class predictions

## settings.json

Gives the settings (arguments) used to run the eval. These are considered differently than optimizations; settings can be tuned within the same optimization.

## description.md

A brief description of what the changes were and why this approach was used. Keep this short and leave the details of the change to the commit.

## next_steps.md

At the end of each iteration, write a brief note for the next agent context. What did you learn? What do you think is worth trying next? What dead ends should be avoided? This file is your main channel of communication with future iterations.

## tuning/

Contains one subdirectory per settings combination tried during the auto-tune sweep. Each subdir has its own `settings.json`, `profile.csv`, and `[timestamp].log`. The best combo's `settings.json` and `profile.csv` are promoted to the parent directory.

Log files can be over 1,000 lines long; do not read them fully into context. The first ~100 lines may reveal buffer bottlenecks; later lines give granular per-worker timing.

## scratch/

Feel free to create a scratchwork directory if you need to conduct some troubleshooting or in-depth analyses.

## tuning_results.json

All combos ranked by `overall_s`, with the winner flagged as `best_combo`.

## files/

Found inside each tuning combo subdir (`tuning/<combo>/files/`), not at the test root. Contains the analysis output for that combo. Deleted automatically after a successful results check. Remains on disk only if a results mismatch is detected (for forensic inspection).

## Comparing tests

To compare tests and see how they rank against each other, run:

```
.venv/bin/python -c "from eval import compare_tests; from pathlib import Path; compare_tests(Path('eval_out'))"
```

This prints tests ranked by overall time, relative to the best performer. By default, only the top 3 results are shown. With many tests, pass `n=<large_number>` to see more:

```
.venv/bin/python -c "from eval import compare_tests; from pathlib import Path; compare_tests(Path('eval_out'), n=50)"
```

To compare the autotune combos within a single run:

```
.venv/bin/python -c "from eval import compare_tests; from pathlib import Path; compare_tests(Path('eval_out/<test_name>/tuning'))"
```

# Context management

Be aggressive about managing your context window. Do not cat large files. Use head, tail, and grep. Do not include eval output in your context. When reading previous tests, read the description.md — not the logs unless you have a specific reason to check something in them. Use `compare_tests()` to get performance numbers rather than reading profile.csv files manually.

# Settings

Settings means parameters like chunk length and number of concurrent audio streamers. eval.py automatically sweeps a grid of these combinations on every run and promotes the best result — you do not tune them manually.

Settings tuning alone is not the primary interest. Focus on code-level changes; the auto-tune handles finding the best settings for whatever code you've written.

# Workflow

## 01. Get situated

Read analyze.py, examine the git history, and explore previous tests. Read the next_steps.md files from recent tests to see what prior iterations recommended.

## 02. Check what's been tried

Run `ls eval_out/` and review the description.md of any test that sounds related to what you're considering. Do this *before* writing any code. Re-attempting a previous failure without a meaningfully different approach is wasted compute.

## 03. Come up with an optimization

**Follow the most recent `next_steps.md` recommendation.** Read the `next_steps.md` from the most recent test and implement what it recommends. This is the primary directive — the previous agent left that note specifically to guide you.

If the most recent `next_steps.md` has no clear actionable recommendation, or if that approach has already been tried, then use your judgment: build off previous SUCCESSes, try to correct a FAILURE that had a good idea behind it, or go in a new direction if things seem stuck.

You can also take some steps in advance of the optimization like profiling GPU usage or running a quick profile on the short audio file in audio_in. Take your time before deciding on an optimization.

## 04. Name optimization

Give a name to your optimization. It should be short, informative, unique to the prior tests, and appropriate for a directory name (e.g. use underscores over spaces).
Prefix the test name with the next number in the testing sequence, e.g. "09_cast_samples". This is a naming convention (not enforced by code) that keeps the eval_out directory ordered.

## 05. Apply optimization

Modify the codebase as you see fit for optimization.

### ***Allowed***

* Generating derivative models, e.g. ONNX models
* Changing settings (see the Settings section above)
* Installing new dependencies

### ***Prohibited***

Do not make any changes that would affect the results, e.g. changing frame hop. If you do, the test will automatically fail as BADRESULTS.

You may not change eval.py in any way.

You may not change the input audio files in any way.

## 06. Run evaluation

Run eval.py with your test name and model name:

```
.venv/bin/python eval.py --test-name <your_test_name> --model <model_name>
```

Available models are directories under `models/`. eval.py automatically sweeps 6 settings combinations and promotes the best to `eval_out/<your_test_name>/`. The output is very verbose; do not include it in your context.

The sweep takes ~5 minutes. Set a timeout for 10 minutes. If the tests aren't finished by then, check any existing results and make a manual verdict (likely SLOWER).

## 07. Verdict

After eval.py finishes, it prints a rankings table and automatically prints the verdict:

```
Final verdict: FASTER/SLOWER/NEUTRAL  (+X.X% vs <best_test>)
```

Verdicts:
- **FASTER**: your test is >5% faster than the current best
- **SLOWER**: your test is >5% slower than the current best
- **NEUTRAL**: within ±5%
- **BADRESULTS**: results check failed. This is NOT printed in the "Final verdict:" line — instead `=== RESULTS MISMATCH ===` is printed during the run for each failing combo, and "No eligible results to promote" is printed at the end if all combos failed. In that case there is no Final verdict line at all.

The rankings table shows per-phase % differences vs. the best performer across columns (`overall`, `inference`, `read`, `resample`, `format`, `write`). Note that very quick phases have noisy % changes — the only metric that determines the verdict is `overall`.

You can re-run the comparison at any time:

```
.venv/bin/python -c "from eval import compare_tests; from pathlib import Path; compare_tests(Path('eval_out'), '<your_test_name>')"
```

## 08. Commit

**No matter the verdict**, commit. The commit message should start with the verdict (FASTER/SLOWER/NEUTRAL/BADRESULTS), then give a brief breakdown of the results.

If the verdict was BADRESULTS or SLOWER, revert the commit before finishing.

### Iterative tests that build on each other

Sometimes an optimization requires several attempts to get right — each attempt fixes a bug revealed by the previous run, all converging on a single final result. In this case, commit everything together under the **final verdict** rather than committing and reverting each intermediate step. The intermediate eval_out directories serve as documentation of the iterative process. Only revert if the final verdict is SLOWER or BADRESULTS.

## 09. Leave notes

Write next_steps.md (and description.md) in your test's eval_out directory. What did you learn during this iteration? What would you try next? What looked promising but didn't pan out? Be specific — the next iteration starts with a fresh context and this is its best starting point.

**Important:** Write and commit these notes *after* any reversion, not before. The revert will delete them if they were part of the reverted commit. Commit the notes as a separate commit so they survive regardless of verdict.


REMEMBER: commit your changes whether they were good or bad. If they were good, add your notes with your commit. If they were bad, add your notes after the reversion and commit again. We need to track bad steps, too!
