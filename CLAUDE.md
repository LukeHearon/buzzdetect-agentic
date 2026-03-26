# Overview

This repository exists for optimizing buzzdetect code in a hands-off agentic loop. Your goal is to minimize the overall time it takes eval.py to run without affecting the results.

# Previous tests

The folder "eval_out" holds all previous tests. Each directory name is the name of a previous test. Contents of each test (except the baseline) are as follows:

## Verdict.json

FASTER means a >5% speedup in overall analysis or a previous agent decided that a smaller speedup was nonetheless significant.

NEUTRAL means no meaningful difference

SLOWER means the test made the analysis slower by >5%

BADRESULTS means the code changes caused the results to differ from the baseline, which is unacceptable

The FASTER/SLOWER/NEUTRAL verdict is made automatically based on a 5% threshold of the *overall* evaluation runtime rather than individual steps. Real-world analyses are often on thousands of hours of audio at once, so the audio analysis speed (which is the bulk of this eval) is most important. Improving, e.g., the file cleaning time may be valuable, but is not the direct target of the current optimization task. See the 07. Verdict step for exceptions.

## comparison.json

Compares the runtime of each processing step to the baseline.

NOTE: the value we want to decrease is "overall"

Profiling is done by phases.

* Overall: the total time to run the eval analysis. This is what we're trying to optimize
* Inference: the time it takes the embedder and model to convert the audio input to class predictions

## settings.json

Gives the settings (arguments) used to run the eval. These are considered differently than optimizations; settings can be tuned within the same optimization.

## description.md

A brief description of what the changes were and why this approach was used. Keep this short and leave the details of the change to the commit.

## next_steps.md

At the end of each iteration, write a brief note for the next agent context. What did you learn? What do you think is worth trying next? What dead ends should be avoided? This file is your main channel of communication with future iterations.

## [timestamp].log

Contains print statements from every step of the analysis. This file will be over 1,000 lines long; do not read it fully into context. You may be able to spot buffer bottlenecks reported within the first 100 lines or so. Could also give more granular information about the times reported by different workers.

## scratch/

Feel free to create a scratchwork directory if you need to conduct some troubleshooting, in-depth analyses, or write a python script to test a variety of settings combinations.

## files/

This folder only exists temporarily, until the results of the analysis can be checked against the baseline to make sure there are no numerical discrepancies.

# Context management

Be aggressive about managing your context window. Do not cat large files. Use head, tail, and grep. Do not include eval output in your context. When reading previous tests, read the description.md and comparison.json — not the logs unless you have a specific reason to check something in them.

# Settings

Settings means the arguments for eval.py, e.g. chunk length and number of concurrent audio streamers. You are free to change settings alongside code optimizations — sometimes a code change shifts the optimal settings. For example, if you change the audio streaming logic, larger chunks may become more efficient.

However, settings tuning alone is not the primary interest. Seek code-level changes first, and tune settings to fit. If you think settings need adjustment after a code change (e.g., the analyzer is outpacing the streamer), you may backup the first run's results to a scratch subdirectory, add a note, and re-run with different settings.

The default settings in eval.py are the current fastest settings.

# Workflow

## 01. Get situated

Read analyze.py, examine the git history, and explore previous tests. Read the next_steps.md files from recent tests to see what prior iterations recommended.

## 02. Check what's been tried

Run `ls eval_out/` and review the description.md of any test that sounds related to what you're considering. Do this *before* writing any code. Re-attempting a previous failure without a meaningfully different approach is wasted compute.

## 03. Come up with an optimization

At the start of each iteration, you will be at the current best version of the code. If the last test was better than the one before it, it becomes the new starting point. Generally, we want to build in this direction. However, if you feel that we've become stuck in a local optimum, feel free to try something radically different. Checkout a previous commit and make changes from there. At worst, we'll just revert to the previous optimum.

You can use any previous results as inspiration, or go in a completely different direction. You might want to build off of previous SUCCESSes, try to correct a FAILURE that had a good idea behind it, tune the settings on a NEUTRAL, combine multiple prior SUCCESSes, etc. Make sure we keep exploring, though, so if a lot of the tests are just iterating off of each other, come up with a brand new direction.

You can also take some steps in advance of the optimization like profiling GPU usage or running a quick profile on the short audio file in audio_in. Take your time before deciding on an optimization.

## 04. Name optimization

Give a name to your optimization. It should be short, informative, unique to the prior tests, and appropriate for a directory name (e.g. use underscores over spaces)

## 05. Apply optimization

Modify the codebase as you see fit for optimization.

### ***Allowed***

* Generating derivative models, e.g. ONXX models
* Changing settings (see the Settings section above)
* Installing new dependencies

### ***Prohibited***

Do not make any changes that would affect the results, e.g. changing frame hop. If you do, the test will automatically fail as BADRESULTS.

You may not change eval.py in any way.

You may not change the input audio files in any way.

## 06. Run evaluation

Run `eval.py -help` to see options for settings.

Run eval.py (the virtual environment with dependencies is in ./.venv), giving your test name as an argument and changing whatever arguments you see fit. The output is very verbose; do not include it in your context.

eval.py will automatically output the test results to eval_out/[your test name]

The script takes ~70s to run at baseline. Set a timeout for 10 minutes.

## 07. Verdict

The verdict of whether the optimization is FASTER or SLOWER is based on a 5% threshold of the overall analysis time. If you have made a change that you are **certain** improves the speed of one of the steps (e.g., writing files is 80% faster), **you may manually change verdict.txt to FASTER**. You may only do this if you are confident that the improvement is a real one. Note that steps that are very quick will consequently have very noisy % changes. +430% might just be noise if the step previously took 0.0002s. Before manually changing the verdict, compare against previous test results or the available baselines.

## 08. Commit

**No matter the verdict**, commit. The commit message should start with the verdict, then give a brief breakdown of the results.

If the verdict was BADRESULTS or SLOWER, revert the commit before finishing.

## 09. Leave notes

Write next_steps.md in your test's eval_out directory. What did you learn during this iteration? What would you try next? What looked promising but didn't pan out? Be specific — the next iteration starts with a fresh context and this is its best starting point.

After your commit (and potential reversion), you have finished your iteration. The next iteration will start with a fresh context. Thank you!
