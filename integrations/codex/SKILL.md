---
name: paperbanana
description: "Generate academic method diagrams or data plots with the first author's PaperBanana implementation and Codex subscription models. Use when the user requests PaperBanana, PaperBanana 作者版, or dwzhu-pku/PaperBanana."
---

# PaperBanana author version through Codex

Installed command: `__PAPERBANANA_COMMAND__`.
Run its `doctor` subcommand to locate the checkout, inspect reference availability,
and verify ChatGPT subscription login. Integration details are in the checkout's
`integrations/codex/README.md`.

The adapter runs the author's actual agents, prompts, style guides, and reference
data. It replaces the model transport with local Codex app-server calls. Long
reference pools are retrieved in batches and the shortlists are reranked because
Codex limits a single input to 1 MiB. This is a fork integration, not an upstream
subscription feature, and does not reproduce the paper's original model results.

## Generate

Extract the relevant method section into a UTF-8 text/Markdown file. If the input
is a PDF, read it using available PDF tools first. Preserve the scientific content
and the user's required labels, arrow directions, and figure language. Derive a
caption from clear user intent; ask only when essential content is missing.

Use the current project/paper workspace for both input and output. Start with one
candidate and one critic round unless the user requests otherwise. Keep automatic
reference retrieval enabled.

```bash
__PAPERBANANA_COMMAND__ generate \
  --input /absolute/project/method.md \
  --caption "Overview of the proposed method" \
  --output /absolute/project/figures/method.png \
  --rounds 1
```

The default mode is `demo_full`. Optional controls include `--aspect-ratio 16:9`,
`--image-size 2k`, and `--candidates 2`. Use a fresh output filename: existing
images are not overwritten. Model and reasoning effort inherit the user's Codex
configuration; change `--model` or `--effort` only for an explicit model choice.

For a statistical plot, provide a JSON data file and add `--task plot`. The author
pipeline produces and executes Matplotlib code. Check displayed values against
the source data before presenting the final plot.

## Observe and verify

A full run can take several minutes. Use short execution yields and bounded
polls, and keep the user informed of meaningful stage changes. Each model call
has a 900-second timeout. A running process is not evidence of success.

The final PNG has a sibling `<stem>_paperbanana_<timestamp>/` run directory.
Check `manifest.json` for completion, actual model, subscription route, upstream
revision, and outputs. Inspect `candidate_*_result.json` for selected references,
planner/style descriptions, and critic feedback. `backend_calls.json` and
`calls/*/events.jsonl` record actual native image-generation events; native
originals are retained in `calls/*_image/native_output.png`.

Inspect the final image with `view_image`, check labels and topology, and link the
image. For additional changes, incorporate the user's concrete feedback into the
method/visual intent and generate a new named version. Do not invent a review
score or claim a pipeline run succeeded when only installation passed.

## Subscription boundary

```bash
__PAPERBANANA_COMMAND__ doctor
```

Require an existing ChatGPT login in `codex login status`. If login expires, ask
the user to complete `codex login`. The adapter never reads or copies auth tokens
and never falls back to metered API providers. If native image generation fails,
preserve intermediate work and report the concrete failure. An API fallback
requires the user's explicit choice.

The author's ordinary `app.py` and `skill/run.py` retain their API behavior. Use
this installed command to activate the subscription adapter. Model-generated
shell, browser, extra MCP, and subagent calls are disabled inside backend turns.
