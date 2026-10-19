# Codex subscription integration

This fork adds a local Codex adapter to [the first author's PaperBanana](https://github.com/dwzhu-pku/PaperBanana).
It runs the original Python agents and style guides with the existing ChatGPT
login in Codex. Text/vision stages use the selected Codex model; diagram rendering
uses Codex's native image-generation tool. No provider API key is required for
this integration.

这是作者版 PaperBanana 的 Codex 订阅适配。保留作者的检索、规划、风格、
绘图和 Critic 流程，通过本地 Codex 登录调用模型。

## Install (macOS / Linux)

Requirements: Python 3.12, `uv`, Git, and a recent Codex CLI with native image
generation and the app-server protocol. The integration was developed with
Codex CLI 0.163.0. Availability and usage limits depend on the signed-in account.

```bash
git clone https://github.com/YuChenSSR/PaperBanana-codex.git
cd PaperBanana-codex
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python scripts/prepare_bench.py --tasks diagram plot
codex login
.venv/bin/python integrations/codex/install.py
~/.local/bin/paperbanana-author doctor
```

If already logged in with ChatGPT, skip `codex login`. Installation writes a
launcher to `~/.local/bin/paperbanana-author` and a skill to
`${CODEX_HOME:-~/.codex}/skills/paperbanana-author`. It does not modify the global
Codex model, API credentials, or MCP configuration. Keep the checkout in place
because the launcher points to its virtual environment. The installer backs up
previous versions of its own launcher/skill and refuses unrelated files.

Open a new Codex chat and request:

> 用 $paperbanana-author 根据这份方法说明生成论文架构图，使用订阅模式。

## Command line

```bash
paperbanana-author generate \
  --input method.md \
  --caption "Overview of the proposed method" \
  --output figures/method.png
```

Defaults: `demo_full`, automatic reference retrieval, one candidate, one Critic
round, 16:9, requested 2K resolution. `--rounds 0` skips critique; `--rounds 2`
allows two rounds. Candidates run sequentially to limit simultaneous subscription
requests. More references and critique rounds increase runtime and plan usage.

Use `--task plot` with a JSON data input for the author's Matplotlib workflow.
For PDF input, extract the relevant method text first. The adapter does not parse
PDFs. Use `--help` for the full argument list.

`--model` and `--effort` override the otherwise inherited Codex settings. The
native image tool chooses its image backend; the selected Codex text-model name
is not used as an image-model API ID. Requested aspect ratio/resolution are sent
as image instructions, not guaranteed pixel dimensions. Native originals are
saved without artificial upscaling.

## What changes from upstream

- No edits to the author's agents, prompts, style guides, or provider modules.
  The wrapper replaces `call_model_with_retry_async` and
  `dispatch_image_generation` within its own Python process.
- The 200-reference diagram retrieval prompt exceeds Codex's 1 MiB input cap.
  All references in that original pool are evaluated in bounded batches, then
  shortlisted references are reranked with the author's retrieval prompt.
  No candidate descriptions are silently truncated. This batching can change
  reference selection relative to a single original-model call.
- Each model call uses an ephemeral app-server thread, the existing ChatGPT
  authentication, and restricted tools. Shell, browser, external MCPs, plugins,
  and subagents are disabled in those backend threads. Host skill discovery is
  disabled for stateless model calls; this uses a version-dependent Codex flag.
- Native images must come from a successful `imageGeneration` event. A path
  mentioned in model prose does not count as an image-generation result.
- Malformed Critic JSON is treated as a failed stage, not as automatic approval.
- Output names are protected from overwrite. The wrapper saves prompts,
  responses, selected references, intermediate images, and provenance.

This is an independent integration in a fork, not a Google/OpenAI endorsement.
Changing the model and retrieval batching means these are not paper-reproduction
results. Running upstream `app.py` or `skill/run.py` directly still uses their
normal API providers; the subscription entry is `paperbanana-author`.

## Artifacts and diagnostics

Beside the final PNG, each run creates `<stem>_paperbanana_<timestamp>/` with:

- `manifest.json`: status, repository revision, model, route, inputs, outputs.
- `candidate_*_result.json`: author-agent descriptions, references, and critique.
- `backend_calls.json`: timing, model usage, native image-event counts.
- `calls/`: actual prompts, responses, native images, and sanitized event logs.

Logs contain the supplied paper text and model outputs. Keep private runs in a
private project directory. The adapter never reads/copies `auth.json`, stores
tokens, or silently falls back to metered APIs. If login or image generation
fails, it stops and preserves diagnostics. Run `codex login` to restore an
expired subscription session.

## Validation

```bash
.venv/bin/python -m unittest discover -s integrations/codex -p 'test_*.py' -v
.venv/bin/python integrations/codex/paperbanana_codex.py doctor
```

Unit tests check image provenance, reference batching, interleaved image inputs,
credential routing, and malformed Critic results. `doctor` verifies local
installation only. A completed live figure run is needed to establish actual
account/model/native-image access. App-server interfaces can change; check a
small live example after upgrading Codex or the upstream repository.

Validation on 2026-10-19: seven unit tests passed, and a complete diagram smoke
test ran with Codex CLI 0.163.0, ChatGPT login, and `gpt-6-astra` / `xhigh`. The
test selected 10 references from the author's 200-reference retrieval pool and
completed 8 text/vision calls and 3 native image-generation calls, including one
Critic-driven revision. The final PNG was visually checked for its specified
three-stage topology and labels. This is an integration check, not a figure-quality
benchmark. The statistical-plot path has not received the same live validation.

## Attribution

PaperBanana research and original implementation belong to the authors named in
the repository README. Original files retain their license and attribution.
The new files in this directory are distributed under the repository's Apache
2.0 license.
