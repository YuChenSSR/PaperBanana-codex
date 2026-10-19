# PaperBanana-Codex: repository workflow

This source describes the actual implementation of the PaperBanana-Codex fork.
Create one complete workflow illustration for the repository README, with concise
English labels and a clean, readable landscape layout. The diagram explains the
default `demo_full` subscription route and identifies the retained upstream API
route as a separate optional path. It is a software workflow, not a claim about
experimental accuracy or performance.

## Preparation and user entry

One-time setup installs Python dependencies, downloads PaperBananaBench reference
data, signs into Codex with ChatGPT, and installs the `paperbanana` skill plus the
`paperbanana-author` command. Summarize these as a small preparation ribbon labeled
"Setup: environment + references + ChatGPT login" rather than four large nodes.

The user supplies either method text/Markdown plus a figure caption (methodology
diagram task), or JSON data plus visual intent (statistical plot task). PDF method
text is extracted by Codex before this pipeline; the adapter does not parse PDFs.
Show this as a short secondary note, not a new PDF parsing module.

Users invoke either the `$paperbanana` skill in Codex or the
`paperbanana-author generate` CLI. Both reach the same local wrapper. Use the short
labels "Codex skill: $paperbanana" and "CLI: paperbanana-author" in one entry box.

The wrapper checks input/output paths, reference availability and the current
ChatGPT login, creates a run directory, and substitutes the two model-call
functions inside its Python process. It instantiates the original author agents.
Use one box labeled "Subscription wrapper" with the subtitle
"Preflight + run context". The original core files are not modified.

## Main author pipeline: preserve this order and topology

The default flow is strictly:

Inputs → skill / CLI → subscription wrapper → Retriever → Planner → Stylist
→ Visualizer → Critic → final outputs.

The five author stages should be the main visual focus, in a shared region labeled
"Author pipeline". The preparation/user-entry elements may occupy a compact upper
row, and runtime details may occupy a separate lower band, so the main stages do
not become tiny.

1. **Retriever** reads PaperBananaBench metadata. The author diagram pool uses the
   first 200 references; the dataset contains 298 diagram references. To stay
   below Codex's single-input character cap, the adapter evaluates the original
   pool in bounded batches, combines shortlists, and reranks them. The result is
   10 selected reference IDs. Label the node "Retriever" / "Batch → shortlist →
   rerank" and its outgoing edge "10 references". Show a separate small database
   labeled "PaperBananaBench" feeding the Retriever.
2. **Planner** receives the user's content and intent plus the selected reference
   text and images. It produces a detailed figure description. Label it
   "Planner" / "Content + layout".
3. **Stylist** refines the description using the author's NeurIPS style guide.
   It preserves the scientific semantics. Label it "Stylist" / "Style guide".
   A small reference card "Style guide" can feed this stage if space permits.
4. **Visualizer** renders the description. Inside or immediately below this node,
   show two alternative task-specific routes:
   - **Diagram → Native image generation** through Codex.
   - **Plot → Matplotlib code → Local execution**. Codex generates the plotting
     code; the original plotting worker executes it. The plot path does not call
     the native image-generation tool.
   Both routes return a raster figure to the same Critic. These are alternatives,
   not sequential steps. Keep the labels short and readable.
5. **Critic** inspects the actual image together with the original content and the
   detailed description. It returns critique and a revised description as JSON.
   Show a clear feedback arrow from Critic back to Visualizer labeled
   "Revise description". Show that this is a bounded loop, with a small label
   "Up to N rounds". Do not loop back to Retriever, Planner or Stylist.

The process stops early when the Critic says no changes are needed, or finishes
when the configured round limit is reached. The latest successful image is
exported. After the last allowed regeneration, the author implementation does not
perform an additional critique, so do not claim that the final output has passed
an extra quality gate. An output edge labeled "Accept / round limit" is sufficient.

The full implementation also renders a planner-stage candidate before the styled
candidate. That baseline can be summarized by an optional short note "Drafts +
revisions" under Visualizer; it need not become a second parallel full pipeline.
Additional user-requested candidates run sequentially.

## Shared subscription runtime: supporting layer, not another pipeline stage

Below the main flow, show a clearly distinct band labeled "Codex subscription
runtime". The backend connects local `codex app-server` via JSON-RPC and reuses the
existing ChatGPT login. It inherits the user's selected model and reasoning effort.
Each model call runs in an ephemeral thread.

Use two compact runtime services in this band:
- "Text / vision model" supports Retriever, Planner, Stylist, Critic, and plotting
  code generation.
- "Native image tool" supports only diagram rendering in Visualizer.

Use dashed support connectors or a shared support bracket so these services are
visually distinct from the solid main data-flow arrows. Do not imply that only
the Planner uses the text/vision model, or that all stages generate pictures.

A small note can say "ChatGPT login • selected model + effort • ephemeral calls".
No API key is required for this subscription route. The adapter does not copy
tokens or silently switch to metered provider APIs. These are properties of the
runtime and should not be drawn as additional processing steps.

## Outputs and failure behavior

The output region includes three concise items:
- "Final PNG"
- "Drafts + critique"
- "Manifest + call logs"

The manifest records repository revision, adapter hashes, model, route and run
status. Call records retain prompts, responses and native image-event evidence.
Only successful native imageGeneration events are accepted as diagram images.
Existing output files are not overwritten. On failure, diagnostics are preserved.
Summarize failure handling in a small note "Failure → preserve diagnostics";
avoid many red error arrows that would obscure the main successful workflow.

## Retained upstream route

The upstream `app.py`, `demo.py`, `main.py` and `skill/run.py` remain available and
use their normal configured provider API credentials. They do not activate this
subscription wrapper. Include a small visually separated gray callout:
"Original UI / scripts → Provider APIs (separate route)".
Do not connect this optional route to the subscription runtime, and do not imply
the original web UI automatically uses ChatGPT subscription login.

## Visual requirements

- Landscape 16:9, requested 4K class, readable when embedded in a GitHub README.
- Use a compact, balanced composition with modest margins. Avoid large empty
  space and avoid a very long single row of tiny nodes.
- Main stage labels should be visually dominant; concise subtitles only. Do not
  reproduce paragraphs, filenames of log files, exact pixel dimensions, the model
  identifier, or reference counts other than the "10 references" edge in the image.
- White background, flat fills, restrained blue/teal for author stages and amber
  or violet for the Codex runtime, with a neutral gray separate-route callout.
- Consistent readable sans-serif text, thin-to-medium dark borders, clear
  arrowheads, generous spacing, and no crossing feedback/support arrows.
- No gradients, shadows, textures, decorative characters, fake charts, provider
  logos, quality badges, numeric review scores, or claims of paper reproduction.
- No overall figure title or caption inside the image; lane headings are useful.
- If layout space is tight, prioritize all five author stages, correct Critic loop,
  the two Visualizer branches, input/entry/output, and the shared runtime. Compress
  setup and the optional upstream route into short notes.

## Source files used to verify this brief

- `integrations/codex/paperbanana_codex.py`
- `integrations/codex/codex_backend.py`
- `integrations/codex/install.py`
- `utils/paperviz_processor.py`
- `agents/retriever_agent.py`
- `agents/planner_agent.py`
- `agents/stylist_agent.py`
- `agents/visualizer_agent.py`
- `agents/critic_agent.py`
