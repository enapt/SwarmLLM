# SwarmLLM provider for OpenClaw

Use the models a [SwarmLLM](https://github.com/enapt/SwarmLLM) node serves —
local, or split across a peer-to-peer swarm — as a model provider in
[OpenClaw](https://github.com/openclaw/openclaw). No per-token bill; the only
cost is the hardware you already own.

The plugin is built on the same OpenClaw SDK helper as the bundled vLLM and
SGLang providers, so it gives you:

- an entry in `openclaw onboard` / `openclaw configure` (**SwarmLLM** under
  "Peer-to-peer inference network"),
- live model discovery from the node — `openclaw models list --provider swarmllm`
  shows what the swarm can serve you right now, with each model's real context
  window,
- model refs of the form `swarmllm/<model-id>`.

## Before you start: two things the node needs

1. **A real access key.** SwarmLLM writes one to `api_key` in its data
   directory (also under *Settings → Access Token* in the dashboard). OpenClaw's
   "local marker" keys such as `ollama-local` send no `Authorization` header at
   all, and every `/v1` route on a node refuses a request without one.

   ```bash
   export SWARMLLM_API_KEY="$(cat ~/.local/share/swarmllm/api_key)"   # Linux
   # macOS: ~/Library/Application Support/swarmllm/api_key
   # Windows: %APPDATA%\swarmllm\api_key
   ```

   The daemon reads the same variable as its own key override, so one export
   serves both sides.

2. **A context window larger than the shipped 8192-token default.** Measured
   on a fresh OpenClaw workspace: the first turn's prompt is **14,633 tokens**,
   and OpenClaw also reserves its `maxTokens` (8192 by default) for the reply,
   so the node must accept about 23k. In SwarmLLM's `config.toml`:

   ```toml
   [inference]
   max_seq_len_override = 32768
   ```

   This is a config-file setting (not a dashboard control, not an environment
   variable). Restart the node afterwards.

## Install

```bash
openclaw plugins install clawhub:openclaw-plugin-swarmllm   # from ClawHub
openclaw plugins install npm:openclaw-plugin-swarmllm       # or from npm
```

From a checkout of the SwarmLLM repository (a local archive is outside
ClawHub's review, so OpenClaw asks for `--force`, and the provider capability
needs consent):

```bash
cd integrations/openclaw && npm install && npm run build && npm pack --pack-destination /tmp
openclaw plugins install npm-pack:/tmp/openclaw-plugin-swarmllm-0.1.0.tgz --force --accept-capabilities
```

## Configure

Interactive:

```bash
openclaw configure        # pick SwarmLLM, accept the default URL, paste or confirm the key, choose a model
```

Non-interactive:

```bash
openclaw onboard --non-interactive --accept-risk --skip-health \
  --mode local \
  --auth-choice swarmllm \
  --custom-base-url "http://127.0.0.1:8800/v1" \
  --custom-api-key "$SWARMLLM_API_KEY" \
  --custom-model-id "meta-llama-3.1-8b-instruct-q4-k-m"
```

Or by hand in `~/.openclaw/openclaw.json`. With `SWARMLLM_API_KEY` exported
and the node on its default port, no provider block is needed at all — the
model list is discovered from the node:

```json5
{
  agents: { defaults: { model: { primary: "swarmllm/meta-llama-3.1-8b-instruct-q4-k-m" } } },
}
```

For a node on another machine, add a provider block **and** the `swarmllm/*`
wildcard — without the wildcard OpenClaw takes the block's own `models:` list
as the catalog and skips discovery:

```json5
{
  models: {
    providers: {
      swarmllm: {
        baseUrl: "http://192.168.1.20:8800/v1",
        apiKey: "${SWARMLLM_API_KEY}",
        api: "openai-completions",
        // Generous on purpose. An agent turn sends thousands of tokens of
        // tool schema, reading them takes minutes on a processor-only node,
        // and a reply that might be a tool call arrives in one piece at the
        // end — so OpenClaw can legitimately see nothing for a long time.
        timeoutSeconds: 1800,
      },
    },
  },
  agents: {
    defaults: {
      model: { primary: "swarmllm/meta-llama-3.1-8b-instruct-q4-k-m" },
      models: { "swarmllm/*": {} },
    },
  },
}
```

Then:

```bash
openclaw models list --provider swarmllm
openclaw agent exec "Say hello" --model swarmllm/meta-llama-3.1-8b-instruct-q4-k-m
```

## What to expect on a small graphics card

Verified end to end against a live node with OpenClaw 2026.8.2: install,
onboarding, discovery of every model the node serves, and a turn that reaches
the node and streams. The honest numbers from an RTX 3070 (8 GB) running a
3B model: OpenClaw's 14,633-token first turn is read in about 15 s, but the
KV cache for that many tokens is 5 GB beside the weights and fills the card,
and a reply to a 30-tool agent prompt then came back at about one token per
second — and only once the model stopped, because a reply that may be a tool
call is buffered until it can be told from prose. A 3B model also does not
follow a prompt of that size; it rambles to the reply cap. Use the largest
model the swarm offers you, keep `maxTokens` modest (the README config sets
4096), and expect a card with more memory, or a peer that has it, to be what
makes the loop comfortable. The node-side work this points at is recorded in
`docs/FUTURE_WORK.md` in the SwarmLLM repository.

## Which model?

Whatever `openclaw models list --provider swarmllm` shows. A node lists every
model the swarm currently covers — held locally, split between this machine and
peers, or served entirely by peers. For a tool-heavy agent loop pick the
largest one on offer: small models call tools less reliably, and serving models
your machine could not hold alone is what the swarm is for.

## Troubleshooting

- **No models listed.** `SWARMLLM_API_KEY` is unset or wrong (discovery sends
  it as a Bearer token and the node answers 401 without it), or the node is not
  running at the configured `baseUrl`. `curl -H "Authorization: Bearer $SWARMLLM_API_KEY" http://127.0.0.1:8800/v1/models`
  should list models.
- **"Context overflow: prompt too large for the model"** (OpenClaw), or in the
  node's own words *"This conversation is too long … the model's limit is
  8192"*. The context override above is not set, or the node was not restarted
  after setting it. OpenClaw's auto-compaction cannot help on a first turn —
  there is nothing to compact yet.
- **Every model shows a 128000 context window.** The node predates
  v0.3.148, whose `/v1/models` reports `context_length`; upgrade it, or set
  `contextWindow` by hand in a `models:` block.
- **`context_length` reads 8192 for every model.** That figure is the NODE's
  configured context, not the model's: it reports whatever
  `max_seq_len_override` in the node's `config.toml` allows (8192 until you
  set it, as in step 2 above) and changes only after the node restarts. The
  model's native window does not matter until the node is allowed to use it.
- **`openclaw models list --provider swarmllm` says "No models found" although
  `curl` to `/v1/models` with the same key works.** One gateway showed this for
  the bundled `ollama` provider too, so it is the gateway's shared discovery
  path rather than this plugin. Declare the models by hand for now:
  `models.providers.swarmllm` with a `models:` array (`id` and `contextWindow`
  per model, as `/v1/models` reports them) and `agents.defaults.model.primary`
  set to `swarmllm/<id>`. Requests then reach the node normally.
- **The turn takes ages and then times out.** Three different causes, and
  they are easy to tell apart from the node's own log.
  - *A model split across machines, and the node is v0.3.153 or older.* Look
    for `Tensor too large` in the node's log. Any prompt past about 8,000
    tokens was refused outright by the machine receiving it, so OpenClaw's
    retries each hit the same wall and the session eventually gave up. Fixed
    in the release after v0.3.153 — upgrade the nodes.
  - *Nothing arrives until the model stops.* Expected, for now: a reply that
    might be a tool call is buffered until it can be told from prose, and
    OpenClaw always sends tools. Raise `timeoutSeconds` as above; the node
    keeps the connection alive throughout.
  - *Reading the prompt is simply slow.* `DIAG` lines in the node's log time
    the prompt pass. A 14 k-token turn is minutes on a processor and seconds
    on a card — use the largest model the swarm offers, and check the node's
    dashboard says the model is on the graphics card rather than the
    processor.
- **Embeddings.** A node does not serve `/v1/embeddings`; point OpenClaw's
  memory search at another embedding provider.

## Development

```bash
npm install
npm run build     # tsc → dist/ (what the package ships and OpenClaw loads)
npm test          # vitest: registration + wizard wiring, and discovery against a real node's /v1/models body
                  # Needs Node 22.5+ — OpenClaw's own state DB imports `node:sqlite`,
                  # so on Node 20 the suite fails to load with
                  # `No such built-in module: node:sqlite` before any test runs.
npm run validate  # build + `clawhub package validate`
```

The plugin is intentionally small: `src/index.ts` calls
`defineSelfHostedOpenAICompatibleProvider` with SwarmLLM's defaults and adds a
hint for unknown model refs. Everything else — auth prompts, discovery, the
wizard — is OpenClaw's own, shared with its bundled self-hosted providers.

## Publishing

From this directory, after `npm run validate`:

```bash
npm exec clawhub -- login
npm exec clawhub -- package publish .
```

ClawHub's trusted-publishing workflow (GitHub OIDC) publishes from a repository
root, so it is not wired up here; if the plugin moves to its own repository,
`openclaw plugins init <id> --type provider` generates the workflow to copy.
