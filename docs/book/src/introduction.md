# SwarmLLM

> **Run AI on your own computer. Run bigger AI together.** SwarmLLM is a free app that gives you a ChatGPT-style assistant running on your own computer. When a model is too big for one computer, computers running SwarmLLM team up over the internet and run it together, each doing part of the work. No account, no subscription, no ads, no crypto.

This site is the full user guide and reference. For source code, releases, and issues, head to [`enapt/SwarmLLM`](https://github.com/enapt/SwarmLLM).

## What you can do with it

- **Chat with AI on your own computer** — start SwarmLLM and it opens in your browser at `localhost:8800`, shows which models fit your computer, and downloads the one you pick.
- **Use models too big for your computer** — computers in the swarm each hold some parts of a model and work through your question together. It is fastest when they are close to each other; across continents it works, but slowly. Today the swarm runs medium-sized models; giant ones are the goal. No computer ever needs to download the whole model.
- **Link your own devices** — put your laptop, desktop and home server into one private group and use them together.
- **Plug it into the apps you already use** — an OpenAI-compatible API, the Anthropic Messages API (it works as a Claude Code backend), an MCP server with seven tools, and optional routing to 12 cloud providers with your own keys.
- **Privacy, honestly** — connections between computers are encrypted, but in the public swarm the computers doing the work see what they process, so don't send anything sensitive. On your own computer, or with Private Mode, your requests stay on your devices. [Details](./architecture/security.md).

Speed depends on your hardware and the model; current measurements and the settings that affect them are on the [Performance](./operations/performance.md) page.

## Where to go next

<div class="next-steps">
<a href="./getting-started.html"><strong>Getting Started →</strong><span>Install SwarmLLM, download your first model, send your first message.</span></a>
<a href="./getting-started/joining-network.html"><strong>Joining the Network →</strong><span>How your computer finds others, and how to keep your requests on your own devices.</span></a>
<a href="./troubleshooting.html"><strong>Troubleshooting →</strong><span>Common problems, diagnostics, and how to file a useful bug report.</span></a>
<a href="./architecture/overview.html"><strong>Architecture →</strong><span>Subsystems, network protocols, encryption model, scheduler design.</span></a>
<a href="./api/openai.html"><strong>API Reference →</strong><span>OpenAI · Anthropic · MCP · Responses · Admin endpoints with examples.</span></a>
<a href="./operations/performance.html"><strong>Performance →</strong><span>The full speedup stack and how to tune it for your network.</span></a>
</div>

## Status

An early (alpha) version that improves every week and updates itself. Splitting a model across computers works, but is only fast when those computers are close together. Every change runs through thousands of automated tests before it ships. [Report issues](https://github.com/enapt/SwarmLLM/issues).

## Platform support

| Your computer | Download | Graphics card |
|---|---|---|
| Windows (64-bit) | Yes | NVIDIA, AMD or Intel for models on your own PC; splitting a model with other computers on the graphics card needs an NVIDIA RTX 30-series or newer |
| Linux (64-bit Intel/AMD) | Yes | NVIDIA RTX 30-series or newer; older cards fall back to the processor automatically |
| Mac with an Apple chip (M1 or newer) | Yes | Processor only for now |
| Mac with an Intel chip | No — build from source (best-effort) | Processor only |
| Linux on ARM | No — build from source (best-effort) | Processor only |

All downloads are on the [Releases page](https://github.com/enapt/SwarmLLM/releases).
