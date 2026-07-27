# Getting Started

SwarmLLM lets you combine your hardware with others to run AI models too large for any single machine — for free, with no API tokens or cloud fees. It's open-source, and traffic between machines is always encrypted.

> **On privacy, precisely.** Encrypted traffic means nobody *between* two
> machines can read what passes between them. It does not mean the machine
> running the model cannot read your prompt — it has to, in order to answer
> you, in the same way any AI provider does.
>
> **Turning on prompt privacy is recommended** — it is the only setting that
> stops other machines reading your prompts. Use the "Enable prompt privacy"
> button above the chat box, or `inference.encrypted_pipeline` in your config.
> Your machine then does the first and last steps itself, and helpers only ever
> see partly-processed numbers.
>
> What it costs, so you can decide:
>
> | | |
> |---|---|
> | **Disk** | you need the first *and* last piece of the model on this machine |
> | **Speed** | your machine swaps data with helpers once per word, so a long answer costs proportionally more time — a few seconds extra on a short reply |
> | **Your hardware** | does more of the work, since the first and last steps run here |
> | **Scope** | set per model, so you can have it on where it matters and off elsewhere |
>
> It is off by default only because it cannot route unless you hold both ends of
> the model. If you do, turn it on.

This guide walks you through installation, downloading your first model, and chatting.

## Prerequisites

- A computer running Windows, macOS, or Linux
- At least 4 GB of RAM (8+ GB recommended)
- At least 2 GB of free disk space (more for larger models)
- An internet connection (for downloading models and connecting to peers)

## Chapters

- [Installation](./getting-started/installation.md) — Download and run SwarmLLM on your platform
- [First Model](./getting-started/first-model.md) — Download and chat with your first AI model
- [Joining the Network](./getting-started/joining-network.md) — Connect to peers for distributed inference

## Quick Commands

```bash
./swarmllm run                  # Start the node (default port 8800)
./swarmllm run -p 9000          # Start on a different port
./swarmllm run -v               # Start with verbose logging
./swarmllm status               # Check if the node is running
./swarmllm chat                 # Interactive CLI chat
./swarmllm bench                # Benchmark inference performance
./swarmllm peers                # List connected peers
./swarmllm version              # Show version number
```
