# Getting Started

SwarmLLM lets you combine your hardware with others to run AI models too large for any single machine — for free, with no API tokens or cloud fees. It's open-source, and traffic between machines is always encrypted.

> **On privacy, precisely.** Encrypted traffic means nobody *between* two
> machines can read what passes between them. It does not mean the machine
> running the model cannot read your prompt — it has to, in order to answer
> you, in the same way any AI provider does. If you want your prompts and
> answers to stay on your own machine, turn on **prompt privacy** (the
> "Enable prompt privacy" button above the chat box, or
> `inference.encrypted_pipeline`). Your machine then does the reading and the
> writing itself, and helpers only ever see partly-processed numbers. It costs
> a few seconds per answer and needs you to hold the first and last part of the
> model.

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
