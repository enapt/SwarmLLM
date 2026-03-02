# Getting Started

SwarmLLM lets you run AI language models on your own computer — and team up with others to run even bigger models together. It's free, open-source, and your conversations stay private.

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
