# Getting Started

SwarmLLM runs AI models on your own computer and, when a model is too big for one computer, lets computers team up over the internet to run it together. It is free and open source — no account, no subscription, no crypto.

> **Is it private? It depends on where the model runs.**
>
> - **On your own computer — yes.** When a model runs entirely on your
>   computer, your conversation doesn't leave it.
> - **In the public swarm — only partly.** Connections between computers are
>   encrypted, so nobody watching the internet can read them. But the
>   computers doing the work for you see what they are working on, and a
>   determined person running one of them could reconstruct your question.
>   **Don't send anything sensitive through the public swarm.**
> - **Private Mode** keeps your requests on your own linked devices — and,
>   unless you switch that off, other SwarmLLM computers on your local
>   network. Your computer still helps others with what it can spare. Set it
>   up under **More → My Devices**.
>
> **Start and finish on this computer** (the button above the chat box) keeps
> the first and last steps on your computer, so no other computer is handed
> your words as text.
> It switches itself on for any model where you hold the first and last parts.
> It is an extra layer, **not** a guarantee: the helpers still compute on
> numbers that published attacks turn back into most of your text. It also
> makes replies slower (a few seconds extra on a short reply) and needs those
> two parts on your disk. Details: [Security & Encryption](./architecture/security.md).

This guide walks you through installation, downloading your first model, and chatting.

## Prerequisites

- Windows (64-bit), a Mac with an Apple chip (M1 or newer), or 64-bit Linux
- At least 4 GB of RAM (8+ GB recommended)
- At least 2 GB of free disk space (more for larger models)
- An internet connection (for downloading models and connecting to other computers)

**What leaving it running means.** While SwarmLLM is open, your computer
shares what it can spare with the swarm. It doesn't notice when you're busy
or gaming — set a **Resource Schedule** in the app to hold back at the times
you choose, or close the SwarmLLM window. It also uses internet data every
day just staying in touch with the swarm, and more while it shares model
parts, so it is not a good fit for a capped connection. It doesn't start with
your computer.

## Chapters

- [Installation](./getting-started/installation.md) — Download and run SwarmLLM on your computer
- [First Model](./getting-started/first-model.md) — Download and chat with your first AI model
- [Joining the Network](./getting-started/joining-network.md) — How your computer finds others, and how to keep your requests on your own devices

## Quick Commands

On Windows, type `swarmllm.exe` instead of `./swarmllm`.

```bash
./swarmllm run                  # Start SwarmLLM (default port 8800)
./swarmllm run -p 9000          # Start on a different port
./swarmllm run -v               # Start with more detailed logging
./swarmllm status               # Check that it is running and what it is doing
./swarmllm chat                 # Chat in the terminal
./swarmllm bench                # Measure speed
./swarmllm peers                # List connected computers
./swarmllm diagnostics          # A report for a bug report — safe to post publicly
./swarmllm version              # Show version number
```
