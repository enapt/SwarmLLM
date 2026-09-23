# How SwarmLLM Compares

SwarmLLM sits beside two kinds of project: apps that run AI on ONE machine
(Ollama, LM Studio, Jan), and networks that split a model across MANY
(Petals, exo, Bittensor). This page compares it with the second kind, where
the differences are less obvious.

| | SwarmLLM | Petals | exo | Bittensor |
|---|---|---|---|---|
| **Install** | Download and run (one program) | `pip install` | pip / source / macOS app | pip + blockchain setup |
| **Language** | Rust | Python | Python | Python + Substrate |
| **Where the computers are** | Internet, your home network, or your own devices — found automatically | Internet (volunteers) | Your home network or Tailscale | Internet (blockchain) |
| **Traffic between computers** | Encrypted (libp2p Noise / TLS; activations additionally sealed with X25519 + ChaCha20-Poly1305) | Encrypted (libp2p) | — | Subnet-dependent |
| **Can the computers doing the work read your request?** | Yes, in the public swarm — so SwarmLLM has **Private Mode**, which keeps requests on your own devices (and, by default, other SwarmLLM computers on your local network) | Yes ([Petals' own wiki](https://github.com/bigscience-workshop/petals/wiki/Security,-privacy,-and-AI-safety)) | Yes (they are yours) | Yes |
| **Rewards** | None — no token, no payment, no blockchain | Name on a monitor page | None | TAO token (real money) |
| **How a model is split** | Pipeline (tensor parallelism exists for fast local networks but is off by default) | Pipeline | Tensor + pipeline | Subnet routing |
| **Model families verified on real models** | Llama, Qwen 2/3, Mistral, Gemma 2, Phi-3/4, GLM-4 (checked at every release). Llama 4, DeepSeek and Qwen 3.5 are implemented but not yet verified on a real model | Llama, Mixtral, Falcon, BLOOM | Llama, Mistral, Qwen, DeepSeek, LLaVA | Any (subnet-defined) |
| **Needs the whole model file?** | No — each computer downloads only its parts | Loads whole blocks | Yes | N/A |
| **Cloud fallback** | Optional, 12 providers with your own keys | No | No | No |
| **Images and adapters** | Vision models (LLaVA verified) and per-request LoRA | LoRA | Vision experimental | Subnet-specific |
| **API** | OpenAI + Anthropic + MCP (works as a Claude Code backend) | PyTorch / Transformers | OpenAI + Claude + Ollama | Subnet-defined |
| **Built-in app** | Dashboard, chat, setup wizard; 21 languages | Basic chatbot | Dashboard | None |
| **SDKs** | Python and JavaScript clients in the repo (with LangChain and LlamaIndex adapters); not yet on PyPI/npm | Python | — | Python |

⚠ **Splitting a model is only fast when the computers are close.** Every
network on this page passes your request through each computer for every word
of the reply, so the round trip between them is paid once per word. Two
computers on the same local network are fast; two on different continents
are slow. SwarmLLM's plan for automatically forming teams of nearby computers
is in [the regional pipelines plan](https://github.com/enapt/SwarmLLM/blob/main/docs/plans/regional_pipelines.md).

Details on the others change quickly — check each project's own page before
relying on a row.
