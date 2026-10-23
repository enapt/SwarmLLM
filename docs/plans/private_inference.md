# Private inference: what this network can promise, and what "no plaintext" would cost (2026-10-23)

**The question** (the user, 2026-10-23): end-to-end encryption with no plaintext anywhere. No
machine but the user's own may be able to read the conversation, while other machines do the
compute. The user also asked whether parallelism would pay for "proper" encryption, and whether
our own kernels could stand in for secure-hardware cards.

**The answer in one line.** A machine that runs PUBLIC layers on the real hidden state can read
it. Hiding the state needs one of three things: cryptography, about two orders of magnitude too
slow on our links; a disguise, broken when the weights are public; or attested hardware, which no
consumer card has. What follows is the evidence, what we promise instead, and the two designs kept
with their triggers (FUTURE_WORK #328, and #327 for the route).

## 1. What a peer reads today

The table is in `docs/ARCHITECTURE.md` § Pipeline Privacy Model:
- The first segment reads token ids, unless `local_embedding_privacy` is on. Even then it is sent
  rows of its own embedding table, which it maps back exactly.
- The last segment samples, so it reads the reply, unless "Start and finish on this computer" is
  on. In a flowing tree the far end draws EVERY row, so it also reads every guess.
- Middle segments read hidden states, which are invertible:
  - 88.4% token accuracy on Llama-65B from any position in a 3-5 machine chain (2503.09022), and
    87-93% across positions.
  - Gaussian noise at σ = 1 still leaves 79.5%, at perplexity 14.2. 4-bit quantization costs
    perplexity 28.3.
  - Over 90% on Llama-3.2 and Phi-3.5 (2503.09291).
  - Keeping the first and last layers local still leaves ~59% recoverable at a 2-layer split
    (2602.16760).
  - Petals' own security page says peers can recover inputs and outputs.

## 2. Why "no plaintext" is not an engineering gap

The attacks above work because the attacker holds the public model. A machine that computes the
public layers on the real state holds everything it needs to invert it. Hiding the state from that
machine leaves exactly three families.

**(a) A disguised state** — permutations, rotations, "equivariant encryption". The machine computes
on a transformed copy at near-zero cost.
- Public weights break the permutation schemes. 2505.18332 recovers prompts "with nearly perfect
  accuracy" from the hidden-dimension permutations of PermLLM (2405.18744), STIP (2312.00025) and
  Centaur (2412.10652). Ritual reproduced it on Llama-3.1-8B: a 50-token prompt in minutes.
- A fixed rotation of the residual stream (QuaRot's invariance, used for privacy) is a Procrustes
  problem against the public Q/K weights. It also falls to norm-matching against the public
  embedding table, which 2606.16461's own attack section calls nearly trivial. Unproven at best.
- Nesa's equivariant encryption (2502.01013) reports +8.88% latency on Llama-3.1-8B. It argues
  security from the size of the permutation space, tests no attack and does not address public
  weights.
- None of these is a basis for a claim.

**(b) Cryptography.** The best measured results for 6B-8B models:

| Work | Model | Time | Link | Traffic | Trust |
|---|---|---|---|---|---|
| PermLLM 2405.18744 | ChatGLM-6B | ~3 s/token; ~7 s/token | 10 ms, 1 Gbps; 20 ms, 100 Mbps | — | 2PC + a permutation assumption |
| Centaur 2412.10652 | LLaMA-7B, 8 in, 1 out | 3.02 s | not stated | 0.39 GB | 2PC, permuted plaintext |
| BumbleBee (NDSS'25) | LLaMA-7B, 8 in, 1 out | 7.96 min | 1 Gbps LAN | 6.27 GB | 2PC |
| SIGMA eprint 2023/1269 | Llama2-7B, 128 tokens | 5.72 min | 305 Mbps, 60 ms | 12.07 GB + 255 GB keys | 2PC + dealer |
| POST 2505.15252 | Vicuna-7B + 68M drafter | ~3.4 s/token (20.32 s a forward ÷ 6.0) | 400 Mbps, 80 ms | — | 2PC, speculative |
| Cachemir 2602.11470 | Llama-3-8B decode | 1.61 min/token | one A100 | — | FHE |
| Cerium 2512.11269 | Llama3-8B, 128-token prefill | 43 s | 8× B200 | — | FHE |

Our link is Chiang Mai ↔ Melbourne: a 387 ms minimum round trip and a home uplink of ~50 Mbit/s.
- PermLLM's ~20 rounds a layer, across 28 layers at 390 ms, is ~220 s a forward from latency alone.
- Centaur's 0.39 GB is 62 s at 6.25 MB/s.
- SoK eprint 2026/935 puts the gap at about four orders of magnitude.

The second problem is independent of speed. Two-party schemes are private only while the two
machines do not compare notes. In an open network one person can run both (a Sybil), and no
protocol can tell.

**(c) Attested hardware.** NVIDIA confidential computing (R580 release notes) covers:
- H100, H200, H20 and H800;
- B200 and HGX B300;
- RTX PRO 6000 Blackwell Server Edition.

It needs a TDX or SEV-SNP host. No GeForce card is on any supported list.
- **Cost:** 4-8%. Llama-3.1-8B at batch 1 ran 123.3 against 132.4 tok/s (2409.03992).
- **Guarantee:** plaintext only inside hardware whose measurements NVIDIA's chain signs. The client
  verifies the GPU report and the CPU quote (NRAS or nvtrust's local verifier), each bound to a
  nonce and to the key it seals to.
- **Out of scope:** physical attacks. TEE.fail extracted TDX and SEV-SNP keys with a sub-$1000
  DDR5 interposer. Hopper's card memory is access-controlled, not encrypted.
- **Who runs it:** NEAR AI, Venice and Phala run this, each on hardware they operate.

### Why parallelism does not close it

More machines divide the WORK. Private inference is bound by the CHAIN of steps: rounds between
parties for MPC, bootstraps a layer for FHE. More machines do not shorten that chain, any more
than more lanes shorten a road.

Our one structural lever is speculation, and it is prior art. POST (2505.15252) checks a plaintext
drafter's guesses in one secure forward and gains 2.1-6.0×, with chains of 4-16. Speculation
divides ROUNDS by E, but it multiplies bytes and compute by the rows checked a token. Our flowing
trees check 22-28 rows a token, so under bandwidth-bound MPC a tree LOSES and a short chain is the
best case. (A message to the user on 2026-10-23 claimed 5-7× for our trees and called the idea new.
Both were wrong: POST already did it, and the rows cancel most of it.)

### "Our own kernels"

Our kernels are programs the machine's owner runs. Confidential computing rests on what the owner
cannot touch:
- a key fused into the chip and certified by the maker;
- encryption on the bus;
- firmware that refuses a debugger.

On a GeForce card the owner controls the driver and every byte of card memory, and can run a
modified build. No software can prove to a distant user that it runs unmodified without such a key.
Obfuscating the kernel is white-box cryptography, which an owner who watches every instruction
defeats. 2602.11088 broke precomputed-noise shielding schemes on LLaMA-3-8B in ~6 minutes a layer.

## 3. What is true, and what we say

1. **Private mode keeps every word among machines you chose.** It uses your pool, and LAN only if
   `private_mode_allow_lan` is on — off it isolates, #352. Every hop is sealed (ChaCha20-Poly1305
   per session, X25519). A relay in between sees only a body sealed to the recipient
   (`crypto::relay_seal`). This is end-to-end in the messaging sense: the endpoints are yours.
2. **"Start and finish on this computer" + `local_embedding_privacy`.** No token id and no sampled
   token leaves. Peers in the middle read hidden states that can be turned back into most of the
   text, and the interface says so in its own words — never "private" or "encrypted"
   (CLAUDE.md § Key Design Decisions).
3. **Not promised:** that a stranger's machine cannot read what it computes on.

## 4. Designs kept, with what would start them

**(a) Private work on attested hardware — FUTURE_WORK #328.** Route a request marked private only to
peers whose GPU report and CPU quote verify. The session key is sealed to a key generated inside the
attested VM, and the reply is signed there. This is the one design under which "no plaintext
outside the user's machine and attested hardware" is TRUE, at 4-8% cost.
- **Needed:** a verifier for NVIDIA's certificate chain and measurements and for TDX/SEV-SNP quotes
  (crypto and attestation → `security-reviewer`); a capability bit; a routing scope beside
  `pool::scope::request_scope`.
- **Trigger:** a swarm peer with a confidential-computing GPU. Today none: 4 peers, one RTX 3060,
  three without a card.

**(b) Client-aided split, weights public — NOT built.** The owner keeps every weight-free step in
plaintext: attention, norms, activations, the KV cache. Two machines compute only the matmuls, on
additive shares. A second server precomputes the masks' products, which do not depend on the input.
- **Rounds:** for the 7B, 2-4 owner round trips a layer (56-112 a forward).
- **Bytes:** ~9 MB a row a forward in a 32-bit ring, plus ~5.6 MB of mask products. That is
  bandwidth-bound on any home link.
- **Risks:** reused masks are attacked (2602.11088), and the two servers must not collude.
- POST's chain speculation is its best case. Recorded so the arithmetic is not redone.

**(c) The route, not the cipher — #327.** On this link the round trip is a bad route, not the
distance:
- **The route:** Chiang Mai → Bangkok → Los Angeles → Sydney → Melbourne, 387 ms.
- **The alternative:** from the same machine, Singapore answers in 40 ms. A detour through a peer
  on that path would roughly halve each miss's cost.
- **Not a leak:** the relay already carries sealed bodies it cannot read.
- **Trigger:** a well-placed peer, which the swarm does not have.
