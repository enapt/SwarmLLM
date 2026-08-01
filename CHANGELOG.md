# Changelog

All notable changes to SwarmLLM are documented here.

## [0.3.60-alpha] — 2026-08-01

Splitting a model across machines now works end to end, and a modest graphics
card can run modern models again.

### Fixed

- **A model split across three machines could fail with a misleading error
  about tensor data.** When one machine took too long, the work was handed to a
  standby — but the abandoned attempt, on being cleaned up afterwards, reported
  its failure against the request that had already moved on. The standby's
  correct answer then arrived to find nothing waiting for it and was discarded,
  and the empty result from the failure was passed to the next machine in the
  chain, which reported it as malformed data. Measured on two machines: the
  standby had produced a correct answer in under ten seconds; the request
  instead failed after three minutes. Answers are now matched to the machine
  they were expected from, so an attempt that has been given up on can no
  longer answer for the one that replaced it. Three consecutive three-way
  splits of an 8B model have since completed correctly.

- **A modest graphics card refused models it had room for.** A 6GB card was
  turning away Gemma 2 2B — a model whose weights are 1.6GB — with the card
  completely empty. The estimate of 5.4GB was accurate: 2.2GB of it was the
  token vocabulary table, held at full precision, which on today's
  large-vocabulary models is more memory than the entire rest of the model. It
  is now held at half precision, which is all the values were ever worth, and
  the result of each lookup is widened before use so nothing else changes.
  Verified by asking two models the same questions before and after — every
  answer identical, including across a three-machine split. Models gain in
  proportion to their vocabulary; Llama 3.x and Gemma 2 gain the most.

- **Work could be sent past a good nearby machine to a distant one.** Two
  separate causes. A machine reachable only through a relay could outrank one
  we were connected to directly, because the rule meant to prevent that added a
  fixed amount to the relayed machine's travel time — and a fixed amount cannot
  order two things. A machine we had never timed was also assumed to be fast,
  so knowing nothing about it counted in its favour. Separately, the choice
  between machines considered current workload before travel time, and workload
  counts whole requests, so a single request already running on a machine 4ms
  away was enough to divert the next piece of work to one 400ms away. Distance,
  speed and workload are now weighed against each other rather than checked in
  order.

- **The test suite could overwrite a running node's API key**, after which
  every request failed to authenticate with nothing in the log to explain it.
  This was reported fixed previously; the guard only covered part of the build.
  Writing the file is no longer a side effect of reading the key. If you meet
  this on an older build, restarting the node restores the file.

### Changed

- **Nodes now learn how fast each machine they work with actually is**, and
  wait accordingly, instead of applying one fixed allowance to everything from
  a laptop CPU to a datacentre GPU. Prompt processing and answer generation are
  measured separately — they differ by around a hundredfold on the same machine
  — and a machine that has not served a model recently is given room to load it
  first. That last case is what caused the failure above: a machine needed two
  minutes to load an 8B model, did the work itself in ten seconds, and was cut
  off at the two-minute mark. Estimates are forgotten after an hour of silence,
  so a machine that was once busy is not written off for ever.

- Per-machine speed figures are now visible in the admin performance view.

## [0.3.59-alpha] — 2026-08-01

### Fixed

- **A model was refused for lack of memory even when the memory was free.** With
  one model resident and a request arriving for another, the node compared the
  new model against a budget the first was still holding and refused outright —
  reported from a node with 6210 MB resident, 5986 MB wanted and an 8000 MB
  limit. Nothing was actually using the resident model; it was simply still
  loaded, and only a later background pass would clear it. Memory is now
  reclaimed from models nothing is using before a refusal is returned, so the
  request succeeds instead of failing. Models with work in flight are never
  touched, so this cannot interrupt an answer in progress, and a model that
  genuinely does not fit on its own is still refused — with a message that no
  longer implies another model is to blame.

## [0.3.58-alpha] — 2026-07-31

### Fixed

- **Running the test suite broke a node on the same machine.** Building the
  daemon's shared state resolves its API key and writes it into the data
  directory, and a test that built one from a default configuration inherited
  the real data directory even though its database was a temporary one. The key
  it generated was written over the running node's key file while that node
  carried on using the one in its own database, so anyone running `cargo test`
  with a node up lost the dashboard, the command-line tools and every saved
  token at once — with nothing in the log, presenting as an unexplained
  authentication failure in the daemon. The writer is now inert in test builds,
  so no future test can reintroduce it by forgetting to redirect the directory.

- **A model could be unloaded seconds after it loaded, killing the request that
  loaded it.** On a node answering its own client, a reply could fail with
  "worker closed connection mid-generate" while the log claimed the model had
  been idle for the full configured timeout — for a model that had existed for
  seven seconds. Three things were wrong at once. Whether a model is busy was
  worked out from bookkeeping that only covers requests passed to other machines
  or served for them, so a node answering its own client locally counted as
  doing nothing; that is now read from the worker itself, which every kind of
  request goes through. A model with no request history was treated as having
  been idle for ever, rather than for at most as long as it had been loaded. And
  the message named graphics memory on machines that have none, sending at least
  one report looking for a graphics fault that was not there — it now reports
  the memory it actually freed and the idle time it actually saw.

## [0.3.57-alpha] — 2026-07-31

### Fixed

- **v0.3.56-alpha shipped with no Windows GPU download.** Every other platform
  built and published normally, but the Windows GPU archive is missing from
  that release entirely — if you use it, stay on v0.3.55-alpha until the next
  release rather than switching to the Windows CPU build, which will not use
  your graphics card. The cause was our own build caching: the job that keeps
  the build cache warm installed a different version of the graphics toolkit
  than the release did, and the cached build files still referred to the older
  one by path. The toolkit version is now part of the cache's identity, so a
  mismatch rebuilds from scratch instead of reusing something that no longer
  matches. Publishing also now checks for each platform's download by name —
  the previous check counted files, and a count could not tell a complete
  release from one missing a whole platform, which is how this reached users
  at all. A release missing a platform is now held back instead of published.

## [0.3.56-alpha] — 2026-07-31

### Fixed

- **The memory limit did nothing at all.** `max_ram_mb` has shipped in the
  configuration file since it was written, documented there and in the
  reference as "0 = auto (50% of system RAM)" — and nothing in the program ever
  read it. Setting it to protect a small machine had no effect whatsoever, and
  no message said so. In practice an 8 GB machine could be driven into swap,
  which slows down every request on it rather than just the model responsible.
  It now works as documented: models loaded on the processor are held to the
  limit, and a figure larger than the machine is reduced to what can actually
  be spared. Left automatic, the allowance is half the machine where the
  graphics card does the work, and most of it on a processor-only node — where
  serving models is the whole point of the machine, and half would have meant
  refusing models such nodes run today. A model that will not fit is refused
  with a message naming what it needed, what the limit is, and how to raise it
  — rather than being loaded into swap. This matters more since v0.3.55 began
  moving models to the processor when the graphics card is full: that fallback
  is what keeps a node answering, and it had no ceiling.

- **A node that had returned any credits was left with its totals permanently
  not adding up.** The repair shipped in v0.3.55 brought the books back into
  balance on nodes that had been running before the "credits returned" figure
  existed — but it skipped any node that had *already* recorded a return,
  on the reasoning that such a node needed no help. That is two different things
  confused for one: returns recorded since the figure was added are counted
  correctly, while the older, unrecorded ones still need explaining, and a node
  can easily have both. A single returned payment between the two releases was
  enough to have the repair skip a node for good. It now works from whatever
  remains unexplained, so it corrects the rest without recounting anything
  already recorded, and does nothing at all on a node whose books already
  balance. Reported from a live node carrying a ~905k discrepancy.

- **A peer that left kept being answered every 30 seconds, indefinitely.** Nodes
  announce themselves over a mesh that forwards messages between peers that
  cannot reach each other directly, so a node that had dropped off kept being
  heard from long after there was any way to reply to it. Each of those
  announcements drew a reply that could only be thrown away, once every 30
  seconds for as long as the node stayed up — in an overnight run a single
  departed peer produced **45% of the entire log**, burying anything else worth
  reading. Replies are now addressed only to peers there is still a live
  connection to. The same blind-addressing pattern is fixed in the two other
  places it appeared, including key rotation, which no longer leaves behind
  half-finished handshake state for peers that are gone.

## [0.3.55-alpha] — 2026-07-30

### Fixed

- **The model list said "network" for models held completely, and "local" for
  one held only in part.** `GET /v1/models` decided this from whichever model
  was loaded most recently rather than from which pieces the node actually
  holds, so the label was effectively inverted — a client choosing models by it
  to avoid network round trips picked exactly wrong. It now reports `local` when
  every piece is here, `hybrid` when some are, `network` when none are, using the
  same count the admin API already got right.

- **The node blamed a peer for a model piece it was missing itself.** A request
  that could not be routed said "a peer went offline mid-request — try again",
  when in fact a piece was absent locally and no peer had it. Following that
  advice means retrying for ever. That case now says so, and names the command
  that fetches the rest of the model.

- **A storage limit larger than the disk was taken at face value.** A 50 GB
  budget was accepted on a filesystem with 15 GB free, and with the piece caps at
  their unlimited defaults the node would keep accepting until the disk was full
  rather than making room. The budget is now capped to what is genuinely free,
  with a warning saying so.

- **Testing a Cerebras key failed with an unexplained 404.** The built-in model
  used to check that provider's key had been retired by them, so a user with a
  perfectly valid key got a bare "model does not exist" on their very first
  attempt, with nothing to indicate our default was at fault rather than their
  key. The dashboard and the daemon also disagreed about which model to use, and
  both were wrong. Note these built-in choices go stale whenever a provider
  retires a model; asking each provider for its own list would be the durable fix.

- **The Sybil check flagged the project's own bootstrap server.** Every node
  behind a router logged "subnet clustering detected" against the anchor's own
  address range every few minutes, which raised the inspection rate on the one
  relay those nodes depend on and buried real warnings. The addresses the daemon
  itself ships as bootstrap servers are now recognised.

### Added

- **Credit figures on `/metrics`.** `swarmllm_credits_earned_total`,
  `..._reserved_total` and `..._returned_total` alongside the existing balance —
  previously only the balance was exposed, so Prometheus users saw far less than
  the dashboard.

- **Model loads now report how much graphics memory they actually used**
  (`vram_after_load_mb` in the log). Groundwork for refusing a model that will not
  fit rather than discovering it by running out: the existing estimate is derived
  from file size and was found to be 56-117% low, because the largest single
  component — the vocabulary table, which is expanded in memory — is invisible to
  it. On one 1B model that table alone is bigger than the whole file.

### Fixed

- **Freeing graphics memory now actually frees it.** When memory ran short the
  daemon made room by dropping models from its own list — but that list is only
  bookkeeping. The memory is held by separate worker processes, and nothing told
  them to let go, so the node believed it had freed several gigabytes that were
  still in use. Dropping a model now stops its worker, which is the only thing
  that returns the memory.

  A model that was moved to the processor after running out of graphics memory
  is now returned to the graphics card once memory comes back, instead of
  staying there for as long as the node runs. `GET /api/admin/stats` gained
  `models_on_cpu_fallback`, because the existing `inference_backend` field
  describes which build is running rather than what any model is actually
  using — it reported the graphics card throughout.

- **A model that will not fit is now loaded on the processor instead of failing.**
  Loading a model never checked whether there was room for it, so on a card that
  could hold one large model but not two, the second simply ran out of memory
  mid-load. That killed its worker and left the model on the processor for as
  long as the node ran. Nobody had to be using the node for it to happen — its
  own background loading was enough.

  Each model's memory needs are now worked out from its actual shape before
  loading, weighed against what is already committed, and a model that does not
  fit is loaded on the processor deliberately, with a clear reason logged and
  shown in the interface. Verified on an RTX 3070: with room for one model, the
  first went to the graphics card and the second was placed on the processor with
  no failure at all, where previously both were attempted and the second died.

  Nodes without a configured memory limit behave exactly as before.

### Clarified

- **Correction to the v0.3.53 note on repeated prompts.** That entry read as
  though it covered any repeated prompt. It does not: the local prompt cache
  already limits itself to one piece short of the whole prompt, so it was never
  affected. The fault was only ever reachable when the saved work came from
  *another node*. Thanks to the tester who worked that out from the outside and
  asked which path was meant — the original wording was too broad.

### Fixed

- **Hosting shards earned no credits at all.** Credit for hosting is worked out
  per gigabyte and then rounded down to a whole number, one shard at a time.
  With the standard piece size of 512 MB that is half a credit per shard per
  hour, which rounded down to nothing — so on a typical node, hosting earned
  zero for ever while spending worked normally. A node that took on more work
  got steadily poorer, which is the opposite of the intent. Reported by an
  operator who went from 5 to 13 shards to help under-replicated models and saw
  nothing credited at any point.

  Hosting is now totalled across everything a node holds before rounding, and
  the fraction left over is carried to the next hour rather than discarded, so
  even a single small shard eventually earns.

- **Two Prometheus metrics never recorded anything.** The total request counter
  and the latency histogram stayed at zero through successful requests, while
  the per-route breakdown recorded them correctly — so anything built on the
  total, or on latency, read zero for ever. They were being recorded in one
  place while requests can complete by three different routes; they are now
  recorded where every route passes through.

- **Existing nodes reported their credit totals as not adding up.** v0.3.54
  started publishing credits returned, but the counter began at zero on nodes
  that had already been running — while the refunds it should have counted were
  already folded into the balance. The reconciliation flag therefore read
  "false" on every existing node, for ever: exactly the false alarm the figure
  was added to remove. The pre-existing difference is now attributed to
  historical refunds on first load, so the totals reconcile. Fresh nodes and
  nodes already recording refunds are untouched.

## [0.3.54-alpha] — 2026-07-30

### Fixed

- **Credit totals looked like they didn't add up.** The dashboard and API
  reported credits earned and credits spent, but not credits *returned*. When a
  request fails, the credits reserved for it are given back — and the "spent"
  figure deliberately never goes down, so on a node with many failed requests
  the two numbers could imply a hugely negative balance while the actual balance
  was positive. One node showed a ~905,000 discrepancy. Nothing was wrong with
  the accounting; the number that reconciled it was simply never published.

  Credits returned and net spend are now reported alongside the existing totals,
  so the figures add up. This is also worth watching: returned credits as a
  share of reserved credits is your node's own request failure rate. On the node
  that prompted the report it was 97%, which had been invisible.

## [0.3.53-alpha] — 2026-07-30

### Fixed

- **Repeating a long prompt reused the saved work slightly wrong.** When a
  prompt is sent again, the node restores the work it already did rather than
  redoing it. The bookkeeping kept one more piece of that saved work than it
  told the rest of the system about, so the last word of the prompt was counted
  twice when the model looked back over it. Nothing failed and nothing was
  logged, which is why it went unnoticed — but it only happened when the
  *entire* prompt was already saved, which is exactly the case this feature
  exists for. Anyone trying to measure the speed-up was measuring a subtly
  wrong answer.

- **Updating could silently take away your graphics-card support, with no way
  back.** Updates deliberately never switch between the graphics-card and
  processor-only builds, so an update can't hand you a binary your machine
  can't run. But a node that ended up on the processor-only build stayed there
  for ever — every later update dutifully fetched the processor-only version
  again, the graphics card sat unused, and nothing anywhere said why. One
  tester had been reinstalling the graphics-card build by hand after every
  update, unable to find an explanation.

  The daemon now says so plainly when it is the processor-only build on a
  machine with a graphics card, and names the file to install once to get back.
  It still won't switch automatically — a card being present is not proof its
  drivers can run the graphics build, and installing one that won't start is
  worse than running slowly.

  Separately, a graphics-card build compiled locally with `--features
  candle-cuda` was misreported as processor-only and would update away its own
  graphics support. It's now correctly recognised.

- **Models were being fed gibberish instead of some of the words you typed.**
  Models that use SentencePiece — Phi-3.5, Mistral, Llama 2 and many others —
  split text into pieces using a lookup table. A fault in that step meant some
  words were handed to the model as raw character codes rather than as words.
  The model received nonsense, and would often say so, which read as the model
  being stupid rather than as a bug on our side.

  It was worse than previously believed: measured against Phi-3.5's real
  vocabulary over a 4,128-line sample of ordinary sentences, code, email
  addresses and accented text, **65% of them were affected**. Asked "What colour
  is a banana?", the model previously replied `The text "a␦␦␦ debido a que
  debido a que…`; it now answers that a ripe banana is yellow. Asked what
  quantization is, it previously answered about datasets, because it never
  received the word.

  Our output is now checked line for line against the reference SentencePiece
  implementation using each model's own vocabulary file, with no differences
  across all 4,128 samples. Models using the other tokenizer style (Llama 3,
  Qwen, TinyLlama) were never affected.

## [0.3.52-alpha] — 2026-07-29

### Fixed

- **A node with one model loaded could answer questions about a different
  model.** If a model had been loaded from a single file — set at startup, or
  downloaded whole from Hugging Face — the node treated *every* later request
  as one it could answer itself, whichever model was actually asked for. The
  reply came back from the resident model, using that model's conversation
  formatting, reported as a success with the requested model's name on it.
  Nothing in the response or the logs indicated the substitution.

  A request is now served locally only when the loaded model is the one that
  was asked for. Anything else goes to the network as it always should have.
  Nodes that run on downloaded pieces rather than whole files were never
  affected.

- **The dashboard claimed models were loaded in graphics memory when they were
  not.** A node with, say, "Llama 3.2" loaded marked every model whose name
  began the same way — other sizes, other quality settings — as being in
  graphics memory too, because the names were compared by prefix rather than
  matched properly. The badge now reflects what is actually loaded.

- **Some models ran at a fraction of their speed on a GPU, with nothing to
  say so.** Models that advertise a very large maximum conversation length —
  Llama 3.2, Phi-3.5 and most other recent releases advertise 131,072 words'
  worth — reserved graphics memory for that entire length the moment they
  loaded, even for a two-line question. On a typical consumer card there was
  not enough to go round, so the graphics driver quietly spilled the overflow
  into ordinary system memory and carried on. Nothing failed, nothing was
  logged, and the model answered normally — roughly fourteen times slower than
  it should have. Measured on an RTX 3070 with Llama 3.2 1B: 3.7 words per
  second before, 46–54 after, using 4.6 GB less graphics memory.

  Models now reserve a sensible working length up front and shrink it further
  if the card is small, the same approach llama.cpp takes. Raise it with
  `inference.max_seq_len_override` if you want the full advertised length and
  have the memory for it. This also leaves room for a second model to load
  alongside the first, which previously could fail outright.

- **The daemon reported working settings as being ignored.** Any setting whose
  default is "unset" — including `inference.max_seq_len_override`,
  `api.rate_limit_rpm`, `logging.file` and `network.gossip_network_id` — was
  announced at startup as *"Unknown setting in config file — it is being
  IGNORED"*, while in fact taking effect normally. Genuine typos are still
  caught and named.

## [0.3.51-alpha] — 2026-07-29

### Fixed

- **Machines on a poor connection were losing reputation for downloads that
  were cut short.** When a model file failed its integrity check it was always
  treated as the sender's fault — the file was thrown away and that machine's
  reputation was lowered. But an integrity check cannot tell "these are the
  wrong contents" from "only part of it arrived": both simply fail to match. A
  machine with an unreliable connection was therefore penalised for our own
  interrupted downloads, over and over, because an unreliable connection
  interrupts repeatedly.

  The size the model's manifest declares is now checked before the contents
  are, so the two cases are told apart. A file of the wrong size is treated as
  an interrupted download: discarded, fetched again, nobody blamed. A file of
  the right size whose contents are still wrong is treated exactly as before.
  Checking the size first is also considerably faster than reading several
  hundred megabytes to reach the same conclusion.

## [0.3.50-alpha] — 2026-07-29

### Fixed

- **One abandoned request could freeze a model for everyone.** Sending a long
  request and then closing the window or killing the client left that request
  running to the end. It kept hold of the model the whole time, so every later
  request for that model waited behind work nobody wanted any more. On a long
  prompt this looked exactly like the node had frozen: no error, nothing in the
  logs, and no way back short of restarting it.

  Requests now stop when the connection to them is lost. Streaming replies are
  unaffected.

### Changed

- **This release is marked as the latest release rather than a pre-release.**
  Nodes older than v0.3.44 ask GitHub for "the latest release", which returns
  nothing at all while every release is marked pre-release — so those nodes
  were told they were up to date no matter how far behind they had fallen, with
  no way to discover otherwise. Publishing one normal release is the only thing
  that reaches them. Newer nodes are unaffected; they already list releases
  directly.

## [0.3.49-alpha] — 2026-07-29

### Fixed

- **Shared and single-machine answers now convert text the same way.** The
  previous release gave the decoder used for a machine's own answers the
  information it needed; the equivalent step used when work is spread across
  machines lives in a separate function and kept the older behaviour. They are
  now consistent.

  **This does not fix the stray `▁` mark some answers still contain.** That was
  the reason for this release and it turned out to have a different cause,
  found immediately afterwards: some words are converted into numbers
  incorrectly *before* the model ever sees them, so the model receives
  gibberish and says so. "banana" is one such word. It is not specific to shared
  work and it is not new — it behaved the same way several releases back, just
  with a differently mangled character. Tracked in docs/FUTURE_WORK.md.

## [0.3.48-alpha] — 2026-07-29

### Fixed

- **Answers could come back in the wrong shape, or answer a question you never
  asked.** Every model expects its question wrapped in its own particular
  format. When a machine co-ordinated a request for a model whose files it did
  not hold — which is the normal case once work is shared across the network —
  it could use the format of whichever model it happened to have loaded most
  recently, or fall back to a generic one that did not fit. Models answer in
  whatever format they are asked in, so the result was a reply that wandered off
  topic, contained stray markup such as `<|user|>`, or answered something else
  entirely. Reproduced with a one-word question that came back carrying another
  model family's markup and a question four times longer than the one asked.
  TinyLlama specifically is now recognised properly; it was among the worst
  affected and is one of the first models most people try.

- **Fetching only part of a model could leave it unusable.** Asking for this
  machine's share of a model divided the work by every peer it had ever seen
  rather than the ones actually connected, so it could take a small slice of a
  model nobody else was holding. Requests then failed with "No node available
  for layer 0" and the only way out was to fetch the whole model. With no peers
  online, a share is now simply the whole model. `get-model` also says plainly
  that a share depends on other machines holding the rest.

- **Settings you never chose now follow their defaults.** Improving a default
  had no effect on anyone who already had SwarmLLM installed: the whole
  configuration was written to disk on first run, and a value on disk always
  wins. Only new installs ever saw an improved default. The config file now
  records only what actually differs from the built-in values. This also fixes
  update checks that were stuck on a six-hour cycle after the default became
  hourly.

- **`swarmllm status` said a model was loaded while it was still downloading**,
  which read as "ready" and led to failed requests. It now also lists anything
  still arriving, with its progress.

- **Clearer errors.** Settings in a config file that SwarmLLM does not recognise
  — a typo, or a key under the wrong heading — are now named in the log instead
  of being silently ignored. And `swarmllm status` and `swarmllm peers` no
  longer answer with a bare authentication error when the daemon was started
  with a custom data directory: they name the file they read and how to point
  both at the same place.

## [0.3.47-alpha] — 2026-07-29

### Fixed

- **Small models sometimes replied with nothing at all.** Ask a small chat
  model a plain question and roughly two times in three it would answer with a
  completely blank message. Nothing reported an error — the reply arrived
  looking like a perfectly normal, successful answer that simply happened to be
  empty. TinyLlama, the model many people try first, was affected on almost
  every question.

  These models are trained to be given a short instruction about how to behave
  (a "system message") before your question. Chat apps normally send one, but
  ours did not, and without it the model would start writing your *next*
  question instead of answering the current one. That text gets removed before
  you see it, which is why the reply came out blank.

  SwarmLLM now supplies a neutral one when your app does not send its own. If
  your app does send one, yours is used unchanged. Models that do not want a
  system message are left exactly as they were.

- **Replies from most models should be a little better across the board.** Most
  models expect a marker at the very start of everything they read, and we were
  not adding it for the Llama family — which covers TinyLlama, Phi, Mistral and
  many popular community models. They still worked, but were being given a
  slightly unfamiliar starting point on every request. Gemma models were
  unaffected and are unchanged.

## [0.3.46-alpha] — 2026-07-29

### Fixed

- **Answers from your own machine had a stray character in place of every
  space.** Asking your machine directly could return something like
  `A▁distributed▁system▁is▁a▁network` — the marker a tokenizer uses internally
  to show where words begin, left in the finished text. Most of the words in the
  reply were affected, so it was not subtle when it happened.

  It only affected answers your machine produced for *you*. Work it did for
  other machines on the network was converted to text by a different route that
  was never affected, which is why it survived so long: every check that
  involved a second machine looked perfect.

  Whether it affected you depended on the model. It also explains a stray `<0x0A>`
  that turned up in a reply the previous day — the same cause, a different
  symptom.

- **A prompt that is too long is now reported as your request being too long.**
  It previously came back as an internal server error, which says the machine
  broke when in fact the one thing you can change is the length of what you
  sent. It also meant software that automatically retries after a server error
  would keep re-sending a request that could never succeed. The explanation
  itself was already right — it tells you the length, the limit, and what to
  change — it was simply filed under the wrong kind of failure.

- **Windows GPU builds are back.** v0.3.45 shipped without them: the graphics
  library needed at link time was not on the search path, and the step that
  fixes exactly this for the other GPU library had never been extended to cover
  it. The version of that library is now pinned as well, so a release cannot be
  broken by it changing on its own.

## [0.3.45-alpha] — 2026-07-28

### Fixed

- **Some models could not be downloaded at all, and pieces already held were
  deleted.** v0.3.44 started checking each piece of a model received from
  another machine against its expected fingerprint before passing it on. Where
  a model's listing carries no fingerprints, that check has nothing to compare
  against — but it treated the absence as a failure, so every piece was thrown
  away on arrival and the machine that sent it was penalised. Affected models
  could never finish downloading, and pieces already on disk were removed as
  they were re-checked.

  What you would have seen is a download that never progresses and eventually
  gives up, with nothing saying why. If a model disappeared from your machine
  after updating to v0.3.44, this is why, and it will download again on its own.

  Pieces are now checked when there is a fingerprint to check them against, and
  accepted otherwise. The protection added in v0.3.44 is unaffected: a machine
  serving bytes that do not match a fingerprint we hold is still refused and
  still loses trust.

- **The dashboard could stop updating its model badges.** Checking which cloud
  models are reachable shared a budget with changing your access key and
  installing updates — things you do deliberately and rarely, so the budget is
  deliberately small. The dashboard does this check by itself, and the budget
  counts per machine rather than per browser tab, so a few dashboards open at
  once used it up and the badges quietly stopped refreshing. It now has its own
  allowance.

## [0.3.44-alpha] — 2026-07-28

Security fixes from an external audit, plus updates that finish by themselves.

(v0.3.43 was withdrawn: it built without Windows binaries. Everything below was
in it, and this release adds the Windows fix.)

### Security

- **Being on a carrier-NAT address no longer counts as being on Tailscale.**
  Your machine hands its dashboard an access key automatically when you reach it
  over Tailscale. Deciding whether the machine was on Tailscale meant looking for
  an address in the range Tailscale uses — and that range is shared with the
  carrier-grade NAT that real internet providers hand out. A machine could
  therefore believe it was on Tailscale without ever having joined: some mobile
  networks address the device from that range, some providers number customer
  networks inside it, and a virtual or VPN adapter can land there by chance. In
  that state it would hand its key to any browser arriving from the same range,
  which on a provider's network can mean other customers.

  The machine now asks the Tailscale service running on it who a connection
  belongs to, instead of guessing from the address. That settles both halves: no
  Tailscale service means this machine is not on it, and an address Tailscale
  does not recognise means the visitor is not on yours. Where that service
  cannot be reached, a stricter check requiring evidence specific to Tailscale is
  used, and anything short of a clear yes asks for the key to be pasted as usual.

  Affects v0.3.41 and v0.3.42, and only machines that had a carrier-NAT address
  of their own. An ordinary home or office network was never in this state.

- **A model piece received from another machine is checked before it is shared
  on.** Pieces arriving over the network were saved, recorded as held, and
  offered to others without their contents being checked against the expected
  fingerprint — that check only came later, when a routine scan came round, up to
  five minutes on. A corrupted or forged piece could spread in the meantime.
  Pieces downloaded from HuggingFace were already checked; this was the one path
  that was not, and the one where the bytes come from a stranger. A piece that
  fails is discarded rather than announced, and the sender loses trust.

- **A machine cannot be made to set aside memory for data nobody sent.** Any
  peer that completed a connection could announce a quarter-gigabyte message in a
  five-byte header, and that much memory was reserved before any of it arrived —
  free for the sender, costly for the receiver. Memory is now committed as the
  data actually turns up.

- **The slice of a model a peer asks for is confirmed to exist.** This was
  already checked, but the check was untested and easy to lose in a future
  change; it is now a named rule with tests, covering ranges past the end of a
  model, empty and backwards ranges, and models claiming no layers at all.

### Added

- **Machines can now install updates themselves.** Updating used to leave the
  node running the old version, because replacing a file does not change a
  program that has already started — so it kept serving and reporting the
  previous build until somebody restarted it. Installing now waits for the
  machine to finish any work it is doing, including work it is doing for other
  people, and then restarts into the new version.

- **Machines find out that updates exist.** Checking was tied to the
  install-automatically setting, which is off by default, so a normal install
  never looked and never showed the notice. Checking is now separate and on,
  while installing automatically remains your choice — under Settings, with four
  levels from never check to install and restart when idle.

- Releases now carry the changelog for that version rather than only a link to
  the list of commits.

### Fixed

- Pre-release builds are included when checking, so "stable" no longer silently
  finds nothing on a project that has only ever published alpha releases.
- Installations managed by a package manager say so, instead of offering an
  install button that cannot work.
- The configuration file shipped at `/etc/swarmllm/default.toml` was never read.
  It is now copied into place when the package is first installed, and labelled
  as a template.
- Windows builds work again. The new Tailscale check talks to the local
  Tailscale service over a socket type Windows does not have, which stopped the
  Windows binaries being produced at all. Windows now reports that it could not
  ask — already treated as "no, ask for the key" rather than as a yes — and
  falls back to the same checks as any machine where that service is
  unreachable.

## [0.3.42-alpha] — 2026-07-28

### Fixed

- **Machines set up before 21 July could no longer find the network.** If you
  first configured SwarmLLM before that date, your machine starts with an empty
  peer list and never joins, however many times you restart or update it, and
  nothing says why. It looks like the newest release broke your networking.

  The starting point every fresh install dials to reach the network arrived on
  21 July. It only takes effect when your settings file does not mention
  starting points at all — but the app saves every setting to that file,
  including an empty list of them, which is what the setting was before the
  starting point existed. So anyone already set up by then had that empty list
  saved, and from then on it quietly overrode the new one.

  An empty list is now understood as "not set" rather than "none", so affected
  machines join again on the next start with nothing to edit. If you deliberately
  run with no starting points — a private or offline network of your own — say so
  with `disable_default_bootstrap = true` under `[network]`.

## [0.3.41-alpha] — 2026-07-28

### Fixed

- **The dashboard now works when you open it from another device.** Opening a
  machine's dashboard from your phone or laptop showed the page but left it
  unable to do anything: the setup wizard's "Start SwarmLLM" button appeared to
  do nothing, settings would not save, and the hardware panel could report "CPU
  only" on a machine with a graphics card. Every one of those was the same
  problem — the page is given its access key automatically only in certain
  situations, and opening it from elsewhere was not one of them, so each request
  it made was turned away.

  A machine that has joined a Tailscale network now serves a working dashboard
  to that network with nothing to configure. Anywhere else, the page explains
  itself and offers a box to paste your access key, which it then remembers for
  that machine. It also tells you the address the machine actually saw you
  arrive from — which, if you reach it through a router or a container, is not
  the address in your browser's address bar and was previously impossible to
  find out.

  There is a new setting, **Allow access from my local network** (Settings →
  Identity & Access, off by default), for reaching a machine through a Tailscale
  subnet router or a container. Those change the address your request appears to
  come from, so it arrives looking like any other device on the network. The
  setting takes effect immediately, without restarting the machine you are
  trying to reach.

  Please note what turning either of these on means: on a network the machine
  trusts, anything able to reach it can obtain its access key, and with that key
  it can control the machine and use it for inference. That is the intended
  trade for your own Tailscale network, whose devices you approved yourself, and
  it is why the local network option is off unless you choose it. Set
  `dashboard_trust_overlay = false` to decline even the Tailscale case.

- **Live updates work when the dashboard is opened from another device.**
  Counters, activity and peers only refreshed on a slow poll instead of
  arriving as they happened, because the live connection was refused for any
  address other than the machine itself.

- **Six panels were blank on every dashboard, on every machine.** Reference
  models, the wishlist, swarm capacity, the capacity plan, quant
  recommendations and the foreign pool catalog asked for their data before the
  page had its access key, so all six were turned away and quietly showed
  nothing — including on the machine's own screen, where everything else
  worked. The wait now happens in the one place every request passes through.

- **The API example in the README could not have worked.** It told you to fetch
  your access key with `curl`, which has returned an error since May. Your key
  is in the `api_key` file in SwarmLLM's data directory, and is shown under
  Settings → Access Token.

- **A setting documented as on by default shipped off.** New installs with no
  configuration file got the wrong value for the new Tailscale option and then
  wrote it into the file they generated, where it looked deliberate.

## [0.3.40-alpha] — 2026-07-28

### Fixed

- **A machine that only helps others no longer has its model shut down
  mid-answer.** A machine reclaims memory from a model nothing has asked for in
  a while, but the check for whether anyone was using it only counted requests
  the machine had started itself — not ones it was answering for other people.
  A machine doing nothing but contributing spare capacity therefore looked
  permanently unused, and after long enough its model was shut down while it was
  still answering, so whoever asked saw the reply stop partway through.

  That is the ordinary case for a machine helping the network, which is the
  point of joining it. Work done for others now counts as the model being in
  use, both while it runs and as the "last used" time afterwards.

- **A busy machine no longer stalls other people's requests.** Reading a long
  prompt is most of the wait, and it was done in fixed-size pieces counted in
  words rather than in time. The same number of words means very different
  things on different hardware — milliseconds on a graphics card, around a
  minute on a modest processor — so on a slower machine a second request
  arriving meanwhile advanced by only one word per turn for as long as the long
  prompt was still being read. Measured at eight words in five and a half
  minutes, which is indistinguishable from a stall.

  The piece size is now set from how long the last one actually took, so it
  suits whatever hardware it runs on. A request with nothing else running is
  never slowed, and on machines where smaller pieces turn out not to help, the
  pacing switches itself off rather than making things worse. Measured on the
  same prompt and machine before and after: on a processor, a co-scheduled
  request went from 470.5s to 48.9s and the long prompt itself from 490.9s to
  192.6s; on a graphics card, from 14.8s to 1.3s and 89.3s to 63.7s. Both
  requests got faster in every case.

### Added

- **A request that hasn't started answering yet now says what it is doing.**
  Loading a model and reading a long prompt can each take minutes with nothing
  reaching you, which looks the same as a machine that has stopped responding.
  Both now report their stage, how far along they are, and roughly how long is
  left — on the dashboard, in the admin API, in the logs, and as progress notes
  on streamed replies. The estimate appears once there is enough measurement to
  make it meaningful rather than guessing from the first moment.

## [0.3.39-alpha] — 2026-07-28

### Fixed

- **A node no longer stays broken after being updated.** Replacing the program
  file while a node is running left it unable to start any inference at all —
  every request failed with "spawn worker: No such file or directory". Because
  the node kept offering its model pieces to the network, other people's
  requests were still routed to it, so it failed those too. A node updated in
  place could sit failing everything until somebody restarted it.

  Linux marks a running program whose file has been replaced, and that marked
  name was being used as a real path. Installing an update and upgrading the
  package both replace the file this way. The self-update was affected too: it
  would have downloaded to a file named after the marker and left the real
  program untouched. The node now recognises the marker, keeps working, and
  says once that it should be restarted to finish switching over.

- **A node that shares part of a model no longer breaks its own chat.** A node
  holding a whole model *and* also serving a slice of that same model to peers
  could answer its own requests using the slice by mistake, failing with an
  internal error instead of a reply. Which copy got used came down to internal
  ordering, so a node could work for hours and begin failing only after it took
  on a second role — reported as a crash when serving a tail slice, and not
  reproducible from a clean start.

  Two checks disagreed: one asked whether any complete copy was present, the
  other then fetched whichever copy came first. Looking up the copy now requires
  it to be complete. Reported as two separate crashes, which turned out to be
  this one cause seen from either end.

- **Long questions no longer fail after five minutes.** Asking something with a
  long prompt could come back with nothing at all, even though the machine was
  working normally and would have answered. Reading a long prompt is most of the
  wait and gets slower the longer it is, so a few thousand words could exceed
  the five-minute cap every request was held to. Running a model now sits
  outside that cap — the limits that bound it already scale with the prompt,
  stop when you close the connection, and notice a client that vanished. The
  same flat cap was removed from two other places it did harm: passing a
  question to another machine to answer, and streaming a reply from a cloud
  provider, where a long answer was cut off mid-flow.

- **A slow connection can now finish an update or a model download.** Both gave
  up after a fixed stretch of time regardless of file size, which quietly set a
  minimum connection speed for using them at all — the graphics build is around
  933MB, so updating needed roughly 3MB a second sustained or it could never
  finish, failing at the same point every attempt with nothing to suggest speed
  was the reason. Both now watch for a download going quiet instead of counting
  total time.

- **A broken node no longer takes your request down with it.** If the node
  chosen to run part of a request could not start its model, the request failed
  there — even with other nodes holding the same model and ready to serve. Such
  a failure is now retried against a different node.

- **A clearer message when a node is asked for work it only holds part of a
  model for.** This used to fail deep in the maths with an unreadable error
  about tensor shapes. It now says plainly that the request needs the pipeline.

- **Ongoing conversations survive a restart when the clock has moved.** Saved
  conversation context was thrown away if the machine's clock went backwards at
  all between shutting down and starting up — which is ordinary: clocks correct
  themselves against the network at startup, and machines that were suspended or
  running as virtual machines resync on waking. Records saved a moment before
  then carry a time slightly in the future, and anything stamped in the future
  was treated as impossibly old rather than as the newest thing present, so every
  ongoing conversation was silently discarded and the next reply in each had to
  start over from nothing. A clock a single millisecond out was enough.

## [0.3.38-alpha] — 2026-07-27

### Fixed

- **Graphics memory is now actually reclaimed when a model goes idle.** A model
  left loaded on the card could stay there indefinitely despite the idle
  timeout — reported with two models still resident two hours and sixteen
  minutes after their last request, on a machine that then ran out of graphics
  memory.

  Unloading requires two things: no recent request, and no sign the wider
  network wants the model. The second had no upper bound, so any model the
  network was even mildly interested in was kept forever and the idle timeout
  never meant anything for it. Both reported models sat only just over that
  line.

  That second check exists for a reason — work done for a peer does not count as
  a request, only ones this machine makes do, so without it a machine could drop
  a model it was busy serving. But interest measured across the network says
  nothing about whether anyone is asking *this* machine. The reprieve now
  expires: past twelve times the configured idle window — one hour at the
  default — memory is reclaimed regardless. Long enough that a model in real use
  is never dropped mid-use, short enough that an unused one cannot hold a card
  all day.

## [0.3.37-alpha] — 2026-07-27

### Fixed

- **Choosing a model on the command line crashed.** `swarmllm chat --model <name>`
  ended in a panic instead of starting a chat, and had been doing so in released
  builds. The top-level `--model` takes a file path while the chat one takes a
  model name, and the argument parser objects when two options of the same name
  hold different kinds of value. Nothing about how you type the commands has
  changed.

- **A shard file of the wrong size hid forever.** If a piece of a model on disk
  disagreed with what the model said it should be, everything that loads models
  rejected it — but it still counted as a piece this machine held. So the model
  reported something missing on every check, nothing said why, and because the
  file kept the name it was never downloaded again. One such file was tracked
  across sixteen releases.

  The size is now reported alongside what was expected, the file is moved aside
  so the name is free, and the machine stops claiming it. The usual download then
  repairs it. The file is renamed rather than deleted.

- **Requests kept running after the client had gone.** Noticing a client had left
  relied on the connection closing properly, which does not happen if a machine
  loses power, a network drops, or a firewall quietly discards the connection. In
  those cases the work continued for nobody — measured at about six minutes of a
  machine's time. Connections now carry liveness probes, so a vanished client is
  noticed in about ninety seconds while a healthy one is never disturbed.

- **The strongest privacy mode could not start.** Prompt privacy keeps the first
  and last parts of a model on your own machine so no helper ever sees your
  prompt or the words chosen from it. It refused to run whenever the only
  available helper held a whole model, because a helper offers what it holds as
  one indivisible piece. It now uses part of what a helper holds, which privacy
  mode was always going to need — it splits the work by definition, so there is
  no faster single-machine route being given up.

- **Opening several dashboards at once broke provider status.** The allowance was
  counted per machine rather than per tab and was sized for one, so a third tab
  started being refused.

### Added

- **Prompt privacy is now on by default wherever it can work** — any model where
  this machine already holds the first and last piece. It cannot be on
  unconditionally, because without both pieces there is no route at all, so it
  applies only where the condition already holds. An explicit choice still wins,
  and `inference.encrypted_pipeline_auto = false` opts out.

- **One step to make prompt privacy possible.** A button on the privacy notice
  and `swarmllm privacy <model>` fetch exactly the pieces needed. Neither sets a
  switch: privacy turns on by itself once the pieces arrive, so there is no
  window where it is on but cannot run.

- **Log detail now reaches the model process.** Running with `-v` produced no
  extra detail from the process where models actually run, which made problems
  there hard to see.

### Changed

- **Clearer language about what encryption protects.** Traffic between machines
  is always encrypted; that does not mean the machine running the model cannot
  read your prompt — it has to, in order to answer, like any provider. The
  documentation said "end-to-end encrypted" for both that and the stronger
  prompt-privacy mode, so the same phrase meant two different things. Prompt
  privacy is now recommended where available, with its costs stated: disk for
  two pieces of the model, more work on your machine, and time that grows with
  the length of the answer.

- **The router prices routes on better information.** A machine it had never
  measured used to cost nothing, so on a freshly started node routes tied and
  the winner came down to internal ordering; it also never measured its own
  hardware, so local work looked free at any size. Both are fixed, and network
  cost is now counted per word for split routes rather than once.

## [0.3.36-alpha] — 2026-07-27

### Fixed

- **Opening the dashboard could stop its own live updates.** The ticket that
  authorises the dashboard's live connection shared a strict request budget with
  the checks that ask cloud providers whether they are reachable. A single page
  load spent that budget between them, and when the live connection lost the
  race it was refused — so the page stopped updating until it retried. Measured
  on 0.3.35, an ordinary authenticated page load had its live connection
  refused three times.

  Four causes, each of them the page working against itself:

  - The live-connection ticket is now budgeted separately. It is limited for a
    different reason from the provider checks — it costs nothing outside this
    machine — so sharing one budget was never right.
  - Provider health checks are budgeted separately too, sized to the
    every-30-seconds rate the page actually polls at. The page's own default
    behaviour previously exceeded the shared limit on its own.
  - Model availability was requested once per provider card, so nine providers
    meant nine separate requests. They are now collected and sent as one.
  - Neither is requested before the page has its access key. Both ran on every
    reconnection attempt, and a page that cannot connect retries continuously,
    which turned such a visit into a request storm.

  A page load now makes one provider request instead of a burst, and the
  server records no rate limiting at all.

- **A duplicated security policy in the page itself logged an error on every
  load.** One directive in it is ignored by browsers when written that way. The
  real policy is sent as a header and is unchanged; the ignored copy is gone.

## [0.3.35-alpha] — 2026-07-27

### Fixed

- **Long questions no longer fail after two minutes.** Asking something with a
  few paragraphs of context could come back as an inference error, even though
  the machine answering it was working normally and would have finished.

  The wait for the first word was a fixed budget, sized against how long
  *writing* an answer takes. It never accounted for *reading* the question,
  which grows with the question's length. On a six-core CPU machine a
  ~600-word question needs around five minutes just to read; the two-minute
  budget expired part-way through, the request retried, and hit the same wall.
  Short questions stayed inside the budget, which is why this only showed up
  once a question got long.

  The budget now grows with the question, and is sized using the model's own
  tokenizer where possible — a character-count rule of thumb tuned for English
  under-budgets Chinese and Japanese by more than half. There is still a
  ceiling, so an unresponsive machine is detected promptly. Short questions are
  timed exactly as before, and faster machines were never affected; this mainly
  decides whether modest CPU-only machines can answer long questions at all.

- **Two machines could stop seeing each other and never recover.** Both stayed
  running and both stayed connected to the same bootstrap node, yet neither made
  a single attempt to reach the other — observed lasting 17 minutes, and only
  cleared by restarting one of them. While it lasted, questions failed with "No
  node available", because the machine holding the rest of the model had become
  invisible.

  A dropped connection scheduled a reconnect only if the peer was mid-request, or
  had dropped before it was ever identified. A machine that was known and simply
  idle matched neither, so it was forgotten with nothing to bring it back —
  rediscovery relies on the other side announcing itself, which it only does when
  it restarts. A reconnect is now scheduled for that case too.

- **A retried request could take down both attempts and the model with it.**
  When a request was retried while the original was still running, the two
  attempts shared an identity that was assumed to be unique. The retry
  displaced the original's reply channel, and the original's cleanup then
  removed the retry's, so both failed. The path also discarded the running
  model, which was healthy throughout — so the next request paid a full model
  reload for no reason.

  It was reported as the model process closing the connection, which pointed at
  the wrong thing entirely. Attempts are now tracked individually: cleanup can
  only remove its own, a displaced attempt says it was superseded rather than
  blaming the model, and the model is left running.

- **Deleting or automatically pruning a shard could disrupt a request that was
  reading it.** Both checks asked whether one shard matched, but a machine's
  share of a model usually spans several, so every shard after the first was
  unprotected — and pulling one away mid-answer surfaces as an unrecoverable
  error. There is now a single place that answers which shards a running request
  actually reads, and all three paths use it. Diagnostics report the real span
  too, instead of always naming the first shard.

- **The router priced routing choices on the wrong signals.** Three separate
  problems, all of which meant it could pick on something other than merit:
  a machine it had never measured cost nothing, so on a freshly started node
  competing routes tied and the winner came down to internal ordering — and a
  machine nothing was known about outranked one already measured and liked;
  the node's *own* hardware was never measured at all, so its own work looked
  free at any size and it would keep layers locally rather than hand them to a
  faster peer; and network cost was counted once per hop, when a model split
  across machines actually exchanges data once per word generated.

  On the pair of machines this was measured on, routing is unchanged to slightly
  faster. It mainly affects swarms with several machines holding overlapping
  parts of a model, so that is where any change in behaviour will show up.

### Added

- **Router option to use part of a machine's share of a model**
  (`inference.parallax_partial_ranges`, **off by default**). Without it, a
  machine holding a complete model is the only route the router can express for
  it, so your own GPU cannot take the first layers while a peer takes the rest.
  It is off because it measured slower where it was tested — the per-word
  exchanges cost more than the faster hardware saved — and the reasoning, the
  numbers, and the one remaining fix are recorded on the option itself and in
  `docs/FUTURE_WORK.md`.

### Changed

- **Downloading a shard now writes as data arrives.** A download showed an
  empty file for around twenty-five seconds before anything appeared, then
  landed all at once, because each range was collected in memory first and
  written at the end. Progress now moves continuously and a large shard no
  longer has to fit in memory on the way through. The downloaded file is
  unchanged, byte for byte.

## [0.3.34-alpha] — 2026-07-26

### Fixed

- **A connection that had stopped working kept being chosen over one that
  worked.** Where several connections to the same machine exist, the most recent
  was preferred, on the reasoning that a connection which has quietly died is
  usually an old one. Observed to be wrong: a connection that had just carried
  three successful requests was passed over for a newer one that swallowed
  everything sent to it, with no error either way, until a ten-second timeout
  gave up.

  Selection now prefers the connection with the fewest requests still awaiting a
  reply. A connection that is answering clears them; one that has died only
  accumulates them, so the choice moves away from a dead path after the first
  failure instead of returning to it every time. Where nothing distinguishes two
  connections the newer is still preferred, which is what the previous rule
  existed for.

- **A retry could pick the same machine that had just failed.** Dropping a
  machine's stale claim to hold part of a model does not survive the retry,
  because the wider network still advertises it, so scheduling re-learns the
  claim and chooses it again. A machine that reports missing data is now barred
  from serving that request outright. The bar covers one request and is cleared
  with it — a single failure is not grounds for refusing a machine everywhere.

### Notes

- Verified against a machine whose claim was deliberately left stale: the request
  previously failed twice with the same error, and now fails once, with the retry
  correctly reporting that nobody can serve that part of the model.

## [0.3.33-alpha] — 2026-07-26

### Fixed

- **Two machines could stop being able to reach each other while appearing
  perfectly connected.** Requests left one side and never arrived at the other —
  no error, no delivery failure — until a ten-second timeout gave up, while a
  working direct connection sat unused the whole time.

  Connections are chosen by preferring a direct path over one that goes through
  a relay, and a relayed connection was recognised by a relay marker in its
  address. That only catches the case where *you* dialled out through the relay.
  When the other machine dials *you* through one, the address has no relay marker
  and no network address at all, just the machine's identity, because there is no
  direct connection to describe. Those were treated as direct, and being the most
  recent connection, they won every time.

  Most likely to affect anyone whose home router does not allow incoming
  connections, which is most people — that is exactly when peers reach you
  through a relay.

- **Diagnostics reported scheduling time that included a failed attempt.** A
  request that failed and retried charged the whole first attempt to
  "scheduling", so the logs showed thirteen seconds of scheduling for work that
  took under a millisecond. Since the troubleshooting guide reads a large
  scheduling time as "struggling to find a machine to serve the model", this
  pointed diagnosis in the wrong direction. The number is now the real assembly
  time, and the log says when a request was retried.

### Notes

- Verified between two machines on one network: the same request went from
  timing out after thirty-one seconds to answering in eight, with three
  consecutive runs averaging four and a half seconds.
- A machine reachable only through a relay stays reachable — the preference is
  for a direct path when one exists, never a refusal to use a relay.

## [0.3.32-alpha] — 2026-07-26

### Fixed

- **The first question you ask after starting a node no longer fails.** A node
  learns which peers hold which pieces of a model by listening for their
  announcements, and a full round of those only comes by every forty minutes or
  so. On a quiet network that left a freshly started node knowing of no one to
  ask, and the direct lookup that would have answered was started and then not
  waited for. The result was an error on the very first question, with the same
  question working seconds later. It now waits briefly for that answer.

  Reported by a tester who saw it after removing a model piece mid-session, and
  reproduced here after an ordinary restart. Anyone who tried SwarmLLM, hit an
  error on their first question and concluded it was broken was most likely
  seeing this.

### Notes

- Only the "nobody known to serve this" case waits. Every other scheduling
  problem still fails immediately, because waiting would add delay without
  changing the answer. If the wait finds nothing either, the original message is
  kept, since it names the part of the model that had no host.

## [0.3.31-alpha] — 2026-07-26

### Fixed

- **A peer that no longer has a model piece is no longer asked for it again.**
  Which peers hold which pieces is shared by gossip, so a node's picture can
  briefly outlive the truth — a peer that deleted or pruned a piece keeps being
  chosen until its correction arrives, and every request sent to it fails
  meanwhile. Now the first such failure drops that peer's claim locally and the
  request is retried immediately against someone else, so what used to be a
  string of failures is usually invisible. Reported by a tester who saw it while
  helping verify the previous release.
- **A reply served over the local network is no longer reported as relayed.**
  The route shown in the chat, the response headers and the logs treated any
  relay-*capable* peer as relayed, so answers that went straight out over the
  local network were labelled as taking the slower path — sending anyone reading
  it looking for a network problem that wasn't there.

### Notes

- Dropping a peer's claim is scoped to the pieces it was actually asked for. A
  request can span several pieces, and the earlier attempt at this dropped the
  wrong one — penalising a piece the peer genuinely had while leaving the bad
  claim in place. Caught by running it rather than reading it.
- A peer's own next announcement re-establishes whatever it really holds, so an
  over-cautious drop costs at most one announcement interval and never loses
  data.
- Also removes a rare test failure that could fail a CI run for reasons unrelated
  to the code under test.

## [0.3.30-alpha] — 2026-07-26

### Added

- **Every answer now says where it came from.** Chat shows "1.25s · 33.8 tok/s ·
  via 2 peers", with the peers and route on hover. Until now the dashboard could
  not tell you whether a reply came from your own machine, one peer, or a
  pipeline spanning the internet — which is the first thing anyone wants to know
  when a reply is slow.
- **A Performance view under Models.** Shows what your node has served for the
  swarm, every peer that has answered part of a request — ping, speed per layer,
  latency, region, slowest first — and your recent answers broken down by which
  machine did which layers. Available in all 21 languages.
- **Routing and timing on every API response.** Responses carry `x-swarm-route`,
  `x-swarm-peers`, `x-swarm-nodes` and `x-swarm-regions`, plus timings in the
  standard `Server-Timing` header that browser developer tools display natively.
  A misbehaving client can now be diagnosed with no access to the server at all.
- **One line per request in the log.** Every completed request writes a single
  `DIAG: request complete` line carrying the route, the peers, the queue and
  scheduling time, time to first token, per-segment timings, throughput and the
  outcome. Previously this had to be reconstructed from a dozen interleaved log
  lines across two machines, which is exactly what made recent bugs expensive.
  `docs/DIAGNOSTICS.md` now opens with that line and a table mapping each
  symptom in it to where to look next.
- **Diagnostics answers "why was that slow", not just "why did that break".**
  `GET /api/admin/diagnostics` gained recent requests, a per-peer performance
  table and what this node has served for others. A new
  `GET /api/admin/performance` returns the same data as JSON, plus an hourly
  trend that survives a restart.
- **Time to first token and time per output token are now measured.** They are
  the two figures that distinguish a backed-up queue from slow generation, and
  wall-clock time alone cannot separate them. Exposed on `/metrics` under the
  OpenTelemetry GenAI names, so collectors and community Grafana dashboards work
  without any translation layer.
- **Your node now counts what it contributes.** Segments and layers served for
  other people, time spent computing them and bytes returned. Every existing
  counter measured requests this node *made*; nothing measured what it *gave*,
  so "is my node actually helping?" had no answer.

### Fixed

- **Small models could not be served by a node holding only their last layers.**
  Llama-3.2 and most small models reuse their input word table as the final
  output layer, and that table lives in the model's first piece. A node holding
  only the last piece — the common case in a real swarm — failed while loading
  it, and with only one holder per piece the whole request failed. The file that
  exists precisely to carry that table was being written by three different code
  paths and read by none. Found by running a genuine two-machine split, and
  independently reproduced on different hardware. Models that ship a separate
  output layer, such as Llama-3.1-8B and Qwen2.5, were never affected.

### Changed

- Prometheus request counts are now labelled by route and outcome only. Both are
  fixed sets, so the number of series stays constant no matter how large the
  swarm grows; per-peer and per-model detail is served on request from the JSON
  endpoint instead, where it cannot accumulate.

### Notes

- On a streaming response the `Server-Timing` header carries only what is known
  before the first byte of the body — queue and scheduling time. Token-level
  figures arrive at the end of the stream. Headers cannot be revised once sent,
  and reporting a zero would be worse than reporting nothing.

## [0.3.29-alpha] — 2026-07-26

### Added

- **The bootstrap node is now labelled in the peer list.** It hosts no models
  and answers no questions by design, so it previously looked identical to a
  peer that was failing. Hovering explains what it does and why holding nothing
  is normal for it.
- **Peers show what they run on.** Each peer now carries a mark for its
  graphics card make, or a CPU tag when it has none — so it is clear at a glance
  whether a peer is fast hardware or a machine helping out with its processor.
  Previously a processor-only peer showed nothing at all, which looked the same
  as a peer we had not heard from yet.
- **Replies show speed, not just elapsed time.** Chat now reports tokens per
  second beside the response time. Elapsed time alone cannot be compared between
  a one-word answer and a long one, so it said little about how the swarm was
  performing.
- **The network map summarises the swarm.** Under the map you now see how many
  computers are taking part, how many have graphics cards, the combined memory
  and storage, and which models the swarm can actually run right now.

### Fixed

- **A negative credit balance now explains itself.** A new user goes below zero
  on their first question — they used the network before sharing anything — and
  saw a minus figure next to their tier with nothing to say what it meant. Every
  other credit message talks about earning, so it read as a debt or a penalty.
  It now says that this is normal, that nothing is restricted, and that the
  balance recovers on its own.

- **A tool call from a local model now carries the arguments the caller asked
  for.** Models were shown a tool's raw parameter schema and asked to fill in
  the arguments; some copied the schema's structure instead of the values,
  sending `{"properties": {"city": "Paris"}}` where `{"city": "Paris"}` was
  expected. The call looked entirely valid and reported success, so nothing
  raised an error — the program on the other end simply found the argument
  missing. Tools are now described by the shape of the arguments to send, and a
  reply that wraps its arguments in a schema is unwrapped when read.
- **A node that cannot start now says why.** Starting on a port another
  program is already using — the most common way a first run fails — printed a
  message that stopped at the colon with no explanation after it. It now names
  the port, says what is likely holding it, and suggests choosing another.
- **A malformed request to the tool endpoint gets a proper error reply.** An
  unreadable request was answered with plain text instead of the structured
  error the protocol defines, so a client saw neither an error code nor a
  message it could act on. Sending a wrong protocol version was also reported
  as unreadable JSON when the JSON was fine.
- **Tool servers that expect an older protocol revision can connect again.**
  The MCP endpoint answered every connection with its own newest revision
  whatever the client asked for, and a client that receives a revision it does
  not know is required to disconnect — so anything pinned to an older one was
  turned away, despite the tools on offer being identical.
- **Streaming replies containing a tool call now follow the Anthropic event
  order.** The reply's text section was left open while the tool section was
  opened and closed inside it, then closed afterwards. The specification is that
  each section is opened, filled and closed before the next begins, and a client
  tracking the current section could lose its place.
- **A tool call written without its outer wrapper is now understood.** Models
  often reply with the call on its own rather than inside the list we asked
  for. That was returned to the caller as raw text, so a perfectly good tool
  call looked like the model had ignored its tools. Output cut short by a
  length limit is still refused rather than guessed at.

## [0.3.28-alpha] — 2026-07-26

### Fixed

- **Llama 3 models now get the prompt format they were trained on.** Their
  chat template was failing to load, and the fallback used a different family's
  format. The model would go along with it and answer in that other format —
  which is where the stray `<|im_end|>` markers in replies were coming from, and
  why an answer was sometimes short or empty. Earlier releases removed those
  markers from the reply; this fixes the reason they were there. Affects every
  official Llama 3.x Instruct model file.
- **Naming a provider explicitly now works on the Anthropic API too.** v0.3.27
  fixed this for the OpenAI-compatible endpoint only, so `/v1/messages` with
  `deepseek:deepseek-v4-flash` was still rejected by the provider. The same
  gap also meant `anthropic:claude-...` was not recognised as an Anthropic
  request at all and took the wrong route.
- **Quantization is now actually reported correctly.** v0.3.27 fixed the
  reading of the tag but read it from the model's display name, which never
  contains one — so every model still reported `Q4KM`. It now reads the model
  id, where the tag lives.
- **Models no longer all report as unserveable.** The "can this be served right
  now" flag on the quantization view was never filled in and always read false,
  including for models the node was hosting itself.
- **A model whose chat template can't be read now falls back to its own
  family's format.** The fallback needs to know which model it is dealing with,
  and six of the seven places that build a prompt never passed that along — so
  the fallback had no choice but to assume a single format for everything, and
  models from other families were asked to reply in a format they were not
  trained on. Mistral models with a system prompt were hitting this, and any
  model whose template we cannot read would have. They now get Mistral, Llama 3,
  Gemma or LLaVA formatting as appropriate.
- **Saving a provider key after a daemon restart no longer fails.** With the
  dashboard left open, reconnection attempts ran every three seconds forever,
  which used up the request budget shared with saving API keys and checking for
  updates — so those kept reporting "slow down" for as long as the tab stayed
  open, and could not recover on their own. Reconnection now backs off, and
  resets once it succeeds.
- **Replies name the model that answered.** Asking the Anthropic-compatible
  endpoint for `provider:model` got a reply claiming to come from a model no
  provider offers, disagreeing with the OpenAI-compatible endpoint for the same
  request.

### Changed

- **Replies are finished in one place instead of three.** The in-process
  engine, the worker subprocess, and replies assembled from remote machines each
  cleaned up generated text themselves, in different orders and with different
  steps. That is why a stray control marker kept returning after each fix — the
  fix landed on one of them and the next reply came from another. They now share
  one finishing step, which also closed two gaps: the assembled-reply path
  cleaned up in the wrong order, so it could still return an empty answer when a
  model emitted a marker before its reply, and it never removed the blank lines
  a stripped marker leaves behind.

## [0.3.27-alpha] — 2026-07-26

### Fixed

- **Naming a provider explicitly now works.** Asking for a model as
  `deepseek:deepseek-v4-flash` failed: the prefix chooses which provider to use,
  but it was being passed along to the provider as part of the model name, and
  they rightly rejected it. It is now removed once it has done its job.
- **Saving provider keys can no longer fail quietly.** The fields are named
  `<provider>_key`, and anything unrecognised was discarded while parsing — so a
  request naming the field `mistral` instead of `mistral_key` saved nothing and
  still reported success. A request that would change nothing now says so and
  names the fields it expects.
- **Quantization is reported correctly.** Any model whose id carries a
  multi-part tag — `q8-0`, `q4-k-m` — was misreported as `Q4KM`, because ids use
  hyphens where filenames use underscores and only the last piece of the name
  was being read. A `Q8_0` model now says `Q8_0`.

## [0.3.26-alpha] — 2026-07-26

### Fixed

- **Stray control tokens no longer appear in replies.** Some model files emit
  their own end-of-turn markers into the answer text — sometimes malformed, with
  characters transposed or missing, and sometimes before the real answer rather
  than after it. Three previous releases each tried to fix this by adding more
  markers to a list of exact text to look for, which could never work: the marker
  arrives split across several pieces, and a mangled one matches nothing.

  Replies are now scrubbed of known control tokens in any spelling — complete,
  cut short, transposed, or with extra characters — and the scrubbing happens
  where replies are produced rather than where they are read, so it applies
  however a request is routed. A marker appearing *before* the answer no longer
  discards the answer with it, which had turned the leak into an empty reply.

  Only known control tokens are removed, so a reply containing your own
  angle-bracket construct is left alone.
- **Token counts on background responses.** A response created with
  `background: true` reported zero tokens used even after completing, while the
  same request run normally reported real numbers. The foreground path was fixed
  in the previous release; this one had the same gap.

### Changed

- The two API surfaces now share a single definition of the text that tells a
  local model how to request a tool. They previously held identical copies, and
  the wording has to match for tool calls to be recognised on both.

## [0.3.25-alpha] — 2026-07-25

### Fixed

- **Template markers no longer appear in replies from non-streamed requests.**
  The previous fix only covered streaming; the non-streaming path never applied
  the stop markers a model's own template defines, so a reply could run to the
  token limit emitting `<|im_end|>` and similar as visible text. Both paths now
  share one implementation.
- **Streamed responses report real token counts.** `/v1/responses` with
  `stream:true` returned zeros while the identical non-streaming request
  reported real numbers. The counts were always available; they simply were
  never requested, and the path serving locally-held models never sent them at
  all.
- **The chat page recovers on its own when a model is still loading.** The first
  message after switching models could arrive before the model finished
  starting, showing an error. Sending the same message again always worked, so
  the page now does that for you — once, so a genuinely broken model still
  reports a problem instead of retrying forever.

### Changed

- The responses API now caps the number of tools per request at 128, matching
  the other two API surfaces and what OpenAI allows. It previously had no limit
  at all, and every tool definition is added to the prompt and counted as input.

## [0.3.24-alpha] — 2026-07-25

### Fixed

- **Tool calls are recognised even when the model explains itself first.** A
  model rarely replies with nothing but the tool call — it usually adds a line
  before or after it. We required the whole reply to be the call and nothing
  else, so a perfectly good tool call came back as raw text the client couldn't
  act on. The call is now found wherever it sits in the reply. Output that was
  cut off partway is still left as text rather than guessed at.
- **Template markers no longer leak from models whose template disagrees with
  their training.** Some model files carry a chat template from one model family
  while the weights were tuned on another, so the model emits the *other*
  family's markers — which our previous fix couldn't catch, because it only
  looked at what the template itself contained. Markers that no model ever emits
  as real text now always end a reply. Markers that could legitimately appear in
  an answer, such as `[INST]` or `</s>`, still only apply when the model's own
  template uses them, so genuine replies about code or XML aren't cut short.

## [0.3.23-alpha] — 2026-07-25

### Fixed

- **Machines can reach each other again after one of them makes contact.**
  When another machine connected to you, it learned an address for you from the
  connection itself — but for an incoming connection that address is the far
  end's *temporary outgoing port*, which nothing listens on and which stops
  existing when that connection closes. We recorded those and published them, so
  every machine we contacted came away with several dead addresses for us and got
  "connection refused" on every later attempt to reach us.

  The knock-on effect was much larger than a failed dial. With no usable address,
  the two machines never form the relayed connection that NAT hole punching is
  coordinated over — so hole punching was never even attempted, the machine
  quietly vanished from the other's peer list, and requests that should have
  routed to it reported that no node had the data. An external tester hit exactly
  this: our announcements reached them, four dial attempts to four dead ports of
  ours all failed, and no session could ever form back.

  Only the address of a connection we started is recorded now. For an incoming
  connection we use what the peer advertises, filtered to addresses actually
  worth trying — and if it advertises none, we record nothing rather than a
  guess, because a wrong address fails every future attempt.

  **After updating, restart both machines** — addresses learned before this fix
  are still cached and still wrong.

### Changed

- Requests to a machine now use its newest direct connection rather than
  rotating across every connection to it. Permitting several connections per
  machine — which NAT hole punching requires — also allowed redundant ones to the
  same place, and rotating across those let a single half-dead connection swallow
  its share of requests.

## [0.3.22-alpha] — 2026-07-25

### Fixed

- **Tools now work with models running on your own machine.** Asking a local
  model to use a tool did nothing useful: on the OpenAI-compatible endpoint the
  model was told how to request one, did so correctly, and the request came back
  as ordinary chat text that no client could act on. On the Anthropic endpoint
  the tools were never mentioned to the model at all, so it replied that it had
  no way to call them. Both halves now work, including the formats different
  model families use natively rather than the one we ask for, and including
  streaming requests — which matters for agent tools like Claude Code, since
  those stream by default.
- **Chat-template markers no longer appear in replies.** Some models returned
  raw markers such as `<|im_end|>` or `<|eom_id|>` as visible text. Two causes:
  one of the three routes a request can take wasn't applying the stop markers a
  model's own template defines, and the list of markers we recognised was missing
  several that current models actually emit — including the one Llama 3.1+ uses
  when it believes it is calling a tool.
- **A failed reply now says why it failed.** When generation couldn't start —
  most often not enough GPU memory after switching models — the response ended
  empty and the dashboard could only guess, showing "the model might still be
  loading". The actual reason is now reported.
- **Downloading a model you already have no longer re-fetches it.** `get-model`
  on a fully-downloaded model re-downloaded every shard — around 353 MB of
  identical data in one report — because nothing checked what was already on
  disk.
- **A machine that fails to serve a request can no longer fail silently.** Such
  a machine was already careful to send back a reason, but if that reply was
  lost it went unrecorded: the requester saw only a timeout, while the server's
  own log showed a completed request. From the outside this was indistinguishable
  from a fault on the requesting side, and it made one tester's problem
  undiagnosable for several rounds.

### Added

- **Diagnostics now lists recent failed requests** — the model, how long it
  took, and which machine served it. That last detail separates "my node has a
  problem" from "one peer has a problem". When several failures share a peer, it
  says so directly. `GET /api/admin/diagnostics`.

### Changed

- The NAT traversal summary no longer reports a node as stuck on a relay when it
  has working direct connections and simply never needed to punch through one.
  It also notes that both machines need v0.3.21 or newer before a direct
  connection can form.

## [0.3.21-alpha] — 2026-07-25

### Fixed

- **Two machines behind home routers can now connect directly to each other.**
  Establishing a direct link between two such machines needs both of them to
  briefly hold two connections at once, but a limit set elsewhere in the code
  refused the second one — so the link was never completed and traffic kept
  going the long way round, through a middleman machine. Those middleman slots
  were then never released either, and once they ran out, a machine could drop
  out of reach entirely rather than simply falling back to the slower path.
  **Both machines need this version for a direct link to form.**
- **A machine reachable only through a middleman can now serve inference at
  all.** Until now it was skipped when choosing where to send work, so the
  relay could rescue a connection that was already working but could never
  make an out-of-reach machine usable — the situation it exists for. Such
  machines are now used, ranked behind directly reachable ones.
- **Requests no longer keep taking the slow path once a fast one exists.**
  Traffic to a machine was spread evenly across every connection to it,
  including the slower relayed one, so even after a direct link was available
  half the requests ignored it.

### Added

- **Publicly reachable machines now help relay traffic automatically.** Relay
  capacity previously came only from machines started with `--anchor`, which
  meant every pair of home users depended on a handful of them. Any machine
  confirmed reachable from the internet now contributes, so capacity grows with
  the network. Set `network.relay_forwarding_auto = false` to opt out.
- **Diagnostics now report NAT traversal.** `GET /api/admin/diagnostics` shows
  whether this machine is reachable from the internet, whether it is donating
  relay capacity, and how many direct-connection attempts have succeeded or
  failed. Previously none of this was reported anywhere, which is why the
  problem above went unnoticed for several releases. A machine that never
  escapes the relay now says so, and notes that this is expected on some home
  connections and that traffic still flows, one hop slower.
- **`network.max_connections_per_peer`** — an escape hatch, not a tuning knob.
  If dropped requests ever appear on a machine with several network interfaces,
  setting this to 1 restores the previous behaviour without a rebuild, which
  confirms or rules out that cause in one step. A node set below 2 warns that
  direct connections are disabled.

### Changed

- A relay now carries up to 128 simultaneous connections instead of 16. The old
  figure came from a default meant for connections lasting two minutes; ours
  last up to an hour, so slots were held far longer than that number assumed.

## [0.3.20-alpha] — 2026-07-25

### Fixed

- **The auto-updater works on NVIDIA/GPU builds again.** `swarmllm update` was
  refusing to install the GPU binary — the download hit a size limit (500 MB)
  that was set before the CUDA binary grew to about 1 GB, so the update aborted
  with "binary too large" and every NVIDIA user was stuck updating by hand. The
  limit is now 2 GB, with room to spare for future growth.
- **More reliable connections between machines on the same network.** On hosts
  with several network interfaces (common on WSL2 and Docker), two machines that
  discovered each other could each open several connections at once; the wrong
  one could quietly swallow distributed-inference traffic and stall a request.
  Now exactly one connection forms per peer, so split-across-machines inference
  on a LAN is steadier. Only affects local-network discovery.
- **The first relayed distributed request no longer needs a retry.** When two
  machines behind NAT jointly serve a model through the relay, the very first
  computed result could be dropped and the request retried; the node now trusts
  a peer that has just relayed to it, so the first request goes through.
- **One out-of-date pool member no longer freezes a whole pool.** If a single
  member's membership signature couldn't be verified (for example, they joined
  on an older build), the entire pool's state updates were discarded for
  everyone. Now that one member is skipped and the rest of the pool stays in
  sync.

### Changed

- Faster GPU release builds — the Windows GPU build cache is now kept warm, so
  releases stop rebuilding it from scratch each time.

## [0.3.19-alpha] — 2026-07-24

### Added

- **Inference across NAT now covers models split across machines.** v0.3.18 made
  a whole model on one un-reachable machine work through the relay; this extends
  it to *distributed* inference — where a model's pieces live on different
  machines behind NAT, none able to connect directly. The tensor traffic between
  them now flows over the same sealed relay (which never sees it) instead of a
  fragile fallback path, so two home nodes can jointly serve a model neither holds
  in full. Only nodes that both understand it use this path, so it changes nothing
  for older peers.

### Fixed

- **Settings save again.** Changing any setting — nickname, contribution level,
  bandwidth, anything — silently failed whenever auto-manage was on: the Save
  button stuck on "Saving…", the panel never closed, and nothing persisted. A
  leftover reference to a control removed in an earlier release threw an error
  that aborted the entire save before it started. Removed the dead reference;
  settings (including the node nickname) save correctly now.

## [0.3.18-alpha] — 2026-07-24

The networking release: two home machines behind NAT can now actually run
inference through each other, and the network won't break across versions again.

### Added

- **Inference completes between two machines that can't connect directly.** When
  two nodes are both behind NAT (home routers, mobile, CGNAT) and can't form a
  direct link, a request now routes through a mutually-reachable relay — usually
  the anchor, or any node that opts in. The relay is a blind pipe: it passes the
  traffic along but the request and the reply are sealed end-to-end, so it never
  sees your prompt or the generated tokens. This is the fix for "the two nodes
  connect and chat but inference never completes."
- **The relay survives losing any single middle-man.** Relay-capable nodes
  announce themselves on the network, and a node short on relay paths finds and
  connects to more on its own — so relaying keeps working even if the bootstrap
  anchor goes down, and spreads out as the network grows its own public relays.
- **Idle GPU memory is reclaimed automatically.** A model that's had no requests
  for 5 minutes, and that the wider network isn't asking for either, is unloaded
  from your GPU. Its files stay on disk and it reloads on the next request, so
  nothing is lost — your VRAM just stops being pinned by models nobody is using.
  Tunable (or disable) via `[auto_manage] idle_unload_secs`.

### Changed

- **The network no longer breaks across versions.** Nodes now negotiate which
  features they each speak, so a newer node never sends an older one something it
  can't understand — a node on one release keeps working with its neighbours on
  the next. New network features are additions from here on, never a hard cutover.

### Fixed

- **A machine you're getting an answer from stops the moment you walk away —
  even if you don't close the connection.** The previous release stopped work
  when a client disconnected; this one also handles a client that keeps the
  connection open but stops reading (a crash, a client-side timeout-and-retry, a
  closed laptop). Instead of a processor pegged for many minutes generating a
  reply nobody reads, the work now stops within a minute.

## [0.3.17-alpha] — 2026-07-24

### Fixed

- **Claude Code now connects on the first try.** Pointing a `claude` session at
  your node (`ANTHROPIC_BASE_URL=http://localhost:8800`) failed immediately with
  a "tool description too long" error, before any answer could be generated —
  Claude Code's built-in tools ship long instructions that ran past a size limit
  that was set too low. The limit has been raised so a standard Claude Code
  session works out of the box.
- **The model name in a reply now matches what you asked for.** On the
  OpenAI-compatible endpoint, a completion's `model` field could come back as a
  slightly different name than the one you requested (for example missing a
  `-fp16` suffix), which confused tools that route by that field. It now echoes
  exactly the model id you sent — the same way the Anthropic-compatible endpoint
  already did.
- **Local streaming replies stop the instant you disconnect.** The previous
  release made this instant for requests handled across the network; this one
  extends it to replies generated entirely on your own machine. Dropping the
  connection now stops the work right away instead of finishing a reply nobody
  is reading.
- **A brief network hiccup no longer cuts off an answer in progress.** When a
  machine you were getting an answer from had two network paths (a common
  setup), one of them closing could make it wrongly conclude you'd left and
  abandon the reply — even though the other path was still working. It now
  double-checks that the connection is truly gone before stopping.

## [0.3.16-alpha] — 2026-07-24

### Fixed

- **A backup-copy model can no longer be re-downloaded, shared, or shown
  anywhere.** Following on from the fix that stops copied-folder names
  (`…FULLBACKUP`) being registered: those names could still slip through other
  paths — a stale record could make the app re-download the copy from its
  original source, keep announcing itself as a provider for it, and show it in
  peers' hosted-model lists and the network map. All of those paths now reject
  the copied-folder name too, so it can't be acquired, served, reported, or
  displayed — closing the loop end to end.
- **A machine serving your request now stops the instant you disconnect, not up
  to ~30 seconds later.** When a request ran on another machine and you dropped
  off, that machine only noticed on its next attempt to send a word back — which,
  for a slow (processor-only) generation, could be tens of seconds of wasted
  work. It now detects the dropped connection immediately and stops right away,
  freeing that machine (and its graphics card) for others.
- **A leftover backup-copy model no longer keeps coming back after a restart.**
  When a copied model folder (`…FULLBACKUP`) had been recorded before the fix
  that rejects such names, it was still being reloaded from the local database
  on startup — so it lingered in your model list as a phantom (holding nothing).
  Those stale records are now dropped and cleaned out on load, and such names
  are filtered from the model list entirely, whatever their source.

### Added

- **You can now see which version each connected computer is running.** The
  peer list now reports every peer's SwarmLLM version and uptime (both already
  shared between nodes) — which makes it obvious at a glance when a peer is on an
  older build, a big help when diagnosing why something behaves differently
  across machines.

## [0.3.15-alpha] — 2026-07-23

### Fixed

- **A machine serving a request for someone who disconnects now stops
  immediately instead of working for nothing.** When you request inference and
  then drop off (closed the tab, lost your connection), the machine running it
  for you had no way to send the answer back and — if you were behind a home
  router — couldn't reconnect to you either, so it kept generating text nobody
  would ever receive. It now notices your connection is gone and stops the work
  (freeing that machine and its graphics card for others) the moment you leave.
- **Docker nodes no longer break routing by advertising an address that means
  something different on every machine.** A node running in Docker advertises its
  container's internal gateway (`172.17.0.1`) alongside its real address — but
  that gateway address isn't unique to one machine, it's whatever the *dialing*
  machine's Docker uses, so trying to reach a peer there quietly loops back to
  your own node. Peers are now reached at their real (public) address and these
  internal addresses are ignored, while a Docker node still connects and
  contributes normally through the address AutoNAT discovers for it. Nodes that
  only have a local-network address (the two-machines-at-home case) are
  unaffected.
- **The network map now places peers in their real country.** A node's location
  is detected from its address, but that detected value wasn't being attached to
  the info shared with other nodes — so the map fell back to showing everyone in
  the viewer's own country. Each node now reports its detected region, so a peer
  in Belgium shows in Belgium.
- **Inference to a slower peer no longer gives up after 10 seconds.** When your
  request ran on another machine that needed a moment to warm up — loading a
  model for the first time, a busy processor-only peer, or a long prompt — the
  first word can take longer than 10 seconds to come back. A safety timer meant
  for genuinely dropped requests was firing anyway, so a peer that had accepted
  the work and was busy on it looked like it had vanished. The timer now stops
  once the peer confirms it received the request, giving real work up to two
  minutes to produce its first word.
- **A request that runs a graphics card out of memory now finishes on the
  processor instead of coming back empty.** The first request to a peer whose
  GPU is too small already switches that model to the (slower) processor for the
  rest of the run — but the request that triggered it used to come back with
  nothing while the next one succeeded. That triggering request now retries on
  the processor automatically, so it returns an answer too.
- **`swarmllm update` now sees new pre-release builds.** While SwarmLLM is in
  alpha, every build is published as a pre-release, and a manual update check was
  filtering all of them out — so it always said "you're on the latest version"
  even when several versions behind. A check you run yourself now reports the
  newest build that's actually available.
- **The dashboard no longer calls a model "ready" when it can't actually be
  served.** Readiness was computed from whether *any* node had ever announced
  holding each shard — including peers that have since disconnected. That
  disagreed with how inference actually routes (only currently-connected holders
  count), so a model could show "ready" and then fail with "not enough peers have
  the shards." Readiness now uses the same reachable-holder test the scheduler
  does, so what the dashboard reports matches what can really run.
- **SwarmLLM under WSL2 "mirrored" networking now uses your real network instead
  of hiding on loopback.** On WSL2 the app applies conservative network settings
  because the default WSL networking sits behind an extra layer of address
  translation. But Windows 11's newer **mirrored** networking mode makes the
  Linux side a normal citizen of your home network — with a real address and
  working direct connections. The app didn't tell the two apart, so on mirrored
  mode it needlessly turned off QUIC and bound to loopback, forcing all traffic
  through a relay and leaving the node hard to reach. It now detects mirrored
  mode (via `wslinfo`, matching how Docker handles it) and keeps full networking
  on, so a WSL2 node joins and serves like any other machine.
- **A backup copy of a model can no longer spread through the swarm.** An earlier
  release stopped a node from *offering* a copied model folder
  (`my-model.FULLBACKUP`, `my-model.old`) to the network. But if another node on
  an older build still had one and gossiped it, everyone else accepted the name,
  counted it, and even started downloading it — a model no one could ever finish
  getting, because the name doesn't correspond to any real model. Copied-folder
  names are now refused wherever they arrive from — a peer, a saved record, or a
  local folder — so they can't be stored, counted, or passed on.
- **A home node behind a router now gets itself back online automatically after
  its relay drops.** Nodes that can't accept direct connections reach the swarm
  through a public relay. If that relay restarted (or the reservation simply
  expired), the node used to sit unreachable until it was restarted by hand —
  it never tried to re-establish the relay path. It now notices the relay
  circuit has gone and re-reserves within about a minute, with no intervention.
  Found live when an anchor restart mid-test stranded a test node.

### Added

- **Getting a shared test model is now one command.** `swarmllm get-model` lists
  the three shared test models (smoke / standard / stress); `swarmllm get-model
  standard` fetches this node's share of one, and `--all` grabs the whole thing.
  This works on a headless machine with no browser — handy for setting up a
  remote test node — and matches what the dashboard offers. The picker in
  Settings → "Testing & Diagnostics" is also open by default now instead of
  tucked away, so it's easier to find.

### Fixed

- **The Docker GPU image now works on more cards.** The last release widened
  NVIDIA support (RTX 20-series and up) in the downloadable binaries, but the
  `swarmllm:latest-cuda` container image was missed and stayed limited to
  RTX 30-series and newer. It now covers the same range.

## [0.3.13-alpha] — 2026-07-23

### Changed

- **GPU acceleration now works on many more NVIDIA cards.** The shared-inference
  engine previously needed an RTX 30-series or newer; anything older — including
  the very common **RTX 20-series and GTX 16 cards** — quietly fell back to the
  (much slower) processor. It now runs on those cards too, and on the new
  RTX 50-series, so a lot more people get real GPU speed out of the box. Local
  models also now build for each card generation from GTX 10-series up, including
  the RTX 50-series' fastest native code path (which needs the newer CUDA 12.8
  toolkit — research found the older compatibility path is up to ~5× slower on
  those cards).

## [0.3.12-alpha] — 2026-07-23

### Fixed

- **Testing a Moonshot (Kimi) or DeepSeek API key no longer wrongly fails.** When
  you pasted one of those keys and pressed "Test", the check tried a model name
  the provider had retired, so a perfectly good key came back looking invalid.
  It now tests against the current model, and the Moonshot "get a key" link
  points at the international site instead of the China-only one.
- **The prebuilt Linux download now runs on more systems.** People on Debian 12
  and other systems with an older core library had to compile SwarmLLM
  themselves, because the ready-made Linux build was made on a newer system than
  theirs and refused to start. It's now built against an older baseline, so it
  runs on Debian 12, Ubuntu 22.04 and later, and most current Linux systems with
  nothing to compile.
- **A computer penalised into a deep credit deficit now recovers on its own.**
  Credits set your priority and how much you can lean on other machines; they
  never stop your own use. A machine driven far below zero — by heavy borrowing,
  or by an earlier accounting bug — used to stay there indefinitely, stuck at the
  lowest priority and unable to share the pool's work. Its balance now drifts
  back toward zero over time, only ever upward and only while it is negative, so a
  one-off penalty is no longer permanent. Machines in good standing are
  unaffected.
- **A model you asked for could sit unusable behind one stuck download.** Models
  are fetched a piece at a time. When a piece stalled, the software immediately
  tried that exact same piece again, over and over — and while it was stuck, a
  different model you had just asked for couldn't start downloading at all, so it
  stayed incomplete and unusable no matter how long you waited. A piece that
  fails to download now waits before being retried, and waits longer each time it
  keeps failing, so one stuck download no longer holds up everything else. The
  model you asked for starts downloading right away.
- **Activity messages now show a device's name, not a code.** Several messages in
  the activity log (a device leaving, being removed from a pool, or sharing a
  model piece) showed a raw identifier even when you'd given the device a
  nickname. They now use the nickname wherever one is set.
- **The "local storage is full" notice now actually appears.** If your browser's
  local storage filled up while saving chat history, the warning meant to tell
  you was quietly dropped. It now shows.
- **One automated check kept rebuilding from scratch.** A step in the project's
  own build pipeline stored its cache under a name GitHub rejects (it contained a
  comma), so the cache was never reused and that check rebuilt everything every
  run. It now uses a valid name and reuses its cache. This affects only the
  project's automated checks, not anything you run.

## [0.3.11-alpha] — 2026-07-22

### Fixed

- **Computers on the public internet kept trying to reach home networks.** A
  machine remembers where it has seen others, and those addresses include the
  ones only reachable inside someone's house — their home network, and virtual
  network adapters created by other software. A server on the internet can
  never reach any of those, but retried all of them every minute regardless. It
  now only keeps trying addresses it could actually reach from where it is.

  Machines on a home network are unaffected and still remember each other's
  local addresses, which is how two computers in one house find each other
  again after a restart. A laptop that moves between networks keeps both sets,
  so nothing is lost by being on the wrong one when it last saved.

## [0.3.10-alpha] — 2026-07-22

### Fixed

- **The dashboard showed no readable text.** Every label, button and heading
  that comes from a translation file appeared as its internal name instead of
  words. A fault in the test-model panel added in 0.3.9 ran during startup and
  stopped everything after it, including the step that applies translations.
  That panel can no longer take the rest of the interface down with it.
  **Anyone on 0.3.9 should update.**
- **"No models available" now explains itself.** It used to say that and
  nothing more, which tells you something is wrong without saying what or what
  to do. It now says whether it is still looking for other computers, has found
  none, or has found some that are not sharing anything yet — and offers a way
  forward in each case rather than leaving you at a dead end.
- **A model said "Installed" when it was not.** The test-model list treated
  simply having heard of a model as having it, and a machine hears about a
  model as soon as any other machine mentions it. It now means what it says:
  this computer is actually storing part of it.
- **A backup copy of a model folder was offered to the network as a real
  model.** Copying a model folder — to `.FULLBACKUP`, `.old` or similar — made
  a model that other machines recorded, counted, and could never obtain,
  because its name came from the folder rather than the model. Such folders are
  now ignored, with a note in the log saying how to fix it if the model is
  genuine.

### Added

- **More in Copy diagnostics.** It now also reports the addresses this computer
  can be reached at, and how many remembered addresses are actually usable — the
  two things most often needed when working out why a machine cannot be found.

## [0.3.9-alpha] — 2026-07-22

### Added

- **A shared test model you can get in one click.** A new machine with no
  models, on a swarm that has nothing to offer yet, previously had nowhere to
  go — the chat screen could only list what was already out there. It now
  offers a small shared model instead, so there is something to actually do.

  The same model answers a request from people testing across several
  machines: if everyone runs the same one, speed results can be compared.
  Otherwise each person measures a slightly different version and the numbers
  were never alike to begin with.

  Three sizes are offered — a tiny one that just checks the network works, a
  recommended one that runs on almost any computer, and a large one for
  testing the limits of powerful machines. You can pick one from Settings, from
  the chat screen when nothing else is available, or tick a box during first-run
  setup. That box is off by default: it is a real download, and most people
  setting up want something to chat with rather than a test model. Test models
  are labelled wherever they appear so they are not mistaken for a
  recommendation.

  By default you get only the share your machine should hold rather than the
  whole thing, since the point is for the swarm to serve it together. Getting
  the whole model is offered separately.
- **Copy diagnostics.** A button in Settings that copies a short summary of
  your node — version, connections, models, recent events — ready to paste when
  reporting a problem. It contains no keys, invite codes or file paths.

### Fixed

- **Release builds were slow again.** The container image build was filling
  GitHub's shared build-cache allowance and pushing out the cache the Linux GPU
  build depends on, which took that build from about 10 minutes back to nearly
  an hour. Image layers are now cached elsewhere, and the two image builds no
  longer overwrite each other's cache.

_(0.3.8-alpha was tagged and withdrawn before publishing — a mistake in the
build-cache change above stopped the container images from building. Nothing
was released under that version.)_

## [0.3.7-alpha] — 2026-07-22

### Fixed

- **A machine could keep being sent work it was no longer able to do.** When a
  computer stopped hosting part of a model — you deleted it, storage cleanup
  removed it, or automatic management moved it elsewhere — other machines were
  never told. They carried on treating it as a place to send that part of the
  work, and requests routed there timed out or failed with "no standby
  available". Waiting did not help, because there was no mechanism by which the
  news could ever arrive. Machines now say which models they are giving you the
  complete picture for, and anything left out is dropped. This also repairs
  "remove model", which has always tried to announce that it no longer hosts
  anything and was never heard.

  Both machines need this version for it to work between them. Mixed versions
  keep the old behaviour rather than breaking, so there is no need to upgrade
  everything at once.
- **Requests were charged for tokens they never used.** A request sets aside
  credits up front based on the *most* it could generate, and that whole amount
  was kept even if the answer came back in one word. At the default settings
  that is 20,480 credits for every request regardless of length — one operator
  reported a balance of -41,400 after a handful of attempts. Requests are now
  charged what they actually used, and the difference is returned. This matches
  how requests below the reservation threshold were always charged, so the two
  no longer disagree.

## [0.3.6-alpha] — 2026-07-22

### Fixed

- **A GPU node could auto-update itself into a CPU-only build.** The updater
  matched a release asset on operating system and processor only, and the only
  bare binaries published were the CPU ones — so a machine running the CUDA or
  Windows-GPU build downloaded the CPU binary and installed it over itself,
  silently losing GPU acceleration. Every variant now publishes its own binary,
  and the updater will only install one built the same way as the one running.
  If nothing matches, it reports no update rather than installing something
  else. Auto-update is off by default, so only nodes that opted in were
  affected.

  Upgrading a GPU node that already has auto-update enabled: one more update
  may still fetch the CPU build, because the *currently installed* binary is
  the one choosing. Reinstall the GPU archive once and it will track the right
  variant from then on.
- **Nodes kept re-dialling addresses that could never work.** Remembered peer
  addresses were stored exactly as each peer advertised them, including their
  loopback and private network addresses. A node on the public internet then
  retried those every minute forever — and a relay node could end up trying to
  reach a peer by relaying through itself. Remembered addresses are now checked
  before being stored *and* before being dialled, so an existing bad list is
  cleaned up on the next start rather than persisting. Local network addresses
  are still kept, since that is how two machines in one home find each other
  again after a reboot. The per-address log line also moved to debug level; an
  idle node was writing one line per remembered address every minute.
- **A model whose chat template failed to load could be sent an empty prompt.**
  Several kinds of broken template produced no text at all instead of
  reporting an error, and the empty result was used as the prompt — so the
  model received none of the conversation. For image models this also dropped
  the marker saying where the picture belonged. Broken templates are now
  detected and fall back to a sensible format for the model.
- **Vision models with newer image-encoder files failed to load.** Support for
  the two possible layouts is now detected per file instead of assumed.
- **A machine that corrupted shared calculation data was never marked down for
  it.** When several machines split one model layer between them, they combine
  partial results; a machine sending corrupted numbers would spoil the answer
  for everyone in the group, but the fault was recorded as a local problem, so
  its reputation was untouched. It now counts against the machine responsible.
  Cases where the culprit genuinely cannot be identified — a group member
  simply being slow, which might be your own machine — still count against
  nobody, deliberately. Only affects setups that split a single layer across
  machines, which is off by default.

## [0.3.5-alpha] — 2026-07-22

**Five externally-reported bugs fixed (R146) + request cancellation and
deferral cleanup (R147).** All five bugs came from a user running 0.3.4-alpha
on two home machines with an 8B model deliberately split across them. Every fix
below was verified on real GPU hardware, not just unit-tested.

> ### ⚠️ Breaking change — device pools
>
> The pool acceptance signature now covers the invitation's expiry, which is a
> wire-format change. **A 0.3.5 node and a 0.3.4 node will reject each other's
> pool member lists.** If you run a pool, upgrade every device in it together.
> Nodes not using pools are unaffected, and normal inference, discovery and
> credits interoperate across versions as before.

### Fixed

- **A tiny model replicated on two LAN machines could fail outright.** When the
  local node already held every layer it still pulled a LAN peer into a
  tensor-parallel group; if that peer went quiet the whole request died with
  `AllReduce timeout after 10s for layer 0`, on a node that could have answered
  alone. Full local coverage now never forms a group, tensor parallelism is
  opt-in (`inference.tensor_parallel`, default off — over Ethernet the two
  round trips per layer cost more than the compute they split), and a failed TP
  segment now falls back to local compute instead of killing the request.
- **A worker that hit a GPU out-of-memory error kept its VRAM forever.** Only an
  explicit unload ever killed a worker, so a failed request left the process
  resident holding its whole allocation — 4.4 GB in the reported case, still
  there minutes later. Each retry then had less memory than the last, so one
  OOM reliably became permanent failure for that model. Fatal device errors now
  recycle the worker. Measured on an RTX 3070: **VRAM 7951 MB → 120 MB in six
  seconds**, and the retry then succeeds.
- **`inference.gpu_layers` did nothing for sharded models.** It was read by the
  legacy llama.cpp path only; the engine that actually serves inference chose
  its device unconditionally. Setting it to 0, 8 or 20 produced an identical
  allocation. Worse, the shipped default was `0` documented as "CPU only" while
  every CUDA build used the GPU regardless. It is now honoured end to end.
  **The default changes from `0` to `-1` (auto)** so existing GPU nodes keep
  using their GPU; `0` now genuinely means CPU-only, and a node configured that
  way with a GPU present says so loudly at startup. Partial offload is not
  supported by this engine and now warns instead of silently ignoring the value.
- **A GPU OOM now falls back to CPU** for that model for the rest of the run,
  rather than repeatedly retrying the allocation that just failed.
- **Failed requests no longer charge credits for our bugs.** Every failure
  applied a flat −50 penalty, including failures the local node caused —
  debugging the three bugs above drove the reporter's own balance from 0 to
  −470 with no peer ever misbehaving. A penalty now requires that a remote peer
  was actually involved and that the error is one they could have caused.
- **Peer credit balances are no longer invented.** With no gossiped balance the
  leaderboard displayed `trust_score × 5000` — at the default trust of 0.5,
  exactly "+2500 credits" for every unknown peer. This was reported as a ledger
  inconsistency; it was never a ledger figure. Unknown balances now render as
  "—".
- **The dashboard VRAM gauge showed an estimate as if it were live usage.** It
  summed each loaded model's estimated footprint and displayed that instead of
  the real figure, with a clarifying tooltip only when real usage *exceeded* the
  estimate. An idle machine read "5.3 GB / 5.7 GB — 93%" against a real ~1 GB.
  Live usage is now always the headline number; the estimate moved to the
  tooltip.
- **A client that hangs up mid-response now stops the work.** Closing the tab
  left the worker generating its full token budget into a channel nobody was
  reading — measured at 754% CPU thirty seconds after the client died. Now falls
  to 0%.

### Added

- Request cancellation between daemon and worker, covering client disconnects,
  timeouts, and hedge losers. Remote peers are told to stop too, so an enabled
  hedge no longer leaves the losing node computing a result that will be thrown
  away.
- `inference.tensor_parallel` (default `false`).

### Security

- Pool acceptance signatures now bind the invitation's expiry, so a captured
  acceptance is implicitly time-bounded and cannot be transplanted onto a
  different invitation. Expired invitations are also now rejected outright. See
  the breaking-change note above.

### Documentation

- The config reference was missing **45** `[inference]` options; all are now
  documented and grouped (batching, prefix cache, speculative decoding,
  hedging/prefetch, activation transfer). 91 → 133 documented options.
- `swarmpool://` invite codes documented, including why a NAT'd node needs its
  external addresses in the payload.
- Python SDK gained `pool.generate_code()` and `pool.join()`.

1099 → 1124 lib tests, 75 integration tests.

## [0.3.4-alpha] — 2026-07-21

**Cloud model & provider currency refresh + audit sweep (R145).** The cloud
surface hadn't been refreshed in ~2 months. No routing-architecture changes —
providers that fetch `/models` dynamically and route by prefix pick up new
models automatically; only hardcoded lists, aliases, and one base URL were
stale.

- **Claude lineup** — Opus 4.7 → **4.8**, Sonnet 4.6 → **5**, added **Fable 5**
  (Haiku 4.5 unchanged) across the model picker, subscription list, and the
  bare-alias resolver (`opus`/`sonnet`/`haiku`/`fable`). The `claude_subscription`
  default now matches Claude Code 2.1's new default (Sonnet 5). `anthropic-version`
  header confirmed current.
- **Claude Code 2.1.215** — verified every subprocess CLI flag is still valid
  (no breaking removals); added the `manual` permission-mode alias.
- **Moonshot/Kimi** — base URL switched from the China-only `api.moonshot.cn`
  to the international `api.moonshot.ai`; refreshed the model list to Kimi K3 /
  K2.7 Code / K2.6 / K2.5 (the K2-0527 and Moonshot-v1 models were discontinued).
- **DeepSeek** — probe/examples updated to `deepseek-v4-flash` ahead of the
  `deepseek-chat`/`deepseek-reasoner` legacy-name retirement (2026-07-24).
- **Mistral** — added `magistral`/`ministral` prefix routing. **OpenAI** needed
  no change (GPT-5.x and o-series already route correctly).
- **Security** — bumped `anyhow` 1.0.102 → 1.0.104 and `memmap2` 0.9.10 → 0.9.11,
  clearing two RustSec "unsound" advisories surfaced by `cargo audit`.
- Docs refreshed (README, book, config examples). 1099 lib tests, no regressions.

## [0.3.3-alpha] — 2026-07-21

**Dashboard peer-clarity + reachability docs (R144).** Follow-on to 0.3.1/0.3.2,
driven by the first external user testing live — their dashboard called the
remote bootstrap anchor a "LAN" peer.

- **Fixed LAN misclassification** — a peer was tagged LAN if any of its
  *advertised* addresses was private/loopback, but a public `0.0.0.0`-bound node
  also advertises `127.0.0.1`, so every remote peer (including the anchor) was
  mislabeled. Now classified only on the actual connection address + the peer's
  observed-us address.
- **Clear peer typing everywhere** — every peer is tagged **Pool / LAN /
  Internet** (green / purple / blue). The header reads "N internet peers / N on
  your network / N pool devices" instead of an ambiguous "peers / lan". The
  backend exposes a mutually-exclusive taxonomy (pool + lan + remote == connected
  peers).
- **Version in the header** — next to the SwarmLLM logo.
- **Honest empty state** — no longer says "Connecting to the network…" when you
  are already connected with no shared models yet.
- **Swarm-resources strip** — computers online (incl. yours), GPU machines,
  combined VRAM, shared storage, regions — "how big is the swarm actually?".
- **Docs** — README promotes out-of-the-box auto-join (default anchor + UPnP +
  AutoNAT v2 + relay) + Discord; `docs/NETWORKING.md` gained an explicit
  AutoNAT-v2 note.
- i18n: 14 new keys × 21 locales. 1097 → 1099 lib tests.

## [0.3.2-alpha] — 2026-07-21

**Fix: `/dns4` bootstrap peers were undialable.** The swarm transport was
missing the DNS-resolution wrapper (`.with_dns()`), so any DNS-named multiaddr —
including the default `swarmllm.duckdns.org` bootstrap anchor added in 0.3.1 —
failed with "Multiaddr is not supported". Result: fresh installs couldn't
auto-join. Added `.with_dns()`; the default bootstrap also gained an `/ip4`
fallback for hosts where DNS resolution is unavailable. (Caught by live
multi-node validation — the unit test only checked the address string was
present, not that it was dialable.)

## [0.3.1-alpha] — 2026-07-21

**Internet reachability & NAT traversal (R143).** Makes SwarmLLM reachable
across the internet, not just the LAN — the biggest gap for real-world use.

- **Default bootstrap anchor** — fresh installs now dial a publicly-reachable
  seed node (`swarmllm.duckdns.org`) on startup and **auto-join the network
  with zero config**, then decentralized discovery (DHT/PEX) takes over. Set
  an explicit `bootstrap_peers` to override (empty list opts out).
- **UPnP** automatic gateway port-mapping (default on) — zero-config internet
  reachability on cooperative home routers.
- **Invite codes now carry a public address** — `refresh_listen_multiaddrs`
  unions confirmed external addresses (UPnP / AutoNAT / relay / manual) with the
  bound listeners, closing the gap where a NAT'd node minted LAN-only codes.
- **AutoNAT v1 → v2** — v1 falsely reported NAT'd nodes as "Public" over QUIC
  (so they never reserved a relay and stayed unreachable); v2 tests each address
  for real reachability. Plus a belt-and-suspenders relay fallback.
- **`network.external_addresses`** — declare a reachable address (or a list, to
  cover TCP + QUIC) for a port-forwarded box / VPS / dyndns anchor.
- **`--anchor` mode** + a hardened `deploy/anchor/` kit (sandboxed systemd unit,
  non-root user, SHA256-verified binary, firewall, DuckDNS updater) for running
  a public bootstrap/relay node. See `docs/NETWORKING.md`.
- **Security**: quinn-proto → 0.11.15 (RUSTSEC-2026-0185, remote QUIC memory
  exhaustion, HIGH), crossbeam-epoch → 0.9.20 (RUSTSEC-2026-0204); installer
  input validation; relay abuse limits reviewed.
- **Auto-updater tracks pre-releases** — the updater listed `/releases/latest`,
  which skips pre-releases, so alpha nodes never auto-updated. It now lists
  `/releases` and selects by mode: `all` tracks alpha/beta (stays patched during
  the alpha phase), `stable`/`disabled` track stable only. Anchors default to
  `all`.

1075 → 1097 lib tests. Known gap: the relay/DCUtR CGNAT path is wired but awaits
live multi-NAT validation.

## [0.3.15-alpha] — 2026-07-23 — post-v0.1.0

Working changelog for commits after the v0.1.0 tag. Will roll into the
next tagged release.

### R142 — Autonomous 8-hour sweep (2026-05-22 → 2026-05-23)

14 sweep rounds dispatched as a self-paced `/loop /sweep`, 15 commits
to `main`, 60+ findings closed. Standout: **3 silent production bugs**
from frontend↔backend JSON wire-format drift — bugs no test caught
because the broken path emits no error:

1. R141 chat empty-state catalog never rendered (WS `stats_update`
   never merged into `App.data.cache.stats` → `cache.stats.wishlist`
   always undefined).
2. R140 maturity-fade button stuck prominent forever (`pool.js` read
   `peer_count` but stats field is `peers`).
3. Auto-manage activity orb permanently zero (`auto-manage-status.js`
   matched PascalCase `'Downloading'`; backend serializes
   `"downloading"`).

Real concurrency bugs: 3 TOCTOU fixed with atomic `remove_if`
(`try_assemble_chunked_forward` double-dispatch, `remove_shard_holder`
holder-drop race, `evict_split_models_lru` cache drift), 2 clock-
dependence bugs (`maybe_reset_window` froze on backward NTP
correction in both hedging + prefetch), atomic ordering on Release-
paired counters, batch `active_count.fetch_add` outside spawn
closure (R103-class leak), `register_multi_turn` orphan
`KvCacheSession`, pipeline-stream `send_forward` orphan on caller-
future cancel, batch panic arm missed `active_pipelines` cleanup.

Security / correctness: HF download size cap bypassed when
`Content-Length` absent (DoS), `auto_switch_quants` Default vs
serde-default mismatch, 4× wrong `SwarmError::Internal` for worker-
died (should be `ServiceUnavailable`), 2 discarded errors that
silently closed SSE streams, `anthropic_split_stream` dropped
`matched_stop_sequence`. Scheduler oracle violation in
`allocate_offline` + mmproj prune missing the same liveness filter.

Hot-path perf: per-token `format!()` in info-level tracing,
`vec![forward.clone()]` deep-copying activation buffer on persistent-
stream non-chunked path.

11 helper extractions, 19 new tests pinning invariants, 4
`.claude/rules/architecture.md` doc drifts fixed, DIAGNOSTICS.md
DIAG strings synced, book introduction + config reference updated.

Deferred to `docs/FUTURE_WORK.md § R142 deferred items`: VLM
`ffn_up/down` weight inversion (needs LLaVA integration test),
LLaVA chat-template fallback edge case, Python SDK R140 endpoints,
test-binary `spawn_test_server` extraction, streaming + invite v2
config-reference doc rows, `apply_update_with_version` Option
cleanup, worker compute waste on cancel.

1056 → 1075 lib tests. Clippy clean default + features
dev,claude-subscription + features llama. Commits:
`05233184..dfcfaa8d`. Detail: `memory/round_log_R142.md`.

### R141 — Auto-manage cold-start UX (2026-05-21)

Closed the "fresh node has nothing to chat with" gap by removing every
silent gate that blocked auto-manage from acting and surfacing what
the swarm already runs directly in the chat empty state. Coordinated
six changes:

- **Trusted-publisher allowlist** in `src/model/huggingface/watcher.rs`.
  `TRUSTED_HF_PUBLISHERS` covers official model authors (meta-llama,
  mistralai, Qwen, google, microsoft, deepseek-ai, HuggingFaceH4,
  stabilityai, tiiuae, 01-ai, NousResearch, allenai, ibm-granite,
  CohereForAI) + curator community (bartowski, TheBloke, unsloth,
  lmstudio-community, MaziyarPanahi, QuantFactory, second-state).
  Models from trusted publishers promote to `DemandVerified` at 10k
  HF downloads instead of 100k — fresh releases from known curators
  reach the swarm in hours not weeks. 24h age gate unchanged.
- **Wishlist `Candidate` status** in `src/model/auto_manage/wishlist.rs`.
  `compute_wishlist` now merges HfTrending entries the swarm hasn't
  adopted (cap 24) as `Candidate` rows with new `hf_repo_id` +
  `task_tags` fields. Frontend renders these with a "Set this up"
  CTA that opens the HF browse pre-filtered to the repo — user
  picks the quant variant, no auto-pick. +10 score bonus for trusted
  publishers.
- **`auto_switch_quants` default → `true`** in `src/config/inference.rs`.
  Recommendation surface that required a button-click became automatic.
  Trust + prune cooldown still guard bandwidth cost. Operators on
  metered links can flip back off.
- **`P2P_PERMIT_STALL_SECS` 600 → 180** in `model/auto_manage/manager.rs`.
  Silent libp2p drops fail over to HF fallback in 3 min, not 10.
- **`activity.hf_sources_cap_reached` activity event** in
  `daemon/dispatch/mod.rs`. Throttled 1st + every 50th drop, warning
  toast pointing the user at Settings cleanup. Previously silent
  `tracing::warn!`-only.
- **Chat empty state swarm catalog** — `createEmptyState` in
  `frontend/js/core/utils.js` builds a `buildSwarmCatalog()` block
  when no model is selected. Three rows: Serveable (one-click select),
  Aspirational (gathering), Candidate (route to HF browse). Chip click
  handlers route through `App.models.selectDropdown` +
  `App.chat.newSession` so the user lands in a fresh chat with the
  model loaded in one click. Re-rendered on every `stats_update` so
  the catalog comes alive within ~2s of daemon start.

i18n: 15 new keys × 21 locales (1156 → 1172 entries per locale) —
idiomatic translations. Tests: +5 watcher + +3 wishlist. 1048 → 1053
lib tests (R140 had brought it to 1048). Clippy clean default + features
dev,claude-subscription + features llama. Commit: `50225f7c`.

### R140 — Pool invite codes v2 (2026-05-19)

The 8-character pool invite code (`A3F7K2M9`) worked only when both
nodes were already on the same libp2p swarm — useful in a mature
decentralized network, useless for helping two fresh nodes find each
other before decentralization is achieved. R140 closed that gap.

New `swarmpool://...` blob (`src/pool/invite.rs`) wraps the existing
8-char code with the inviter's reachable listen multiaddrs. Encoded as
JSON → ChaCha20-Poly1305 (random embedded key, anti-IP-harvesting
only) → base64url. ~300-500 chars, fits in a copy-paste. Inner
payload: `{ version, pool_id, pool_name, multiaddrs[], code (8-char),
expires_at_unix }`.

- New `SharedState.listen_multiaddrs: ArcSwap<Vec<String>>` — live
  snapshot rebuilt by NetworkManager on `NewListenAddr` /
  `ExpiredListenAddr` / `ListenerClosed` / `ExternalAddrConfirmed`.
  Filtered via `addr_is_remotely_reachable` that drops loopback,
  unspecified, link-local, and AWS IMDS but keeps Tailscale CGN
  (100.64.0.0/10) — the WAN-bootstrap use case needs that range.
- `handle_join_with_code` dual-mode: v2 blob → dial each multiaddr
  via `NetworkCommand::DialAddress` then broadcast existing
  `PoolMessage::JoinRequest`. Legacy 8-char → direct broadcast
  preserves on-swarm flow. Wire protocol unchanged.
- Generation rejects empty addresses: if `listen_multiaddrs` is empty
  (daemon hasn't bound yet), `handle_generate_invite_code` returns
  `ServiceUnavailable` instead of handing out a useless code.
- Frontend: dropped the fake-QR pattern (only hashed 8 chars, was
  never scannable — misleading). Monospace code box + Copy button
  sized for ~500-char v2 blob. Paste field upgraded from `<input
  maxlength=8>` to `<textarea>`. Join handler sniffs prefix to route.
- Maturity-fade UI: while local node sees <50 swarm peers, "Add
  Another Device" sits in the dashboard header. ≥50 peers demotes it
  to Settings — swarm is mature enough that DHT discovery is reliable.

5 i18n strings refreshed × 21 locales + 1 dead key removed (`pool.
scan_or_type` — for the fake QR). 18 new tests (codec roundtrip,
tamper, expiry, version, truncated/oversized, prefix sniff; 3 listen-
addr filter; 5 PoolManager paths). 1030 → 1048 lib tests. Commits:
`c49632af..d7d77e6e`.

### R139 — Tier 4K communication-computation overlap (2026-05-19)

Five commits closing FUTURE_WORK Tier 4K with a research-driven scope
pivot. Phase B turned out to already be shipped via existing async
architecture; documented and skipped. Original "worker streams row-
tiled output during matmul" pivoted to **daemon-side STREAM-chunked
encrypt+send** on a single libp2p stream (age STREAM construction +
TokenWeave K=2-4 sweet spot + Tink Streaming AEAD precedent) after
research found no production inference system streams forward-output
tensors (Triton decoupled, vLLM v1, NVIDIA Dynamo/NIXL — all single-
tensor responses).

- **Phase C** (`11333f67`) — ChaCha20-Poly1305 encrypt/decrypt
  offloaded from NetworkManager event loop. CPU-bound sealing in
  `handle_send_tensor` and the open in `handle_tensor_payload`
  (TENSOR_TAG_ENCRYPTED arm) now `tokio::spawn` tasks. New
  `NetworkCommand::SendEncodedTensor` carries the encrypt-result
  back. ~50-200µs/forward event-loop savings; under concurrent
  decode this is the difference between smooth event-loop
  responsiveness and observable jitter on libp2p ping / gossip /
  connection events.
- **A-rev.1** (`4b5fc10c`) — Wire-format `ChunkMeta { chunk_idx,
  total_chunks }` trailer 0x05 + AAD binding via
  `build_layer_forward_aad`. Reorder, wrong-total, and cross-transfer
  substitution all fail Poly1305 before reaching dispatch. 11 new
  tests.
- **A-rev.2/3** (`1d0a5d55`) — Receiver assembly on
  `SharedState.pending_activation_chunks: DashMap<Uuid,
  ChunkAssemblyState>` (root-level). `chunk_layer_forward` splits at
  byte offsets; passthrough when activation ≤ chunk_size. 4 config
  knobs (`streaming_chunked_send`, `streaming_chunk_size_bytes`,
  `streaming_min_activation_bytes`, `streaming_chunk_assembly_ttl_secs`),
  default off.
- **A-rev.4** (`e32c0a5d`) — Sender wired in
  `pipeline/distributed.rs::forward_through_segments` persistent-
  stream path. RR fallback intentionally NOT wired (needs per-chunk
  Acks — deferred).

1015 → 1030 lib tests (+15) + 4 new swarmllm-types tests. Clippy
clean default + features dev,claude-subscription. Commits:
`11333f67..1108c817`.

### R138 — Autonomous defer-batch sweep-log triage (2026-05-18)

Eight commits closing ~20 deferred sweep-log items via 8 real fixes +
~15 verification-only entries:

1. **Auto-manage rescan respects `auto_manage_paused`** (R104 closure).
   Rescan still runs locally (correctness — picking up manually-
   placed shards) but the network re-announce gates on
   `auto_manage_enabled`.
2. **`active_count.fetch_add` inside spawn closure** (R103 closure).
   No leak on `tokio::spawn` OOM panic.
3. **`CreditBalance` `#[serde(default)]` + doc convention** (R105
   closure). Forward-compat for schema upgrade; node never silently
   restarts at zero balance on field addition. `swarmllm-types` got
   a `[dev-dependencies] serde_json` so 15 previously-dead lib tests
   now run.
4. **`private_mode`/`offline_mode` moved out of `pool_state` tree**
   (R105 closure). New `TREE_NODE_MODES` + `restore_node_mode()`
   migration helper. Each tree single-typed.
5. **`check_integrity` strict per-tree type validation** (R105
   closure). `validate_strict` routes each `CRITICAL_TREES` entry
   through the actual `swarmllm_types` type. Type mismatches that
   passed JSON-Value validation are now flagged corrupt.
6. **`credit_percentile_cache` no longer held across DashMap iter**
   (R97 closure). Three-phase pattern; router task no longer blocks
   on long iters.
7. **`api.metrics_auth_required` config flag** (R101/R102 closure).
   Tightens `/metrics` for public-internet nodes by removing the
   loopback exemption.
8. **Credit forward per-window value cap** (R102 closure).
   `CREDIT_FORWARD_MAX_VALUE_PER_WINDOW = 200k` credits/min/member
   on top of the existing count cap.

Plus ~15 verification-only sweep-log closures for items intervening
rounds had already addressed (R66/R67/R68/R89/R97/R102/R103/R104/R105/
R123). 1005 → 1015 lib tests (+10) plus 15 newly-runnable
`swarmllm-types` tests. Clippy clean default + features dev,claude-
subscription + features llama. Commits: `d122e9e8..cb60b2ed`.

### Sweep arc R122 → R124 (2026-05-12 → 2026-05-13)

Three rounds of standard-rotation sweeps after R121 closed. R122 and
R123 found 28 fixable findings; R124 stalled (worktree agents hit the
10-min watchdog) and was closed without findings — diminishing returns
on top of the R101-R109 security arc and the R110+ structural work.

**R122 (commit b530c42b) — 22 findings auto-fixed**

- 20 SwarmError variant fixes across `src/inference/process_pool.rs`
  (14 sites — model-worker subprocess lifecycle: spawn, IPC connect,
  socket bind, send Forward/BatchForward/Generate, worker-dead),
  `src/update.rs` (4 sites — apply-update file operations),
  `src/api/admin_providers.rs:1029` (current_exe), and
  `src/api/claude_session/handlers.rs:166` (create_dir_all temp).
  All previously surfaced HTTP 500 ("internal error") for subprocess /
  OS-level failures that should be 503 ("service unavailable"). Same
  pattern R118-R120 fixed for `claude_sub.rs` and
  `claude_session/manager.rs`. The `get_or_spawn` slow path already
  used `ServiceUnavailable`, making the inconsistency visible.
- `src/inference/tensor_util.rs:11-12` — `DTYPE_TAG_F32` /
  `DTYPE_TAG_Q8_0` had `pub` visibility but only in-file callers.
  Tightened to module-private `const`.
- `frontend/js/components/notifications.js` — three call sites
  inlined `500ms / 2000ms` latency-tier thresholds with two distinct
  class systems (`dot-*` vs `health-fast/ok/slow`). Extracted to
  `_dotLatencyTier()` and `_healthLatencyTier()` helpers.
- Doc drift: test count 909→913, i18n key count 1122→1130 (post-R121).

**R123 (commit d2a876bf) — 6 findings auto-fixed; 3 false-positive
i18n orphans caught**

- `src/api/admin.rs:362` — `update_config` silently mapped any
  unknown contribution-mode string to `Moderate` via `_ => `. Now
  returns 400 Validation per the contract.
- `src/api/admin.rs:471` — config-save `std::fs::write` OS failure
  promoted from `Internal` to `ServiceUnavailable`.
- `src/api/anthropic/handlers.rs` — two near-identical
  `MessagesResponse` builders extracted to
  `fn build_messages_response(request_id, model, output) ->
  MessagesResponse`. Third call site (legacy executor in
  `mod.rs:245`) uses a different result type and stays inline.
- `frontend/js/core/utils.js:411` — duplicate active-state predicate
  now defers to `App.downloads.isActiveDlState`.
- `docs/ARCHITECTURE.md` — `ModelMgmt` sub-struct diagram missing
  `contribution_auto` (R121); `dashboard-shards.js` listed twice in
  JS component list. Both fixed.
- **Verification gating caught 3 false positives**: Agent 4 flagged
  `settings.section_contribution`, `section_identity`, and
  `section_preferences` as orphaned i18n keys; the required cross-grep
  (`frontend/js/`, `index.html`, `css/`) found `data-i18n` callers
  in `index.html:302/245/270` for all three. Logged as wontfix so
  next sweep doesn't re-claim them.

**R124 (commit 66426fb4) — inconclusive**

Three specialised agents (concurrency + hot-path, security + authz,
Rust idioms + perf) all stalled at the 600s worktree watchdog. Closed
without findings; the deeper categories the round targeted (security,
concurrency) were already drilled hard in R101-R109 (44 fixes), so the
miss is not load-bearing. Re-run if a specific concern surfaces.

**Stats**

- 28 findings auto-fixed across R122-R123 (22 + 6)
- 3 false-positive i18n orphans caught by R120 verification rule
- 5 deferred (separator divergence, country-names i18n, etc.) —
  logged in `.claude/sweep-log.jsonl`
- sweep-log.jsonl grew 1131 → 1173 (+42 entries)
- 913 lib tests pass; clippy clean default + dev features

### R121 — Auto-manage scale-back at swarm saturation (2026-05-12)

Auto-manage learned to scale a node's contribution DOWN, not just up.
At swarm scale (1000s of nodes), a popular model is held by far more
peers than the geo-aware target needs — an idle node's shards become
redundant and just waste VRAM. R121 lets auto-manage shed those shards
voluntarily, without waiting for VRAM/disk pressure to build.

**New config field** — `[node] contribution_auto: bool` (default `true`).
Auto mode lets auto-manage scale contribution up AND down within the
user-set `[node] contribution` cap and `[resources]` caps. Manual mode
(`false`) pins contribution at the user-set level — pre-R121 behaviour.

**Saturation-aware prune.** `model/auto_manage/prune.rs` gains
`effective_prune_target(target, pressure, holder_count, contribution_auto,
min_replicas)`. When `contribution_auto` is true AND
`holder_count >= 1.5 × target`, the function bypasses the RELAXED-state
+1 nudge from `pressure_adjusted_target` and uses the raw target — so
the shard is eligible to prune even at zero local pressure. Severe
saturation (`holder_count >= 2 × target`) gets a flat +1.0 prune-score
bonus to break ties. All existing prune guards still apply
(active-pipeline, pinned/locked shards, configured-range, would-eliminate-
region, can-reacquire, recently-acquired, encrypted-pipeline models).

**Hot-reload.** `state.models.contribution_auto: AtomicBool` mirrors the
config field. `PUT /api/admin/config` updates the atomic so the toggle
takes effect on the next prune tick without a daemon restart. `state.config`
remains startup-frozen — the atomic is the only runtime source of truth
for the toggle, and the field is documented in `.claude/rules/architecture.md`
as a SharedState invariant.

**API.** `ConfigUpdate` and `GET /api/admin/config` gain
`contribution_auto: bool` and `max_gpu_vram_mb: u64`. The latter was
previously in `[resources]` config but not in the API — users had to
edit TOML and restart to set a VRAM cap. The VRAM cap is hard-capped
at 1 TiB on PUT to prevent UI typos from disabling VRAM accounting.
`OperationalParams` gains `contribution`, `contribution_auto`,
`max_gpu_vram_mb` for completeness (broadcast on hot-reload).

**Frontend.** Auto/Manual toggle in Settings panel above the existing
contribution segmented control. In Auto mode, "Contribution Level"
relabels to "Upper Cap" and the hint explains the scale-back semantics.
Wired via `App.settings._applyContributionMode(modeAuto)`. New i18n
keys: `settings.contribution_label_cap`, `settings.contribution_mode_*`
(8 keys total) translated across all 21 locales by translator-agent.

**Tests.** 4 new unit tests in `prune::tests` cover the saturation
override at boundaries (not-saturated/fall-through, saturated/no-pressure,
severe-saturation, min-replicas floor). All 913 lib tests pass; clippy
clean default + `--features dev,claude-subscription`.

**Deferred to a follow-up.** Setup wizard exposes the toggle (the wizard
already has the segmented contribution control + auto-manage checkbox;
the toggle slots in but is additive). Holder counts at >50 use the
gossip-cached value rather than a separate uncapped DHT-provider count
— at realistic targets the 50-cap is well above SATURATION_FACTOR×target,
so prune still fires correctly, only the displayed redundancy_ratio
underestimates how aggressively to prune. Both items captured in
`docs/FUTURE_WORK.md`.

### Security & stability sweep arc R92 → R109 (2026-05-01 → 2026-05-08)

Eighteen rounds of security and stability sweeps, ~150 fixes total.
Highlights: 14 HTTP/P2P/crypto vulnerabilities (R101), 17 concurrency/
economic/DoS findings including a CVE-adjacent issue (R102), 13 authz/
numeric/cancel/CI fixes (R103), 10 auto-manage audit fixes + shard-
reality messaging (R104), 11 inference math + distributed pipeline +
DB + HF + metrics fixes (R105), 17 logprobs/oracle/tier-bypass/spec-
DSD bookkeeping fixes incl. dedup helpers (R106), 9 tier-bypass/weak-
ordering/DoS-cap/crash-ordering fixes (R107), 9 Qwen3.5 KV mask + API
parity + hot-path perf fixes (R108), 3 R108-deferred completions (R109).
AAD now covers spec/kv-truncate trailer fields. Per-page bootstrap
nonce on `/api/admin/api-key`. MCP method-string reflection cap.

### Model management redesign — R110 → R116 (2026-05-08 → 2026-05-10)

User-visible swarm and model UX rebuild for non-technical audiences:

- **R110 — swarm-capacity foundation.** New `state.metrics.swarm_capacity`
  (ArcSwap<SwarmCapacity>) snapshot eagerly refreshed on peer connect/
  disconnect so the dashboard banner stays consistent with the peer
  panel under churn (the 1.5s stats-cache coalesce alone was too lazy).
  Makes collective swarm power visible.
- **R111 — wishlist subsystem.** `state.models.wishlist` shows users
  what auto-manage is actually planning. Surfaced through a new Swarm
  tab (`frontend/js/components/swarm-tab.js`) with wishlist + Capacity
  Plan views.
- **R112 — HfWatcher.** New subsystem (now 12 Tokio tasks; was 11). Hourly
  HuggingFace trending-GGUF poll seeds the wishlist and auto-promotes
  models above download/age thresholds to `DemandVerified`.
- **R113 — Capacity Plan / What-If.** Turns "contribute" from abstract
  to concrete: shows what specific shards your VRAM could host.
- **R114 — HF browse polish.** Task filter chips + status-driven CTAs.
- **R115 — onboarding storage preview** and i18n parity audit.
- **R116 — audit closure.** R110-R115 i18n translations finalized
  across all 21 locales (1122 keys / 1124 entries per locale, native
  language strings rather than English fallback). Active-pipeline
  guard added to `delete_model`/`delete_shard` admin handlers — yanking
  a shard mid-token-loop now returns 503 instead of corrupting state.
- **Dashboard rework.** Network Status panel + Models tab + inline HF
  browser. Removed the legacy HF download modal.

### Sweep arc R117 → R120 (2026-05-10 → 2026-05-12)

Four sweep rounds, 22 findings auto-fixed:

- **R117** — `spawn_check_and_load` helper consolidates the "shard
  landed → reload model → refresh dashboard" pattern across three call
  sites. `SystemContent::to_plain_text` dedup. `browse.*` translations
  added to all 20 non-English locales. Modal-only dead code dropped
  from HTML/CSS/JS/i18n.
- **R118** — 7 findings: error variant misuse cleanup (subprocess paths
  Internal → ServiceUnavailable), input validation cap on `tasks`
  query param, dead `#[allow(dead_code)]` attribute removed, doc drift
  fixes (HTML template count 13→11, dead "provider badges" prose
  removed).
- **R119** — 5 findings: dead pub visibility tightened (`MAX_TRENDING_
  ENTRIES`), 7 subprocess sites Internal → ServiceUnavailable in
  `claude_sub.rs`, 3 `write_to_stdin` sites in `claude_session.rs`,
  `openai/responses/translate.rs` upstream parse failures
  Internal → ProviderError {502}. Dead `App.hf.download` removed with
  4 orphaned i18n keys cleared across all 21 locales.
- **R120** — 5 findings, 1 wontfix: more dead pub visibility tightening
  (`coalesce_byte_ranges`, `cross_node_prefix_holders`, `EVENT_BUFFER_
  CAP`). MCP `tool_delegate` replaced 70-line inline /v1/messages
  dispatch with `dispatch_model_call` helper. Verification gating
  caught 6 false-positive i18n orphan claims (callers existed in
  init.js/utils.js/chat.js) — logged as wontfix instead of deleted.
- **completeness.md** codified the new sweep rules: "Verify before
  deleting sweep findings" and "Re-exports and visibility downgrades".

909 lib tests + 75 integration tests passing at the end of R120; clippy
clean on default + `--features llama`. CI now ignores 5 known accepted
RUSTSEC advisories (added RUSTSEC-2026-0097 ignore to match
`SECURITY.md` table; rand custom-logger advisory is warning-level
without a custom logger but is now explicitly ignored for parity).

### Sweep arc R76 → R81 (2026-04-29 / 2026-04-30, autonomous)

Self-managed overnight sweep covering: doc drift + dead code (R76),
concurrency + lifecycle (R77), error paths + resilience (R78), security
+ wire format (R79), frontend (R80), hot-path performance (R81). Each
round spawned 3–4 parallel review agents, applied auto-fixable
findings in batched commits. ~7 commits.

- **R76 (doc drift + dead code)** — Removed unused `SplitModel::n_kv_head()`
  accessor (no call sites). Fixed `request_response` timeout doc claim
  300s→600s. Clarified `GossipSub mesh_outbound_min` scales with peer
  count (1→4 across 6 buckets) instead of fixed=1. Corrected several
  ARCHITECTURE.md path references (`pipeline/distributed.rs`, not
  `pipeline/mod.rs`; `state/activity.rs` for ActivityEvent struct).
  i18n key count 1014→1015 (verified). CLAUDE.md Bronze tier:
  "negative balance" → "zero/negative" to match `priority.rs`.
  Replaced bare `#[allow(clippy::excessive_precision)]` on CLIP
  constants with rationale comment.
- **R77 (concurrency + lifecycle)** — Five concurrency bugs:
  - `dispatch/mod.rs:880` pool_tx RwLock guard held across send().await
    (could deadlock dispatcher AND any installer of pool_tx). Clone
    Sender out of guard before send.
  - `dispatch/mod.rs:261` per-peer LayerForward count load-then-add
    race (admits MAX+1 forwards from one peer). Switch to optimistic
    fetch_add → check prev → fetch_sub on overshoot.
  - `dispatch/mod.rs:456` router_tx send().await blocking the dispatch
    loop on a backlogged InferenceRouter. Switch to try_send.
  - `dispatch/mod.rs:402` pending_vision_results get-then-remove
    TOCTOU vs health-monitor's stale-entry sweep. Atomic remove +
    re-insert on sender mismatch.
  - `process_pool.rs:905` spawn_failures cooldown bypass when fresh
    failure recorded between (clear) check and spawn_lock acquisition.
    Move cooldown check inside the lock.
  - `network/manager/requests.rs:398` peer_to_node Ref held across
    peer_registry.get_mut (gotcha #10 pattern). Clone NodeId out of
    Ref before second DashMap access.
- **R78 (error paths + resilience)** — Wrong error variants and
  silent failures:
  - `inference/router/mod.rs:285` emit `InsufficientCredits {balance,
    required}` (→402) instead of `CreditError(string)` (catch-all
    →500). Same fix on queue-full ServiceUnavailable.
  - `error.rs:203` `SwarmError::Config` maps to 500/server_error,
    not 400/invalid_request_error (rule: Config is startup-only).
  - `chat_template/eval.rs:284` guard `% 0` modulo. Peer-supplied
    GGUF chat templates can crash the worker subprocess every request.
  - `chat_template/eval.rs` depth-limit recursion (Cell<u32> + RAII
    DepthGuard, MAX_TEMPLATE_DEPTH=256). Prevents stack overflow
    from `((((...))))` or nested for-in-for templates.
  - `network/manager/tensors.rs:212` gate per-tensor-forward
    connection_addrs Vec<String> dump on tracing::enabled!(DEBUG).
    At default info level the eager allocation burned ~50–100 KB
    throwaway heap per LayerForward.
  - `model/acquisition.rs:344` set `AcquisitionState::Failed` when
    manifest save fails (was leaving the dashboard stuck on
    Downloading forever).
  - `api/admin_models/lifecycle.rs:58` log DB::remove errors instead
    of `_ = ...`. Asymmetry vs file-removal path caused divergence
    after partial deletes.
  - `update.rs:97` log warn when `current_exe()` falls back to
    "swarmllm" relative path (sandboxed envs).
- **R79 (security + wire format)** — Replay protection, input
  validation, IPv6 CORS:
  - `dispatch/mod.rs:808` (NicknameGossip): one-sided staleness per
    gotcha #44. Future-dated records were squatting peer nicknames
    for up to 24 hours.
  - `dispatch/mod.rs:518` (CreditTransaction): freshness window
    (30s skew, 5min max age). Was admissible indefinitely.
  - `inference/split/gguf_meta.rs`: cap block_count, embedding_length,
    head_count, head_count_kv at sane upper bounds (256 / 65 536 /
    256). Crafted GGUF was driving worker into oversized KV/mask
    allocations.
  - `pool/manager/gossip.rs:286`: cap inbound device_name to 64 bytes.
    Local handler caps at 32 chars but inbound gossip path was
    persisting multi-MB strings to all pool members' redb.
  - `api/mod.rs`: add `validate_optional_sampling` for top_p,
    top_logprobs, presence_penalty, frequency_penalty (was silently
    clamped inside build_sampling_params, violating OpenAI-compat
    spec contract).
  - `api/openai/mod.rs`: cap response_format.json_schema.name (256B)
    and .schema (64KB). Bypassed validate_content_size.
  - `middleware.rs` cors_layer + `websocket.rs` origin allowlist:
    add `http://[::1]:{port}` for IPv6-only browsers.
- **R80 (frontend)** — Two real bugs:
  - `dashboard.js`: ResizeObserver/IntersectionObserver attached to
    .shard-matrix per model card stayed live after `list.innerHTML=''`
    wiped the DOM, accumulating across each `models_changed`
    re-render. Add `_disconnectMatrixObservers()` called before each
    of the 3 wipe sites.
  - `chat.js`: `saveSessions()` now surfaces QuotaExceededError as a
    warning toast instead of silently dropping chat history. New
    i18n key `chat.storage_quota_exceeded` across all 21 language
    files (1015→1016 keys, 1017→1018 entries per locale).
- **R81 (hot-path performance)** — Three measurable wins:
  - `pipeline/distributed.rs:787` per-token info!→debug! for the
    `Sending LayerForward to remote segment` log. ~4 String allocs/
    token/segment saved at default level.
  - `split/executor.rs:278` gate the unconditional `Instant::now()`
    on `tracing::enabled!(DEBUG)`. clock_gettime syscall per forward
    eliminated when info-only.
  - `router/mod.rs` hoist `chatml_fallback()` out of the if-let so
    the same prompt String is reused by both
    `check_multi_turn_reuse` and `register_multi_turn` on the
    cache-miss branch.

Out-of-band fix: formalized the `tests/fixtures/tiny_model/` fixture
deferral (Option B) — false aspirational claim removed from CLAUDE.md
and `local_embedder.rs` test gated on `SWARMLLM_TEST_MODEL_DIR` env
var pointing at a real on-disk model dir. New entry under § Deferred
Items in `docs/ARCHITECTURE.md` explains why a synthetic random-weight
GGUF would only catch parser/IPC plumbing bugs already covered by
unit + in-process tests.

### Sweep arc R82 → R89 (2026-04-30, autonomous)

Continuation of the overnight sweep arc, six new rounds plus a follow-up
deferred-item batch. ~17 commits. Each round spawned 3–4 parallel
review agents and applied auto-fix findings as they landed.

- **R82 (RAII guard scope)** — Document why
  `pending_layer_results` RAII guard cannot be applied in
  `forward_through_segments`: borrow conflicts with later `&mut self`
  segment calls would force an unwieldy lifetime gymnastics. Note in
  source comment + sweep log; the sibling `failover_segment` IS
  patched in R85.
- **R83 (correctness)** — Prune race window in `auto_manage/prune.rs`
  (compute redundancy, decide to prune, but didn't re-check between
  the decision and the file rm; R83 atomic-marks as pruning before
  fs::remove). Escrow lock pattern aligned with credit-tx pattern
  (apply balance change first, then audit log). Smaller fixes:
  `unwrap()` paths reachable on bad input, redundant `clone()` in
  hot scoring loop.
- **R84 (pool gossip + error types + i18n)** — Pool gossip caps for
  `DeviceStatsReport.device_name`/`models_hosted`/`model_name_len`
  (multi-MB strings could otherwise be smuggled into every peer's
  pool state via inbound gossip). Two error-type fixes:
  `SwarmError::Internal` → `Validation` for API input rejections,
  `SwarmError::Config` → `Internal` on a runtime-only path. i18n
  count-interpolation pattern (`{count}` placeholder) replacing
  ad-hoc `' ' + n + ' '` string concat.
- **R85 (security hygiene)** — Zeroize on drop for
  `crypto/session.rs::CachedSession` and provider API keys
  (`crypto/provider_keys.rs`) so process memory dumps don't leak
  recently-rotated session keys / tokens. Also: failover path
  `pending_layer_results` leak (gotcha #70). Pattern lifts the RAII
  guard from gotcha #45 — clean up the slot on `wait_for_result`
  Err path so a double-timeout (primary + standby both unreliable)
  doesn't permanently deplete `MAX_PENDING_LAYER_RESULTS=1024`.
- **R86 (dispatch + cleanup)** — Shard continuation orphan
  (`pending_shard_continuations` map grew without cleanup if the
  client disconnected mid-stream; TTL sweep added). Dead `NodeStats`
  fields removed (`bytes_uploaded`/`bytes_downloaded` superseded by
  `shared_state.shard_bytes_served` atomic). Two flaky tests
  stabilised. Python `swarmllm-client` stats endpoint mirrored the
  R67 atomic split. **Follow-up**: shard-serve disk I/O + bandwidth
  throttle moved off the swarm event loop (gotcha #11 / #71). The
  inbound `SwarmRequest::ShardTransfer` handler had been doing
  `read_shard_chunk_async()` + `tokio::time::sleep()` (bandwidth
  cap) inline; at a 1 Mbps cap with 4 MB default chunk, the loop
  froze ~32 s. Now stashes the `ResponseChannel` in
  `pending_shard_responses` keyed by ticket and `tokio::spawn`s the
  I/O+throttle, mirroring the `PrefixKvFetch` pattern.
- **R87 (cancel observation + dispatch hardening)** — Cancel signal
  added to four fast paths that had forked from `execute_distributed`
  but omitted the cancel observation (gotcha #72: `local_exec`,
  `try_speculative_distributed`, `try_dsd_distributed`,
  `try_remote_generate_fastpath`). Dispatch hardening:
  - one-sided u64-millisecond gossip timestamp checks for
    `RegionShardSummary` and `ModelDemandGossip` (gotcha #73,
    extends gotcha #44 — `saturating_sub` returns 0 when `ts > now`,
    so future-dated messages bypassed the staleness gate);
  - dispatch loop sub-system sends switched from `.send().await` to
    `try_send` for `pool_cmd_tx` (gotcha #74 — the dispatch loop is
    the network event loop's only consumer, blocking it on a slow
    PoolManager starves every other inbound message);
  - cap raises for inbound caps that had been at TinyLlama-era
    sizes.
- **R88 (auto-update default + credit invariants)** — `AutoUpdateMode`
  default flipped `Stable` → `Disabled` (gotcha #75). Until binary
  signing (C1) lands, `src/update.rs` only verifies a SHA256 sidecar
  fetched from the same release as the binary; a compromised
  maintainer/CI token can publish a matching pair. Defaulting to
  Stable was silently downloading unsigned binaries on every node's
  startup. `apply_credit_direct` now reverts in-memory mutation on
  DB persist failure (gotcha #76 — the silent double-credit was
  undetectable from logs). Rebalancer cooldown scope tightened
  (was per-process when it should have been per-(model, target-region)).
  HTTP error-mapping fixes for two paths.
- **R89 (deferred-item follow-ups)** — Three deferred items closed:
  - HF download size cap + vision overflow guard + IPC header cursor
    fix (`147f474`).
  - **OpenAI `frequency_penalty`/`presence_penalty`** implemented per
    OpenAI spec (gotcha #77): penalty applied *before* temperature
    scaling, threading `generated_ids` through standard worker decode
    + batched decode + distributed-path `LayerForward.generated_ids`
    / `IpcForward.generated_ids` (`#[serde(default,
    skip_if=is_empty)]` so zero-penalty requests stay on existing
    wire shape). Only the LAST segment of `forward_through_segments`
    samples; intermediate segments don't.
  - **Auto-compute BLAKE3 for zero-hash shards + mmproj**
    (gotcha #78). `auto_manage/scan.rs::rescan_local_shards`, when
    seeing a manifest shard with `hash == [0u8; 32]` whose file
    passes the size check, computes BLAKE3 in `spawn_blocking`,
    persists, THEN registers as holder. Without this, manually-
    placed shards with zero-hash placeholder manifests would be
    announced to the network using only a size check. mmproj path
    in `auto_manage/download.rs::trigger_mmproj_download` does the
    same.

### Sweep arc R90 → R91 (2026-04-30 / 2026-05-01, autonomous)

Two more rounds, 25 fixes pushed. Continued the same pattern of 4
parallel review agents per round with worktree isolation.

- **R90** (`50af1b8`, 12 fixes) — Hot-path safety, dedup helpers,
  doc drift:
  - **Dead `updateShardsLive` summary block** in `dashboard.js`
    referenced 6 undefined counters and a `data-model-summary`
    element that no longer exists (replaced by torrent-style health
    bar in commit `8ac662c`). Removed the block + 6 orphan i18n keys
    across all 21 locales.
  - **`finish_speculative` inline copy** in `dsd.rs` replaced with
    method call (already on `PipelineExecutor`).
  - **`build_layer_forward_aad`** extracted to
    `network/protocol/encrypted.rs` so encrypt-side
    (`network/manager/tensors.rs::handle_send_tensor`) and decrypt-
    side (`decode_layer_forward_encrypted`) compute the AAD bytes
    from a single function. Drift between the two would silently
    break every encrypted forward — gotcha #80 pins the contract.
  - **`gossip_timestamp_fresh`** extracted in
    `daemon/dispatch/mod.rs` — the gotcha #44 one-sided staleness
    check was duplicated for `RegionShardSummary` and
    `ModelDemandGossip`. Now centralised; gotcha #81 records the
    helper as the canonical entry point for new gossip types.
  - **`MAX_RESPONSES_INPUT_ITEMS = 1024`** + per-`InputMessageItem`
    extras enforcement added to `validate_responses_ingress`. Closes
    a DoS surface where a request with thousands of message items
    each carrying 32 × 4 KB extras could bypass the top-level
    `extras` cap.
  - **DIAG demoted** `info!` → `debug!` for `build_prompt from
    header` (was firing on every prefill at default log level).
  - **`REMOTE_GENERATE_TOKEN_CHANNEL_CAP = 256`** named constant
    replacing magic literal.
  - **`/api/admin/hf/download` deprecation note** in
    `docs/ARCHITECTURE.md` — frontend MUST use `/download-shards`
    per CLAUDE.md "no implicit full model downloads" rule.
  - **`update.checking` i18n key** added across all 21 locales (the
    update download button was reusing `settings.detecting`, which
    is semantically hardware detection — confusing in the update
    context for users reading the translation literally).
  - **Doc count corrections** — lib tests 897 → 887 in CLAUDE.md
    Testing + Status sections; i18n keys 1030/1032 → 1025/1027 in
    CLAUDE.md + ARCHITECTURE.md.
- **R91** (`9993960`, 13 fixes) — Visibility, magic-number hoists,
  freshness dedup:
  - **Visibility narrowed** `pub(super)` → `fn` on four
    internal-only helpers: `eligible()` in `speculative.rs`,
    `dsd.rs`, `remote_generate.rs`, and `argmax()` in
    `speculative.rs`. None had cross-file callers.
  - **R90 regression caught** —
    `speculative.rs::finish_speculative` needed `pub(super)` for the
    cross-file call from `dsd.rs` introduced in R90's inline-copy
    removal. Default-feature `cargo check` skipped the llama path
    that triggered the breakage. Gotcha #79 records this footgun:
    visibility-tightening or cross-file refactors that touch the
    spec/dsd modules MUST verify with `cargo check --features
    llama`.
  - **Non-llama `DraftState` stub** had an unused `pos: usize` field
    guarded by `#[allow(dead_code)]`. Per
    `.claude/rules/completeness.md` `#[allow(dead_code)]` is a
    smell, not a fix. Converted to a unit struct (the stub is never
    constructed in non-llama builds).
  - **Hoisted 8 block-scoped consts** to module level:
    `MAX_TP_GROUP_SIZE` (`scheduler/mod.rs`),
    `MAX_MODEL_FILE_BYTES` (`huggingface/download.rs`),
    `MAX_DEVICE_NAME_BYTES` / `MAX_MODELS_HOSTED` /
    `MAX_MODEL_NAME_LEN` (`pool/manager/gossip.rs`),
    `MAX_INPUT_ITEMS_QUERY_LEN` / `INPUT_ITEMS_DEFAULT_PAGE_SIZE` /
    `INPUT_ITEMS_MAX_PAGE_SIZE` (`responses/mod.rs`).
  - **`check_signed_freshness`** added to `credit/ledger.rs`,
    `pub(crate) const CLOCK_SKEW_TOLERANCE_SECS` /
    `BALANCE_REPORT_MAX_AGE_SECS` exported. The
    dispatch-mod credit-transaction freshness check now uses the
    helper instead of duplicating both the constants and the
    one-sided staleness logic. Gotcha #82 records the helper as the
    canonical entry point for any new signed credit-typed message.
  - **Hardcoded `0.7` / `0.9`** in `responses/stream.rs` →
    `super::DEFAULT_TEMPERATURE` / `DEFAULT_TOP_P`. Streaming-path
    fallback now matches the non-streaming path which already
    referenced the named constants.
  - **`compare.js` inline error extraction** → `U.extractErrorMessage`
    helper (with JSON-stringify fallback to preserve debug info).
  - **`dashboard.js`** — peer-label fallback `'unknown'` →
    `I18n.t('utils.unknown_model')`; removed dead `|| tier`
    fallback (`I18n.t` returns the key on miss, never falsy).

After R91 the tree has **887 lib tests + 75 integration tests**
passing, clippy clean on both feature sets, both default and
`--features llama` compile. Sweep log
(`.claude/sweep-log.jsonl`) totals **825 entries** across 91 rounds.

### Deferred-item follow-ups (R72/R75 leftovers, 2026-04-29)

Three commits cleaning up the structurally-deferred items the sweep
arc surfaced:

- **Worker crash-loop backoff** (`bab361c`) — `ModelProcessPool` now
  tracks per-model spawn failures; arriving requests during the
  cooldown window get `ServiceUnavailable` instead of waiting for
  `WORKER_CONNECT_TIMEOUT_SECS=30s` per attempt. Backoff steps
  1→2→4→8→16→32→60 s, reset on first successful spawn.
- **Cancellation token in inference loop** (`bab361c`) —
  `InferenceRequest.cancel: Option<Arc<AtomicBool>>` (`#[serde(skip)]`),
  `SharedState.cancel_signals` map keyed by an opaque token via
  the `x-swarmllm-cancel-token` HTTP header. Pipeline checks
  `request.is_cancelled()` per-token. Wired end-to-end through
  `responses/background.rs` so `/v1/responses/{id}/cancel`
  actually interrupts in-flight inference within one forward.
- **Stop-sequence KV truncate** (`9f9f22e`) —
  `pipeline/distributed.rs` sends a finalising
  `LayerForward(truncate_kv_to=ptc, activations=[])` to every remote
  segment after a stop string fires on a session-keyed request.
  Without this the next session turn would see the stop tokens
  still in the remote KV and produce contaminated output.
- **Escrow refund on inference failure** (`9f9f22e`) —
  `finalize_request` now calls `refund_escrow` on the `Err` arm.
  Previously `refund_escrow` had no production caller; failed
  requests left credits locked until the cleanup tick. Credit
  enforcement isn't gating users yet, but the bookkeeping is solid
  for when it is.
- **Pre-release surface trim** (`1d7eb39`) — every internal module
  in `src/lib.rs` now `#[doc(hidden)]`; `api`, `config`, `error`,
  `types`, `update` are the documented stable API. New
  `tests/integration/end_to_end.rs` exercises the full HTTP +
  shutdown lifecycle (`#[ignore]`'d, ~10 s).

C1 (binary signing) stays open with a fully-researched options write-
up at `memory/signing_options.md` — recommendation is **minisign**;
landing it requires a key-custody decision from the maintainer.

### Sweep arc R69 → R75 (2026-04-29 / 2026-04-30)

Long-running self-managed sweep covering: post-audit follow-ups (R69),
hot-path performance (R70), concurrency + lifecycle (R71), error
recovery + resilience (R72), observability + operability (R73), API +
wire-format correctness (R74), and pre-release readiness (R75). Each
round spawned 4–5 parallel review agents, applied auto-fixable
findings in batched commits, and surfaced architectural items for
follow-up. ~20 commits, broad coverage:

- **Performance** — `PendingLayerResultGuard` shared between dsd /
  speculative; `req_id_str` and `SamplingParams` no longer cloned per
  decode tick in `step_decode_pool`; per-token `default_eos` HashSet
  hoisted out of the decode loop; `verify_tokens.to_vec()` dropped
  from spec-round LayerForward; admin/stats and websocket
  build_stats_message no longer hold RwLock guards across blocking
  work; `health/monitor.rs` switched from `block_in_place` to
  `spawn_blocking` for the 30s sysinfo refresh; frontend
  notifications + dashboard render skips on equal state +
  sessionStorage debounced.
- **Concurrency** — `batch_scheduler_loop` and
  `pipeline_stream::spawn_accept_loop` + `handle_inbound_stream` now
  observe the watch-channel shutdown signal; PendingLayerResultGuard
  applied to speculative.rs's two leak sites; `apply_pipeline_guard`
  RAII catches panic-induced leaks of `active_pipelines` /
  `active_count`; `dashboard_rx` Lagged path now sends a re-sync
  message instead of silently dropping.
- **Resilience** — `apply_update` collapsed into a single signature
  that always re-checks version; HF probe wrapped with the
  NETWORK_RETRY_DELAYS exponential backoff (was a bare `.await?`);
  pool slaves now forward the `penalty_serve_failure` to the master
  via the same path as success-case spends (was hitting the slave's
  local balance and gating its own future requests); update.rs
  `info.downloaded = true` only set when staging path is the
  preferred (same-filesystem) location, not the temp_dir EPERM
  fallback.
- **Observability** — credit spend logs promoted from debug to info
  with `DIAG:` prefix (the earn side already had it; the spend side
  was dark); `escrow_held` + `escrow_pending_count` exposed on
  `/api/admin/credits`; `subnet_counts` retain on register so a NAT
  flip doesn't pile a peer into multiple buckets.
- **API correctness** — `SwarmError::InsufficientCredits` returns a
  distinct `insufficient_credits` error_type instead of the
  `rate_limit_error` mismatch (SDK retry logic was treating credit
  stops as rate-limits); `SwarmError::Network` maps to 502 +
  `network_error`; streaming finish_reason='error' replaced with
  'stop' (not a valid OpenAI value); `/v1/models` `created` is the
  manifest publish_date timestamp not Utc::now() (was unstable per
  call); `sse.rs::data_frame` emits a structured error on
  serialization failure instead of silently emitting an empty
  `data:` line; spec_logits doc-comment in swarmllm-types matches
  gotcha #29's γ+1 contract.
- **Hygiene** — `WorkerOptions` struct bundles run_worker's 7 runtime
  knobs (was 13 args under `#[allow(too_many_arguments)]`);
  `SplitLoadOptions` for split/loader; `speculative::softmax`
  wrapper deleted; `// SYNC:` comment dropped (gotcha #18 is the
  canonical 3-path warning); inline "deferred ChaCha encryption"
  paragraphs collapsed to one ARCHITECTURE.md pointer; mmap SAFETY
  comment expanded with candle's qtensor_from_ggml copy-semantics
  proof.
- **Release prep** — i18n files now sorted alphabetically by key for
  reliable parity audits (1015 keys + 2 metadata = 1017 entries per
  locale, all 21 languages confirmed in parity); `default.toml`
  documents the speculative-fast-path ChaCha bypass next to
  `enable_encryption = true`; `yamux_substream` CI tests now run on
  macOS (the Linux-only guard was a copy-paste from the
  multi-process integration tests).

890 lib tests pass throughout; clippy clean both feature sets;
pre-push hooks passing on every commit.

### Performance

- **`PrefixCache` lookup hit no longer needs a write lock** —
  `Entry::last_hit` is now an `AtomicU64` driven by a per-cache logical
  clock; the read-lock walk records the hit via a `Relaxed` store
  instead of upgrading to write. Removes the entire write-acquire from
  every cache hit and its sleep-based test fixture. Touches
  `src/inference/split/prefix_cache.rs` only.
- **`local_exec` holds `loaded_model_info` once per batch item** — was
  acquired twice per request (chat-template prompt + stop-string list).
  Now extracted from a single guard.
- **`KvCacheManager::check_multi_turn_reuse` takes `&HashSet<NodeId>`
  instead of `&[NodeId]`** — the per-holder `contains()` was O(peers).
  Caller in `inference/router/mod.rs` collects the DashMap directly
  into a HashSet so the conversion is paid once.
- **`GET /api/admin/responses` streams + bounded heap** — new
  `Database::for_each_json` helper streams JSON-encoded records without
  materialising a full Vec; the listing endpoint maintains a min-heap
  of size `≤ limit` ordered by `created_at`. Memory now O(limit)
  instead of O(total_records); only survivors get the full preview JSON
  built.

- **zstd compression on `WIRE_TAG_PREFIX_KV`** — flag-gated via
  `NetworkConfig::prefix_kv_compression` (default off). Send-side reuses
  the existing tensor-compression helpers and falls back to raw when the
  compressed form isn't smaller. Receivers always decompress regardless
  of the flag, so flipping it on a single peer doesn't require a
  coordinated upgrade. Expect 30–50% wire reduction on KV snapshots
  (zero-padded regions compress well); WAN bench will decide default-on.
- **Hot-path syscall savings** — gated per-layer `Instant::now()` in
  `split/executor.rs` behind `tracing::enabled!(TRACE)` (28 syscalls/token
  on a 28-layer model wasted at default log level), and same for
  `pipeline/distributed.rs` per-token `fwd_start` behind `enabled!(DEBUG)`
  with the matching DIAG emit dropped to debug-level for consistency.
- **Worker IPC capacity** — bumped reader→main channel from 16 to 64 in
  `model_worker.rs`. The admit-coalescing drain loop pulls 16/tick and a
  single decode tick can be 100–500 ms on CPU 7B; bursts would block the
  reader on `.send()` and delay the cross-node `PrefixFetchResult`
  fast-path short-circuit.

### Security & validation

- **Black-hat sweep 2026-04-29** — six parallel adversarial reviewers
  (network, crypto/auth, pool/credit, HTTP API, inference path, supply
  chain). 7 commits landed (`f4ff02b..fff42b4`); see
  `memory/audit_2026-04-29.md` for the full rollup. Highlights:
  - **Network**: gossip timestamp staleness in `events.rs` was
    symmetric `saturating_sub` — doubled the replay window to ~10 min;
    now one-sided per gotcha #44. PEX response no longer calls
    `kademlia.add_address` on unauthenticated peer/multiaddr pairs
    (Kademlia eclipse vector). `spec_logits` decoder hard-caps
    `num_positions ≤ 32` and `vocab_len ≤ 512_000`.
  - **Pool & credit**: `TREE_POOL_REMOVAL_REPLAYS` is no longer cleared
    on restart — the 5-min freshness window IS the replay window, and
    a saved `PoolRemoval` could re-evict after a planned restart. Now
    timestamps each entry. `handle_inbound_acceptance` verifies the
    acceptance's `invitee_node_id` matches the pending invitation
    (anyone learning the `invitation_id` could otherwise steal the
    slot). `track_forward_participation` caps peer-controlled
    token count at 8192 — `LayerForward.token_count = u32::MAX`
    would otherwise mint ~43B credits per serving node per flush.
  - **Inference**: intermediate-segment activations are now
    shape-validated against the input we forwarded; a malicious peer
    returning a wrong-shaped tensor would otherwise crash the next
    worker (gotcha #20). NaN/Inf rejected from peer-supplied f32
    tensors at three sites (ring AllReduce, TP attn/ffn AllReduce
    result, `spec_logits` rows before argmax in
    `greedy_accept_reject` — IEEE 754 NaN argmax is non-deterministic
    and lets a malicious peer steer accepted tokens).
  - **Shard & adapter integrity**: shard startup size check tightened
    to exact match (was ±10%); LoRA safetensors now BLAKE3-pinned via
    a `<filename>.blake3` sidecar so a swapped adapter can't silently
    produce wrong inference output.
  - **HTTP API**: `POST /api/admin/update/{check,apply}` now
    loopback-only (auto-downloads + binary swap should not be remote).
    `provider-model-status` rejects path-injection in `model_id` via
    allowlist (`[A-Za-z0-9._:@-]`, max 256, no `..`). `join-network`
    rejects private/loopback/link-local multiaddrs (P2P-layer SSRF).
    `GET /api/admin/responses` only returns `input_preview` /
    `output_text_preview` to loopback callers (single shared API key
    leaked prompt prefixes across users). Removed unreachable dead
    code: the non-loopback `x-swarm-internal-token` block in
    middleware.rs (per gotcha #30, `internal_auth_token` is
    per-process random and never crosses node boundaries).
  - **Update + supply chain**: `apply_update` re-verifies version is
    strictly newer than `env!("CARGO_PKG_VERSION")` at apply time
    (downgrade-by-replay protection). All GitHub Actions in `ci.yml`
    and `release.yml` pinned to commit SHAs with the tag in a trailing
    comment; `.github/dependabot.yml` opens weekly bumps. `libc::umask(0o177)`
    around the AF_UNIX socket bind in `process_pool.rs` so the IPC
    socket is created at 0o600 atomically (closes a TOCTOU between
    bind and post-bind `set_permissions` where a local attacker
    racing inotify on `/tmp` could connect first and impersonate
    the worker). `db.redb` chmod'd to 0o600 on Unix after create.
    `.env` loaded BEFORE the Tokio runtime spawns worker threads
    (`std::env::set_var` is unsound in a multi-threaded process).
  - **Deferred (still open)**: C1 — auto-update binary signing.
    SHA256 sidecar comes from the same release as the binary, so a
    compromised account/CI token publishes both together. Real fix
    needs an offline keypair embedded as `env!()` pubkey + cosign
    or minisign signature as a third release asset. Tracked in
    `docs/ARCHITECTURE.md` § Deferred Items.

- **PRIVACY: `chat_completions` no longer leaks prompt content into
  the activity event bus** — the event message included up to 60
  characters of the last user message via `prompt_preview`, and
  `emit_activity` broadcasts to every authenticated dashboard
  subscriber AND replays via `activity_history` to new connections.
  On a multi-tenant or shared-host node this leaked one user's
  prompt to others without their knowledge. The Anthropic handler
  did NOT reproduce this pattern. Removed the preview entirely;
  the event now carries only model, message count, and max tokens.
- **`lock_shard` / `unlock` now propagate DB write errors** — the
  handler used `let _ = db.insert_raw / .remove(...)` for the
  `locked_shards` persist, silently discarding write failures while
  still mutating the in-memory DashMap. On a DB error the handler
  returned `status: "ok"` despite a state divergence from disk;
  after restart the auto-manage pruner could remove a shard the
  operator believed was pinned. Same pattern was already corrected
  for pool pins, auto-manage policy, encrypted pipeline, and HF
  trust pin in earlier sweeps. Now persists first and surfaces any
  error as 500.

- **CRITICAL: `ResponsesRequest.extras` was an unbounded ingress vector**
  — the `#[serde(flatten)]` catch-all for unknown top-level JSON keys had
  no cap on count or per-value size. A request with thousands of unknown
  keys (or one very large value) was materialised into the in-process
  `HashMap<String, Value>` before any validation could reject it AND
  forwarded verbatim to upstream cloud providers on the proxy path.
  Added `MAX_RESPONSES_EXTRAS_COUNT = 32` and
  `MAX_RESPONSES_EXTRA_VALUE_BYTES = 4096`, checked first in
  `validate_responses_ingress`.
- **`/v1/responses/{id}/input_items` query parameters length-capped** —
  `after`, `before`, `order`, `include` were unbounded. Synthetic
  `item_N` cursors are short by construction so any large value is
  hostile. Capped at 64 bytes each at handler entry.
- **CSP meta tag in `frontend/index.html` synced to server header** —
  the meta lacked `blob:` in `img-src`, `frame-ancestors 'none'`,
  `base-uri 'self'`, and `form-action 'self'`. The server header in
  `middleware.rs::security_headers` already had these and takes
  precedence at runtime, but the meta covers the
  static-server / file-open / cached paths where the daemon's header
  isn't applied. Added an inline comment marking middleware.rs as
  authoritative.
- **CRITICAL: `responses.js` was calling a non-existent function** —
  `App.data.authFetch` doesn't exist; the symbol is `App.authFetch`.
  Every Responses dashboard interaction (load, retrieve, cancel,
  delete) was silently failing with `TypeError`. Same pattern that
  was caught in `auto-manage-status.js` previously — the responses.js
  refactor that introduced the `_action` helper reintroduced it.
  Fixed via 3-site replace.
- **`/v1/responses/{id}` path-param validation** — added
  `validate_response_id` helper mirroring the
  `previous_response_id` cap (≤64 ASCII alphanumeric `+_-`). Called
  at the top of `get_response`, `cancel_response`, `delete_response`,
  `list_input_items`, and `get_response_maybe_stream`. Closes a path
  where a megabyte-long `{id}` could inflate logs / DashMap key
  materialization for `BACKGROUND_CANCEL` / `BACKGROUND_STATE`.
- **`GET /api/admin/responses ?status=` length cap** — raw status
  string was split with no length cap, allowing an unbounded
  `Vec<String>` allocation. Added a 256-byte cap before splitting.
- **`Q8_0` tensor non-finite guard** — F32 path already rejected
  NaN/Inf in deserialized activations; Q8_0 path dequantized blindly.
  Added a matching `is_finite()` check after `dequantize_q8_0` so a
  malicious or broken peer can't poison subsequent attention via
  NaN/Inf-dequantizing Q8_0 blocks. Also flipped the F32+Q8_0
  truncation error type from `Internal` (HTTP 500) to `Inference`
  (a truncated wire payload from a peer is a network fault, not a
  local code bug).

- **`/v1/responses` ingress validation** — `validate_responses_ingress`
  runs BEFORE the cloud-proxy / Anthropic-bridge / local-inference
  branches. Caps `previous_response_id` ≤64, `instructions` ≤2 MB,
  `user` ≤256, `model` 1..=256, `truncation`/`service_tier` ≤64,
  `metadata` aggregate ≤64 KB. Closes a path where attacker-sized
  strings could reach upstream provider request bodies and log lines.
- **`/v1/messages` `max_tokens` cap** — Anthropic handler now rejects
  `max_tokens > 32768` (matches the local sampling-params clamp ceiling)
  at ingress instead of silently clamping for local + forwarding raw to
  upstream proxies.
- **HF-proxy rate limiting** — `/api/admin/hf/probe` and
  `/api/admin/hf/search` no longer get the loopback admin-GET exemption.
  A runaway local script or a malicious browser extension on
  `localhost:8800` can no longer loop-call them and burn HuggingFace
  API quota.
- **Pool invite-code validator** — now checks ASCII alphanumeric AND
  length (was: length only after trim+uppercase, accepting some malformed
  inputs silently).
- **Identity leaderboard `?limit=`** — clamped to `[1, MAX]` (was: silent
  empty array on `limit=0`).

### CI

- **macOS test + clippy** — `.github/workflows/ci.yml` matrix now runs
  `cargo test --lib --bins` and clippy on `macos-15` in addition to
  Linux. Integration tests stay Linux-only with explicit guards until
  the first macOS failure decides whether to fix or skip-list per test.
  Build job already had macOS — this closes the test-coverage gap from
  `docs/plans/next_steps.md` § 4.

### Correctness

- **`pool::handle_leave_pool` lock ordering** — was the only pool
  handler holding the `pool_state` read lock across
  `rate_limiter.check_and_record()`. Now matches
  `handle_create_invitation` / `handle_accept_invitation`: extract
  `pool_id` under the guard, drop, rate-limit, then later acquire the
  write lock for mutation. All 4 pool handlers audited for ordering
  consistency.

### Reliability

- **CRITICAL: M9 background spawn now `catch_unwind`s a panic** —
  the non-stream `/v1/responses` background path was a bare
  `tokio::spawn` with no panic guard, while the V8 streaming path
  has had `AssertUnwindSafe(...).catch_unwind()` around it since
  V8 landed. On a panic anywhere in `run_background_inference`
  (translate / chat_completions / buffer / parse / chat→responses
  translate) the redb record was stranded at `status:in_progress`
  forever — no terminal state ever written — and a polling client
  would see `in_progress` indefinitely. Wrapped the spawn body in
  `catch_unwind`; on panic we stamp a terminal `failed` record into
  redb and call `unregister_background_cancel`.
- **`POST /v1/responses` (M9 background) now returns 202 Accepted**
  — was returning HTTP 200 even though the response was queued for
  async completion. V8 streaming was already correct. Clients that
  branch on status code to detect sync vs deferred completion would
  have misclassified M9 background as a synchronous result.

- **`BACKGROUND_CANCEL` TTL sweep** — the response-id → cancel-flag
  registry could leak entries when a Tokio task was cancelled
  externally (e.g. process shutdown mid-flight) before its cleanup
  path ran. Added a parallel `BACKGROUND_CANCEL_AGES` map keyed by
  response id with insert-time `Instant`; new
  `register_background_cancel` / `unregister_background_cancel`
  helpers wrap every insert / remove site so the two maps stay
  consistent. New `prune_stale_background_state` drops entries
  older than `BACKGROUND_CANCEL_MAX_AGE_SECS = 7200` (2 h —
  generously above any real background-inference run) plus the
  matching `BACKGROUND_STATE` entry. Wired into the existing
  hourly responses sweep.
- **`/api/admin/network_code` peer-iter caps** — the dashboard's
  invite-code refresh handler walked every peer and every
  advertised address with no bound. Added
  `NETWORK_CODE_PEER_SCAN_CAP = 64` and
  `NETWORK_CODE_ADDR_PER_PEER_CAP = 16`. A public-facing IP is
  almost always advertised by the first few peers, so capping the
  inner loops preserves the happy path and bounds the worst case.

### Polish

- **`dispatch_inference` / `router_inference_stream` no longer
  thread an unused `&AppState`** — the parameter was passed
  through both functions and read by neither. Dropped from both
  signatures + 3 call sites in `openai/mod.rs`.
- **`is_prefill` DIAG field uses `PREFILL_ACTIVATION_THRESHOLD_BYTES`
  instead of a hardcoded `100_000`** — the constant is `pub(crate)`
  and documented as a tuning knob; the bare literal would silently
  diverge the diagnostic from the real classifier.
- **`responses.js` `_statusCell` fallback fixed** — `I18n.t(key,
  string)` passes the second arg as an interpolation **object**,
  not a fallback string. Switched to the
  `translated !== key ? translated : raw_status` pattern used
  elsewhere so unknown future status values render as the raw word
  rather than the literal i18n key.
- **`notifications.js` removed a leftover `console.warn`** — the
  ws-ticket fetch catch block logged on every transient reconnect
  failure (visible in DevTools console). The user-visible
  connection-lost banner via the `onclose` path is the right
  surface; logging adds nothing.

### Observability

- **`forward_batch` fallback log gained `model_id`** — the
  `kv_offset mismatch — falling back to sequential` debug log was
  missing the model key, making it impossible to correlate which
  model triggered the fallback in a multi-model deployment.

### Refactor / dedup

- **`buffer_and_translate_chat_response` helper** collapses the
  three-step buffer/parse/translate boilerplate that
  `create_response` and `run_background_inference` each hand-rolled
  with its own error-handling shape. Returns
  `Result<ResponsesResponse, String>`; each caller wraps the
  human-readable error string into its native error type. Site 1
  went from 13 lines + 2 match arms to 8 lines + 1 `.map_err`;
  site 2 went from 32 lines + 3 match arms to 14 lines + 1 match.
- **`speculative::greedy_accept_reject`** extracts the bit-identical
  14-line accept-reject arithmetic shared between
  `try_speculative_distributed` (Item 2) and `try_dsd_distributed`
  (Item 12). Token emission and gamma-controller bookkeeping that
  surrounds the call stay per-path because the divergence there is
  real.
- **`build_chat_completion_response` helper** in
  `api/openai/streaming.rs` collapses the response-shape
  construction shared between `router_inference` (with logprobs +
  session id) and `split_non_stream_response` (without). The
  empty-vec → `None` logprob conversion lives in one place.

- **`ResponseError::new(code, message)` constructor** replaces 11
  identical `ResponseError { code, message, extras: HashMap::new() }`
  struct literals across `responses/{mod, stream}.rs`.
- **`new_response_id()` / `new_message_id()` helpers** in
  `responses::mod` replace the
  `format!("resp_{}", uuid::Uuid::new_v4().simple())` /
  `format!("msg_{}", ...)` idiom that appeared at 7+ sites across
  the responses module. The `rs_` (reasoning item) and `resp_test_`
  (test fixture) prefixes intentionally stay inline — different
  conventions.
- **`apply_shard_window_change` helper** in
  `admin_models::helpers` extracts the post-window-compute /
  pre-event-emit block shared between `unload_shard` and
  `load_shard`: empty-window→`evict_and_unload` else
  `restart_with_window` + `evict_split_models`, then
  `clear_model_load_history` and `signal_dashboard(ModelsChanged)`.
  The activity-event + tracing tails stay per-handler (different
  messaging is the right answer there).
- **`DEFAULT_TEMPERATURE` / `DEFAULT_TOP_P` promoted to `pub(super)`**
  in `responses::mod`, mirroring the existing `DEFAULT_MAX_OUTPUT_TOKENS`
  shape. `translate.rs` now imports via `super::` instead of holding
  its own private literal — closes a drift hazard.
- **`_isActiveDlState(state)` helper** in `downloads.js` replaces 4
  duplicated `(state === 'downloading' || state === 'awaiting_manifest')`
  predicates and normalises a missing `typeof === 'string'` type guard
  in one of the four sites.
- **`detect_tp_groups` dropped its unused `_manifest: &ModelManifest`
  parameter** — the function operates on `candidates` and `segments`
  only.

- **SSE parser dedup across `responses/{stream,anthropic_bridge}.rs`** —
  promoted `drain_sse_data_payloads` and `find_subslice` from private
  to `pub(super)`, extracted a new `parse_sse_block_data_lines` helper
  for the per-block parsing both call sites need, and dropped
  `anthropic_bridge::find_event_boundary` (was the same logic under a
  different name).
- **`speculative_common_eligible` helper** — `speculative.rs` and
  `dsd.rs` shared a 6-line `eligible()` prefix (decoding flag, draft
  model loaded, greedy temperature, no encryption, etc.). Extracted
  to `pipeline/mod.rs`; each path now calls it after its own
  path-specific flag check.
- **`network-map.js` map-stats helper** — `render()` and
  `updateFromWs()` built the same I18n-formatted node/region count
  text inline; extracted `_updateMapStats(totalNodes, totalRegions, maxCount)`.
- **`responses.status_unknown` i18n key** — added across all 21 locale
  files; was missing, so any non-English user seeing a response with
  an unrecognized status value got the raw English `unknown`.
- **`auto-manage-status.js` plural fallback** — the active-download
  fallback string used `shard(s)` parenthetical pluralization;
  replaced with a proper singular/plural branch.
- **`detect_tp_groups` cleanup** — dropped an unused `_manifest:
  &ModelManifest` parameter that did nothing at the call sites; the
  function operates on `candidates` and `segments` only. Also renamed
  `_is_last` → `is_last` in `tensor_parallel.rs::execute_tp_segment`
  (the underscore convention falsely implied unused; the param is
  read at lines 316 and 357).

- **`CreditDelta` enum replaces `is_earning: bool`** on
  `apply_credit_direct`. Variants `Earning` / `Spending` / `Refund`;
  the new `Refund` leaves both monotonic counters
  (`lifetime_earned`/`lifetime_spent`) untouched, which is the
  semantics escrow refunds need but couldn't get from the old bool.
  `EscrowManager::create_escrow` and `refund_escrow` now route through
  the helper instead of hand-rolling the balance write + persist.
  `cleanup_expired` keeps its manual block (its retry-on-failure
  semantics need an in-memory revert that `apply_credit_direct`
  deliberately doesn't do); the reason is documented inline.
- **`pipeline::fastpath_request_disqualified`** helper extracts the
  shared TP-empty + LoRA + vision-images guard from
  `remote_generate.rs`, `speculative.rs`, and `dsd.rs`. Per-path
  shape / encryption / flag preconditions stay separate because they
  ARE subtly divergent (1-segment vs 2+-segment vs all-remote, and
  remote_generate has a per-model encryption gate the others don't).
- `SharedState::resolve_peer_id_bytes` helper replaces a 5-site
  `peer_id_map.or_else(peer_registry)` lookup duplication across
  `inference/pipeline/{distributed,remote_generate,speculative,dsd,
  tensor_parallel}.rs`.
- `crypto::hkdf_sha256_derive_32` helper replaces 3 sites that
  duplicated the same `Hkdf::new + expand + .expect("32 bytes is a
  valid HKDF-SHA256 output length")` pattern across `provider_keys`,
  `gossip_seal`, `session`.
- `cli::discover_model` helper replaces a 14-line `GET /v1/models`
  block duplicated in `bench.rs` + `chat.rs`.
- `auto_manage::manager::read_shard_pins` helper replaces a `pool_state.try_read()`
  + `shard_pins` extraction duplicated in `scoring.rs` + `prune.rs`.
- `cli::bench::tokens_per_sec` and a `pub const SWARMLLM_GITHUB_REPO`
  in `update.rs` (previously bare `"enapt/SwarmLLM"` string in two
  places).
- Various small stale-doc + dead-code cleanups; see commits `36af419`,
  `c9acbfc`, `c10956e`, `ccfbf14`, `712a4da`, `d8a840c` for the
  per-sweep summaries.

### Tests

- 821 lib tests passing (was 816 at v0.1.0). Clippy clean both feature
  sets. Pre-push hooks enforce `cargo fmt && cargo clippy --all-targets
  -- -D warnings` on every commit.

## v0.1.0 — 2026-04-25

First non-alpha tag. Cuts off the v0.1.0-alpha.2 line and rolls every
post-alpha commit (`v0.1.0-alpha.2..main`, ~56 commits) into a single
release.

### What's new since alpha.2

- **OpenAI Responses API (`/v1/responses`)** — full v1 (M1–M9) + v2
  (V1–V8) surface. Cloud-proxy passthrough, local inference via Chat
  translation, Claude/Anthropic-Messages translation bridge,
  multimodal input, redb persistence with 30-day TTL, chained
  `previous_response_id`, background mode, resumable SSE +
  background streaming (202 + Location handshake), `input_items`
  pagination, dashboard panel under the new Responses tab. See
  `docs/plans/responses_api.md` and `docs/plans/responses_api_v2.md`
  for design + full bench numbers in `docs/bench_results/`.
- **Audit fixes (`audit_2026-04-24.md`)** — 2 CVE backports for
  vendored libp2p-gossipsub (CVE-2026-33040 + CVE-2026-34219), 5
  cargo-update CVE patches (quinn-proto + rustls-webpki), spec_logits
  IPC wire-format fix (no more 5–7× JSON bloat for f32 row-major
  payloads), Anthropic / OpenAI proxy unknown-field preservation +
  beta-header forwarding, cache-token usage roundtrip, escrow refund
  loss + false-toast + total_bytes + spawn_region + no_peers_interval
  fixes, hot-path mem::take in forward_through_segments. See
  `audit_2026-04-24.md` in memory for full chronology.
- **Audit fixes (this session)** — DNS TOCTOU closed via custom
  reqwest resolver that filters private IPs at request time; shard
  size check tightened to exact-match for zero-hash placeholder
  manifests; prune cycle re-checks resource pressure between scan and
  execution so over-pruning can't happen; Grafana docker-compose
  refuses to start without explicit env vars (no more silent
  admin/admin).
- **Dep upgrades** — axum 0.7 → 0.8 (route patterns rewritten,
  WebSocket Message types adapted, async_trait removed, tower-http
  TimeoutLayer signature change), redb 3 → 4 (1.5× write throughput,
  on-disk format compatible via existing UpgradeRequired migration
  path), tower-http 0.5 → 0.6, candle 0.9 → 0.10 (vendor was already
  at 0.10.1 — refreshed dep declaration to match).
- **Distributed inference** — Items 1–7 + Item 8 (full architecture
  + cross-over demo: TinyLlama 1.1B GPU loopback ~100 ms slower vs
  control on small prompts; Qwen2.5-7B CPU loopback **12.9× iter-1
  TTFT** on 640-token prompt), Item 12 DSD multi-segment spec
  decoding (flag), Item 13 Q8_0 activation compression (~3.76× wire
  saving, flag), Item 16 Parallax scheduler phases A+B+B.2+C+C.2 on
  by default. Headlines: prefix-cache hit gives 29.4× wall-clock on
  repeat prompts; remote-generate fastpath gives 1.93× decode; Phase
  4 batched chunked prefill gives 1.57× tok/s @ c=4. See
  `docs/plans/distributed_inference_speedup.md`.
- **Cross-platform IPC** — Daemon ↔ model-worker IPC ported from
  Unix-socket-only to the `interprocess` crate (AF_UNIX with 0o600
  perms on Linux/macOS, named pipes with default-DACL on Windows).
  Runtime-validated on Windows (CPU + GPU, single-node + multi-node
  + split shards) on 2026-04-23. See note below.

### Test + lint baseline at v0.1.0

- 816 lib tests passing (782 → 816 since alpha.2), clippy clean both
  with default features and `--no-default-features --features
  dev,claude-subscription`.
- Integration tests green.
- 38/38 M1–M9 + 27/27 V1–V8 curl matrix pass against a live daemon.

### Known deferred items (post-v0.1.0)

- **libp2p 0.55 → 0.56** — ✅ landed post-v0.1.0 (commit `be9c32c`).
  Vendored libp2p-request-response re-ported to upstream 0.29.0 with the
  Tokio watchdog patch re-applied; the `confirmed`-flag patch (gotcha
  #16) is OBSOLETE in 0.29 and was dropped. Vendored libp2p-gossipsub
  removed entirely — upstream 0.49.4 ships the CVE-2026-33040 +
  CVE-2026-34219 fixes our backport carried.
- **`POST /v1/responses/compact`** — V9 of the v2 plan, deferred
  indefinitely until a concrete caller asks for it.
- See `docs/ARCHITECTURE.md` § "Deferred Items" for the full list.

### Notes for upgraders from v0.1.0-alpha.2

- Provider proxy now uses a custom DNS resolver. Hostnames that
  resolve to private/internal IPs (RFC 1918, link-local, loopback)
  are rejected at request time. The pre-flight `validate_provider_url`
  helper still runs for friendlier error messages but is no longer
  the only defense.
- `monitoring/docker-compose.yml` no longer supplies `admin/admin`
  defaults for Grafana. Run `cp .env.example .env && $EDITOR .env`
  before `docker compose up -d`.
- Anyone with a v3 redb file: the existing `Database::open` path
  already backs up the file (`db.redb.bak`) and recreates fresh, so
  in-place upgrade Just Works but loses prior state. New deployments
  start clean on v4.

---

### Detailed change log: alpha.2 → v0.1.0

> Prior tag: `v0.1.0-alpha.1` (2026-03-18, 674 tests) → `v0.1.0-alpha.2` →
> `v0.1.0` (2026-04-25, 816 tests). The summary above is the user-facing
> rollup; this section preserves the per-feature detail captured during
> the alpha → 1.0 grind so reviewers don't have to read 56 commits.

#### Cross-Platform IPC

Daemon ↔ model-worker IPC was Unix-socket-only between 2026-04-18 and
now, silently breaking Windows builds. Ported to the `interprocess`
crate (`local_socket` + tokio): AF_UNIX with 0o600 perms on
Linux/macOS, named pipes with default-DACL (current-logon-session)
on Windows. Security parity, no protocol changes (the framed codec
was already transport-agnostic).

**Runtime-validated on Windows (2026-04-23):** Both the CPU
(`swarmllm-windows-x86_64-cpu.zip`) and GPU
(`swarmllm-windows-x86_64-gpu.zip`) Windows binaries from this release
were smoke-tested on a real Windows host (Ryzen 7 5800H + RTX 3070
Laptop, 8 GB VRAM).

- **CPU single-node:** startup → BLAKE3 shard verify → named-pipe IPC
  handshake → worker subprocess model load → inference (HTTP 200 in
  5 s on TinyLlama 1.1B Q4_K_M) → API-triggered graceful shutdown
  (drain in 3 s, no orphaned worker).
- **CPU multi-node, single-segment:** node B bootstrapped from node A
  over loopback TCP. Noise + Yamux handshake green, GossipSub
  propagated model availability, node B's `/v1/models` listed
  TinyLlama as `owned_by: network`. Cross-node inference via Item 4
  remote-generate fast path: 8 tokens, trust score updated.
- **CPU multi-node, split shards (forced 2-segment pipeline):** A
  hosting shard 0 (layers 0-12), B hosting shard 1 (layers 12-22),
  auto-manage disabled to prevent cross-fill. Pipeline scheduler
  assembled `["A:0-12", "B:12-22"]`, encrypted activation tensor
  (172 KB) shipped over libp2p, B's worker subprocess computed the
  tail layers, response channel returned tokens. Generated coherent
  output ("The capital of France is Paris…") in 6 s.
- **GPU single-node:** dynamic-loading cudarc resolved bundled CUDA
  redist DLLs (cublas64_12.dll, cublasLt64_12.dll, cudart64_12.dll,
  curand64_10.dll, nvrtc64_120_0.dll, nvrtc-builtins64_124.dll) at
  process load — no CUDA Toolkit on the test host. RTX 3070 detected,
  worker subprocess loaded model on `device=Cuda(CudaDevice(1))`,
  inference returned coherent content. Graceful shutdown clean.

**Side-effect validation:** auto-manage's peer-to-peer shard transfer
also exercised on Windows during the multi-node run — A's missing
shard 1 was downloaded from B and vice versa via libp2p
request-response before auto-manage was disabled for the split test.

macOS aarch64 binary remains compile-validated only.

#### Distributed Inference Speedup Arc

A multi-session effort to speed up distributed inference, tracked in
`docs/plans/archive/distributed_inference_speedup.md`. Items 1–16 numbered in
plan order; default-on items landed as they shipped, flag-gated items
are off until benchmarked on real workloads.

**Default-on stack (user-facing in [Performance chapter](docs/book/src/operations/performance.md)):**

- **Item 3 — Continuous batching** (2026-04-19): fused `forward_batch`
  over concurrent Generate requests. 1.34–1.55× GPU throughput at batch
  2–8. CPU falls through to sequential with no regression.
- **Item 4 — Remote-generate fast path**: single-segment distributed
  inference runs the full decode loop on the remote worker instead of
  per-token coordinator round-trips. **1.93× decode speedup**.
- **Item 5 — Cross-request prefix cache**: worker keeps an LRU of prefill
  KV snapshots keyed by prompt prefix. **29.4× wall-clock** on
  re-submission of the same 513-token prompt.
- **Item 7 Phase 1+2 — BatchGenerate + Sarathi chunked prefill**
  (2026-04-19): SlotTable admits concurrent requests, each Prefilling
  slot advances by `prefill_chunk_tokens` (default 128) per decode tick.
  **17–23× TTFT fairness** at concurrency 2/4/8 on RTX 3070 +
  TinyLlama Q4. See `docs/plans/benchmarks/round4.md`.
- **Item 7 Phase 4 — Batched prefill forward** (2026-04-19): fuses
  concurrent same-shape prefill chunks into one `forward_batch`.
  **1.57× aggregate tok/s @ c=4** with uniform 180/180/180 ms TTFT.
  See `docs/plans/benchmarks/round5.md`.
- **Item 8 — Cross-node prefix KV sharing** (2026-04-19/20): when node B
  receives a prompt whose prefix peer A already prefilled, B fetches
  A's KV snapshot over the wire instead of re-prefilling locally.
  Full pipeline: PrefixCacheAnnounce gossip → cross-node index →
  PrefixFetchProbe → trust-gated SendPrefixKvFetch → BLAKE3 verify →
  NaN/Inf scan → hydrate → suffix-prefill.
  **Measured 12.9× iter-1 TTFT speedup on Qwen-7B CPU-CPU localhost**
  (151.7 s → 11.8 s on a 672-token prompt, Round 6 bench 2026-04-20).
  TinyLlama on GPU is the fast-prefill corner case where the fetch path
  is ~100 ms slower than re-prefilling (28 MB snapshot vs 460 ms
  prefill).
- **Item 16 — Parallax scheduler** (2026-04-18/19): shortest-path DP
  over observed per-layer latencies (EMA over recent forwards), replacing
  the greedy latency-only sort. Phase B.2 cross-gossips top-32 observed
  latencies via `NodeCapability.observed_latencies`. Phase C.2 adds a
  soft acquire/prune bias in `AutoShardManager` driven by a per-shard
  stability counter (≥3 consistent ticks before it acts); hard
  constraints (pinning, trust, VRAM) always win.

**Flag-gated:**

- **Item 2 — Distributed speculative decoding** (`speculative_distributed`):
  draft-target speculation across nodes. 40–52% accept rate in a
  llama-cpp-draft / candle-target pairing.
- **Item 6 — SWIFT self-speculative** (`swift_self_speculative`):
  target model acts as its own draft by skipping a layer range. Shelved
  on CPU until flash-attn-with-mask lands.
- **Item 12 — DSD (decentralized speculative decoding)**
  (`decentralized_spec_decoding`, 2026-04-18): multi-segment pipeline
  with γ-token speculation + KV truncation primitives + ~410 LOC
  coordinator loop in `pipeline/dsd.rs`. End-to-end WAN benchmark
  pending.
- **Item 13 — Activation compression Q8_0** (`activation_compression`):
  intermediate pipeline hidden states quantized to Q8_0 on the wire.
  ~3.76× compression, RMS error <0.005. End-to-end multi-segment
  benchmark pending.
- **Item 1 — Persistent pipeline stream** (`persistent_pipeline_stream`):
  one long-lived libp2p bidirectional stream per pipeline session.
  Wire-verified; no measured latency win because the bottleneck was
  elsewhere (Items 4 + 7 solved it).

#### Round 6 Bench Findings (2026-04-20)

The Item 8 two-daemon loopback bench caught three wire bugs before the
measured numbers above landed:

1. `SwarmMessage::PrefixCacheAnnounce` missing from the `TOPIC_MODELS`
   arm in `NetworkManager::handle_broadcast` — Phase 1 announces
   silently dropped at the gossip layer. Loopback self-index path
   masked it in single-node tests.
2. `WorkerMsg::PrefixSnapshotResponse` / `DaemonMsg::PrefixFetchResult`
   carried `payload: Option<Vec<u8>>` inside the JSON-framed IPC header.
   `serde_json` encodes `Vec<u8>` as a JSON array of integers (~5× size
   bloat), so a 28 MB snapshot became a ~102 MB header and blew past
   the 64 MiB `MAX_HEADER` cap.
3. Three chained cross-node-fetch timeouts (`PREFIX_FETCH_TIMEOUT_MS=500`
   in the worker, 400 ms daemon network timeout, 500 ms serving-worker
   IPC timeout) were sized for TinyLlama's 28 MB snapshot. A Qwen-7B
   snapshot is 73 MB and takes ~500–1000 ms to serialize+wire — every
   timeout fired and silently converted real hits into misses. Bumped
   to 3000 / 2500 / 2000 ms respectively, keeping the worker timeout as
   the outer bound.

#### Code Sweep (105 issues found, 58 fixed)
- **Round 1**: 10 parallel review agents across all 109 .rs files — 68 issues (9 CRITICAL, 32 HIGH, 22 MEDIUM), 41 fixed
- **Round 2**: Second pass — 37 new issues (5 CRITICAL, 22 HIGH, 10 MEDIUM), 17 fixed
- Key fixes: max_seq_len 2048 cap, ShardReader cross-tensor bleed, TensorPayload auth, escrow double-charge, IPC framing overflow, API key leak, Gemma embedding scale in forward_batch, hardcoded sampling in distributed forward

#### Credit System Overhaul
- Balanced rates: `rate × tokens` on both earn and spend (no layer multiplier)
- Minimum balance enforcement: `MIN_BALANCE_FOR_INFERENCE = -1000`
- Atomic credit accumulation via `pending_credit_earn` AtomicI64
- Anti-Sybil peer balance deduplication by NodeId
- Priority tiers require positive balance for Gold/Platinum

#### Device Pool Invite Codes
- 8-char one-time codes (e.g., `A3F7K2M9`), 24h expiry, Ed25519 signed
- CLI: `swarmllm pool create/invite-code/join/status/leave`
- API: `/api/pool/generate-code`, `/api/pool/join`, `/api/pool/device-name`, `/api/pool/credit-split`

#### Pool UX Overhaul
- Device nicknames, online/offline status, per-device stats, combined VRAM display
- QR code for invite codes, credit split configuration (0-50%)
- "My Devices" tab with full management UI

#### Terminology Clarification
- "My Devices" vs "Swarm Peers" — clear separation in setup wizard, share popover, dashboard

## [0.1.0-alpha.2] - 2026-03-18

### Release & Scale Readiness (Phase 19)
- **Docker release packaging**: Production `docker-compose.yml` (CPU default, GPU via `--profile gpu`), `.env.example` with all configurable env vars, GitHub Actions CI/CD pushing CPU + CUDA images to GHCR on git tag
- **Docker dev cluster**: 3-node `docker-compose.dev.yml` with static subnet, TCP bootstrap, container-optimized config (`config/docker-cluster.toml`)
- **Setup wizard redesign**: 4 steps → 3 steps (About You → Connect → Ready), invite code paste field, auto-download ON by default, hardware-aware model recommendations based on VRAM, dynamic summary
- **mDNS simultaneous-dial race fix**: When two nodes discover each other via mDNS simultaneously, both connections could fail. Added `pending_redial` queue with hash-based jitter (2-5s) for automatic recovery
- **Upload bandwidth enforcement**: `max_bandwidth_mbps` config now enforced on shard serving with proportional delay (was stored but never applied)
- **Manifest publisher claim**: Copied shard directories now properly gossiped — publisher set to local node_id with manifest hash recomputed on startup shard scan
- **Invite code error messages**: Invalid invite codes now return descriptive 400 errors instead of generic 500
- **Scalability (S1)**: Shard announce delta compression — only broadcasts when shard set changes or periodic re-announce every 10 cycles
- **Scalability (S2)**: P2P shard transfer fallback in auto-manage — when no HuggingFace source known, downloads from peer holders instead of doing nothing
- **Scalability (S3)**: peer_registry capped at 200 entries — evicts highest-latency non-LAN non-pipeline peers when over limit
- **Scalability (S4)**: Gossip broadcast frequency scales with `log(peer_count)` — 30s at ≤10 peers, 120s at 1K, 240s at 10K. Health pings stay at 30s
- **Docker image**: 181MB CPU image (debian:bookworm-slim), multi-stage build, non-root user, health checks
- **Tested**: 5-node Phi-3.5 distribution on Proxmox server (trust promotion, auto-manage, target replicas)

### Local Embedding Privacy
- **Config**: `local_embedding_privacy: true` in `[inference]` — requesting node embeds tokens locally, sends hidden-state activations (not raw token IDs) to remote first-segment nodes
- **LocalEmbedder**: Loaded from shard_000.bin at startup, uses candle for token→embedding conversion
- **Pipeline integration**: Pre-embedded activations skip remote embedding, reducing token exposure to relay nodes

### Deep Code Sweep — 56 fixes across 4 passes (16 parallel review agents)
- **Pass 1 (15 fixes)**: gossip_seal future-epoch removal, manager.rs unwrap safety + shard download cap, AllReduce zstd zip-bomb cap, shard atomic truncate, KV-cache eviction map fix, pipeline pending_vision cleanup + VLM hidden_dim expansion, escrow persist-before-balance, ledger bucket_balance div_euclid, error body truncation, manifest streaming BLAKE3, huggingface u64 range + progress retry, acquisition duplicate guard, protocol pre_embedded defaults
- **Pass 2 (10 fixes)**: sampling order consistency (temperature→top-k→softmax), TP block_in_place wrapper, stale logprobs clear, pending_tensor_channels leak fix (Instant timestamps + periodic sweep), num_layers saturating_sub, protocol unwrap→expect, anthropic proxy error truncation, escrow cleanup balance persist, model_id u16 length guard, max_tokens=0 early return
- **Pass 3 (17 fixes)**: duplicate streaming finish event guard, pending_layer_results failover leak, KV-cache orphan eviction, multi-turn session overwrite prevention, TP tp_size minimum guard (≥2), peer_http_url LAN/Tailscale fix, DB backup-on-upgrade, inverted --shards range validation, tied_output_weight streaming read, pool cosign cryptographic separation, pool_registry cleanup, escrow cleanup count, gossip epoch bound (reject >2 epochs old), invite code decrypt-fail no fallthrough, auto_update download gate, multi-image VLM remote guard, dead code removal
- **Pass 4 (14 fixes)**: completion_tokens EOS corruption (used clean_tokens.len → generated_tokens.len), keystore 0o600 permissions on private key files, WebSocket connection counter RAII guard, credit earn crash-window (single persist after earn+forwarding), MCP error body API key scrubbing, all_shards_available cache cap (1000 entries), TP attention GQA modulo wrap for tp_size > n_kv_head, AllReduce duplicate rank warning, pool state db.remove() on leave (fixes null deserialization type mismatch), health monitor future-timestamp evasion (clamp to zero), supervisor dead code cleanup (MAX_RESTART_ATTEMPTS), GGUF total_size saturating_add (overflow on malicious headers), gossip epoch fallback tightened (3→2 epoch window)
- **Model/storage (6 fixes)**: huggingface total_size==0 guard, retry HTTP status check, atomic tmp+rename for mmproj/header, LoRA rank==0 guard, auto_manage path traversal sanitization

### Feature Wiring — 8 previously unwired features now fully integrated
- **Priority tier enforcement**: `calculate_tier()` with real network percentile from peer credit gossip; `max_concurrent_for_tier()` enforces per-tier concurrent request limits in `drain_queue()`
- **Apply penalty on failure**: Credit penalty (configurable `penalty_serve_failure`, default -50) applied on distributed inference failure; penalty uses `apply_credit_direct` for immediate balance update
- **AllReduce registry cleanup**: `cleanup_stale()` removes entries where the receiver was dropped (timed out), wired into HealthMonitor's periodic 30s tick
- **Pipeline error broadcast**: `broadcast_pipeline_error()` notifies all pipeline participants on distributed inference failure, enabling peers to update shard availability
- **Pipeline affinity (KV cache reuse)**: Multi-turn sessions reuse previous pipeline assignment when all nodes are still connected, avoiding cold KV-cache on every turn
- **Relay service credits**: Tracks relay circuit open/close times in SharedState (`active_relay_circuits` DashMap), accumulates seconds in `relay_seconds_served` atomic counter, drains periodically in CreditLedger to `earn_relay_service()`
- **DHT record verification**: `verify_dht_value()` Ed25519 signature check on all Kademlia `GetRecordOk` results in NetworkManager — unsigned/invalid records are logged and ignored
- **Logprobs in API response**: `sample_token_with_params_and_logprobs()` in tensor_util collects per-token log probabilities via `SamplingContext`, stored in `PipelineExecutor.collected_logprobs` (Mutex), mapped to OpenAI-compatible `ChoiceLogProbs` in the `/v1/chat/completions` response. Works for split model (candle) inference paths

### Security Audit (Phase 16) — ~90 fixes across 5 rounds
- **Round 1-3**: Mandatory gossip signing, transport-authenticated dispatch, RFC 6479 anti-replay, signed DHT records, ephemeral key auth, path traversal fix, HF input validation, constant-time auth, CSP hardening, rate limiter cleanup, queue caps, input limits, WebSocket Origin validation, credit signature verification, XSS fixes
- **Round 4**: StreamingToken auth guard, peer IP bypass scoped to inference paths only, `.env` loader blocks dangerous env vars (LD_PRELOAD/PATH/DYLD_*), TOCTOU guard via `loading_models` DashMap with RAII `LoadGuard`, metadata hostname blocklist (Azure/AWS/DO/Oracle/Alibaba), IPv6 multiaddr extraction
- **Round 5**: All dispatch handlers require `authenticated_sender` (LayerResult, InferenceRequest, PipelineAssignment, InferenceError, TpAllReduceResponse), plaintext fallback removed (seal failure → drop), PEX SSRF filter (private/loopback/link-local IPs), shard serve requires peer_registry membership, pending_tensor_channels capped at 256, pending_tp_partials capped at 512, image_data 20MB cap, PoolMessage identity binding (CreditForward.from_node_id/MemberLeft.node_id must match sender), tool params size limits, lora_adapter validation, peer error body truncation/scrubbing, invite code capped at 4K, MCP research restricted to local/network models

### Frontend Polish
- CSS: removed unused variables, fixed hardcoded colors → CSS vars, removed duplicate rules, `@media (prefers-reduced-motion)`, light theme semantic color overrides
- Accessibility: `role="alert"` + `aria-live` on WebSocket banner, `aria-expanded` on hamburger, `aria-live="polite"` on chat messages, `scope="col"` on table headers
- JS: replaced inline styles with CSS classes, wired aria-expanded toggle

### Bug Fixes (Post-Audit)
- **Critical**: Credit balance overflow (`i64 +=` → `saturating_add`) and missing persistence in `track_forward_participation` — credits now survive daemon restart
- **High**: Divide-by-zero panic from malformed GGUF with `head_count == 0` (4 sites, remotely triggerable via HF probe)
- **High**: Silent null body sent to cloud provider on serialization failure (`unwrap_or_default` → proper error propagation)
- **Medium**: `model_request_counts` DashMap unbounded growth — now gated on registered models only
- **Medium**: `peer_shard_downloads` orphaned entries on peer disconnect — cleanup in ConnectionClosed handler
- **Medium**: Rate limiter cleanup task ignoring shutdown signal — now uses `tokio::select` with `shutdown_rx`

### Infrastructure
- **Workspace migration**: 3-crate Cargo workspace (`swarmllm`, `swarmllm-types`, `swarmllm-frontend`)
- **Ring AllReduce**: Bandwidth-optimal for ≥4 TP ranks, auto-selected by `choose_allreduce_strategy()`
- **Package distribution**: Homebrew formula, AUR PKGBUILD, deb/rpm packages, systemd service file
- **macOS CI**: Re-enabled on macos-15 runner
- **Docker**: Fixed Dockerfiles for workspace build
- 674 tests passing (606 unit + 22 integration + 31 module + 14 yamux + 1 VLM E2E)

### UX & Internationalization
- **i18n** — 20 languages (Arabic, Chinese, Czech, Dutch, English, French, German, Hindi, Indonesian, Italian, Japanese, Korean, Polish, Portuguese, Russian, Spanish, Swedish, Thai, Turkish, Ukrainian, Vietnamese)
- **Theme toggle** — Light / Dark / System theme with persistent preference
- **Basic/Advanced mode** — Toggle for simplified vs power-user UI
- **Plain-English UX pass** — Removed jargon, clearer labels and error messages for beginners
- **Compare UX** — Prompt textarea moved out of collapsed section, All/Local/Cloud filter buttons, chat source indicators, tok/s display fix for slow models (shows 0.5 instead of 0)
- **Provider UX** — `.env` file support for API keys, key source selector (auto/env/dashboard), error badges with click-to-settings
- **GPU OOM → CPU fallback** — Models that exceed GPU VRAM automatically retry on CPU (split fast-path preserved, not slow pipeline path)
- **Anthropic API model routing fix** — Requests now route to the correct model instead of always using the first loaded model

### Codebase Quality
- **Refactored**: `daemon.rs` (4015 lines → module directory), `admin.rs` (4225 lines → 4 modules), `split.rs` (10K lines → 6 modules)
- **Extracted**: `swarmllm-frontend` crate with dev mode for instant UI changes without full rebuild
- 674 tests passing (606 unit + 22 integration + 31 module + 14 yamux + 1 VLM E2E)

### Model Trust & On-Demand Loading (Phase 14)
- **Model Trust System** — demand-driven trust prevents trash models from auto-propagating
  - `ModelTrustLevel` enum: Discovered → Pinned → DemandVerified → NetworkPopular
  - Auto-manage only downloads shards for `DemandVerified`+ or user-`Pinned` models
  - Models promoted to `DemandVerified` after 3 real inference requests
  - `NetworkPopular` promotion when 3+ unique holder nodes serve a model
  - 7-day inactivity decay (Pinned models immune), persisted to redb
  - Trust level exposed in admin API (`trust_level` field on all model objects)
- **On-Demand Shard Loading** — inference requests trigger auto-loading from disk
  - Router detects shards on disk but not loaded in VRAM, triggers `check_and_load_model()`
  - LRU eviction makes room automatically (protected: active pipeline models)
  - Loading coordination via `DashMap<ModelId, Notify>` prevents concurrent loads
  - No more need to pre-load all models at startup
- **Kimi 2.5 support** — `k2*` prefix routing to Moonshot provider
  - Static fallback models: kimi-k2-0527, moonshot-v1-8k/32k/128k
  - Existing kimi* and moonshot-* routing preserved
- **UI improvements**
  - Trust level badges on model cards: Popular (green), Verified (accent), Pinned (yellow), Unverified (gray)
  - HF browser: prominent "On Swarm — N nodes" badge vs "New to network"
  - Download button renamed to "Add to node" (clarifies seed shard semantics)
  - "Local only" indicator when no peers host the model
- **Storage**: `get_all_json()` method on Database for key-value iteration with subkeys
### Claude Code Integration (Phase 13)
- **Full Anthropic Messages API** (`POST /v1/messages`) — complete Claude Code compatibility
  - `tools`, `tool_choice`, `metadata`, `thinking` (extended thinking) request fields
  - `tool_use`, `tool_result`, `thinking`, `redacted_thinking` content blocks
  - `cache_control` on system blocks (Anthropic prompt caching)
  - Full pass-through to Anthropic cloud (all fields preserved including tools and thinking)
  - Anthropic→OpenAI translation proxy for non-Claude cloud models (GPT-4o, DeepSeek, etc.)
  - Tool calls and thinking blocks converted to text for local GGUF inference
  - `ResponseContentBlock` refactored from struct to enum (Text, ToolUse, Thinking variants)
- **MCP `compare` tool** — send same prompt to multiple models concurrently (up to 10)
  - Returns side-by-side results with `content`, `latency_ms`, `input_tokens`, `output_tokens`, `status`
  - Supports local, network, and cloud models in same comparison
  - Routes through `/v1/messages` for consistent routing logic
- **Claude Code as client**: `ANTHROPIC_BASE_URL=http://localhost:8800 claude --model qwen2.5-coder-7b`
- **Model Compare dashboard page** — side-by-side multi-model comparison UI with streaming
- 6 new unit tests (tool_use, tool_result, thinking, tools request, response serialization, internal conversion)
- 665 tests passing (597 unit + 22 integration + 31 module + 14 yamux + 1 VLM E2E)

### Published Benchmark Data
- **GPU (RTX 3070 8GB):** TinyLlama 1.1B 27.2 tok/s, Gemma-2 2B 20.6 tok/s, Phi-3.5 3.8B 46.4 tok/s, Qwen2.5 7B 29.0 tok/s
- **CPU (Ryzen 7 5800H):** TinyLlama 4.2 tok/s, Gemma-2 3.5 tok/s, Phi-3.5 1.8 tok/s, Qwen2.5 2.4 tok/s
- GPU speedups: 6.5x to 25.8x depending on architecture
- Methodology: 100 output tokens, 3-run average, single model loaded, Q4_K_M quantization

## [0.1.0-alpha.1] — 2026-03-07

First public release. Single Rust binary (~31MB) for decentralized P2P LLM inference.

### Inference Engine
- **11 model architectures**: Llama, Llama 4, Qwen2, Qwen 3.5 (hybrid SSM+attention), Gemma/2, Phi-3, Mistral, Starcoder2, DeepSeek-V2/V3 (MoE+MLA), GLM-4
- **4 architectures verified** with real models: Llama (TinyLlama-1.1B), Qwen2 (Qwen2.5-Coder-7B), Phi-3 (Phi-3.5-mini), Gemma2 (Gemma-2-2B-IT)
- **Distributed inference** verified on 2-node real LAN (WSL2 laptop + Proxmox server) with 5 models, crash recovery, auto-reconnect
- **Tensor parallelism** via AllReduce (star topology) with RTT-based LAN peer detection
- **VLM support**: LLaVA-v1.5-7B verified end-to-end (CLIP vision encoder + correct fine-tuned text model from second-state/Llava-v1.5-7B-GGUF), distributed mmproj, chat UI image upload (camera button, paste, drag-drop)
- **LoRA adapters**: per-request loading, verified with Qwen2.5-Coder-7B + rank-16 adapter
- **Speculative decoding** with draft model + rejection sampling
- **Cross-request batching** (GPU batch tensors, configurable `max_batch_size`)
- **Multi-turn KV-cache** with session reuse, cross-request prefix caching, chunked prefill
- **Flash attention** (CPU + GPU) and **paged attention** (CUDA block pool)
- **Structured output**: ResponseFormat API with JSON grammar state machine + schema validation
- **Sampling**: temperature, top-k, top-p, frequency/presence penalty, stop sequences

### API & Compatibility
- **OpenAI-compatible API**: `POST /v1/chat/completions` with streaming (SSE), `tool_calls`, `tool_choice`, `logprobs`, `top_logprobs`, Tool role
- **Anthropic Messages API**: `POST /v1/messages` — full Claude Code compatibility (tools, tool_choice, thinking, cache_control, metadata)
- **MCP server** at `/mcp` — `chat`, `models`, and `compare` (multi-model comparison) tools for Claude Code, Cursor, and MCP-compatible agents
- **12 cloud provider fallback**: OpenAI, Anthropic, DeepSeek, Mistral, Groq, NVIDIA NIM, Cerebras, SambaNova, Fireworks, Together, DeepInfra, Moonshot/Kimi
- **Hidden states API**: `/v1/internal/hidden-states` for research (activation inspection, adapter insertion)
- **Embeddings**: `POST /v1/embeddings`
- **~62 admin REST routes** for dashboard, config, model management, downloads, providers
- **WebSocket** live updates (2s stats + prune event notifications)
- **Prometheus metrics** at `/metrics` (6 gauges + histogram)

### SDKs & Integrations
- **Python SDK**: `pip install swarmllm-client` — sync + async clients, streaming
- **JavaScript/TypeScript SDK**: zero-dependency, streaming support
- **LangChain integration**: `ChatSwarmLLM` provider
- **LlamaIndex integration**: `SwarmLLM` provider
- **Benchmark CLI**: `swarmllm bench` — sequential latency + concurrent throughput, JSON output

### Networking
- **P2P**: libp2p 0.55 with TCP+Yamux (primary) and QUIC transport
- **5-layer discovery**: mDNS (LAN), persistent peer cache (redb), encrypted invite codes, peer exchange (PEX), Kademlia DHT
- **NAT traversal**: libp2p relay circuits + DCUtR hole punching
- **GossipSub**: 6 topics for shard announcements, credit gossip, health, governance
- **Unified protocol**: `/swarmllm/1.0.0` — JSON control messages + binary tensor payloads (type-tag byte)
- **Wire compression**: zstd for tensor payloads

### Security
- **E2E encryption**: X25519 key exchange + ChaCha20-Poly1305 symmetric encryption
- **Forward secrecy**: ephemeral re-keying with key rotation
- **Sealed gossip**: all gossip messages authenticated (no plaintext fallback)
- **Replay protection**: nonce tracking + rejection
- **Shard integrity**: BLAKE3 content hash verified on every load
- **API auth**: Bearer token with auto-generation, loopback-only key retrieval
- **Provider key security**: at-rest encryption (AES-GCM), zeroize on drop, log scrubbing
- **Content-Security-Policy** header, IP-based rate limiting, CORS lockdown, SSRF protection
- **KV-cache privacy mode**: configurable per-session data isolation

### Model Management
- **Shard-only operation**: nodes download individual shards (~512MB each), never need a full model
- **HuggingFace integration**: search, browse, byte-range shard downloads with resume/retry
- **VRAM-aware auto shard management**: rarity-scored acquisition, popularity-based scoring
- **Smart shard pruning**: auto-remove over-replicated shards based on demand, resource pressure, and region diversity
- **Per-shard lock/pin** and per-model prune toggle
- **BLAKE3 integrity verification** on every shard load

### Credit System
- **Credit ledger**: earn credits by serving inference, hosting shards, seeding data
- **4 priority tiers**: Platinum (top 10%), Gold (top 30%), Silver (positive), Bronze (zero/negative)
- **Dual-signed transactions**: Ed25519 signatures from both parties
- **Credit escrow** for large requests
- **Anti-gaming**: rate limits, spot-check verification, subnet clustering detection
- **Sybil resistance**: trust scoring with decay, reputation tracking

### Identity & Pools
- **Ed25519 cryptographic identity** per node
- **Nicknames** with leaderboard
- **Device pools**: multi-device credit pooling with dual-signature invitation protocol

### Frontend
- **Embedded web dashboard** (vanilla HTML/CSS/JS, no build step, < 200KB)
- **4-step setup wizard** for first-run experience
- **Chat interface**: multi-turn + streaming, switchable Linear/Messenger layout, image upload (camera button, paste, drag-drop) for VLM models
- **Model browser**: HuggingFace search, shard grid visualization, download progress
- **Network map**: peer visualization with region grouping
- **Mobile-responsive** layout with dark theme
- **Reasoning model support**: DeepSeek R1 think token rendering

### Operations
- **Single binary**: ~31MB, zero runtime dependencies
- **CLI**: `run`, `status`, `chat`, `bench`, `peers`, `test-split`, `version`
- **Config priority**: CLI flags > env vars (`SWARMLLM_` prefix) > config.toml > defaults
- **Config hot-reload** via SIGHUP or API
- **Graceful shutdown**: SIGTERM handler with subsystem drain
- **Auto-updater**: checks GitHub Releases, downloads + self-replaces with restart prompt
- **JoinSet task supervisor**: automatic restart-on-crash for all 10 subsystems
- **Database**: redb v3 (embedded, ACID, ~15% faster than v2)

### Platform Support
- Linux x86_64 (CPU + CUDA + ROCm)
- macOS aarch64 Apple Silicon (Metal)
- macOS x86_64 Intel (CPU)
- Windows x86_64 (CPU + CUDA)

### Model Loading
- **Auto-extract gguf_header.bin** from shard_000.bin when header is missing (daemon pre-pass)
- **Single-GGUF manifest**: Full GGUF files stored as shard_000.bin generate 1-shard manifests (not split into logical shards)
- **Single-shard mmap fallback**: Models with 1 shard and no tensor entries load via mmap instead of ShardReader
- **Probed flag fix**: Models with gguf_header.bin correctly show as probed in admin API

### Test Suite
- 659 tests: 591 unit + 22 integration + 31 module + 14 yamux + 1 VLM E2E
- All passing, clippy clean, rustfmt clean
- CI: GitHub Actions (fmt → clippy → test → build)

[0.1.0-alpha.1]: https://github.com/enapt/SwarmLLM/releases/tag/v0.1.0-alpha.1
