## 0.6.0 + SwarmLLM patches (not upstream)

- igd-next 0.18 instead of 0.16 (2026-10-23): 0.16 could not reach a router that lists an
  absolute control URL.
- The EXTERNAL port is chosen (2026-10-24, `src/choice.rs`): the router's table is read
  (`GetGenericPortMappingEntry`) and a port another computer holds is never asked for; the
  port a mapping holds is kept across renewals and is the one a removal deletes — and only
  while the table still says it is ours. A refusal (718, or 501 as libtorrent reads it) moves
  to the next candidate. A router that silently REPLACED another client's mapping handed
  8810/8800 to whichever of two nodes renewed last, while both advertised it.
- A row forwarding to ANY of this computer's listen addresses at the same internal port is
  ours (`Ours::own_ips`): a node before this patch mapped WSL's 10.255.255.254 too and the
  router kept that row past its lease, so read as another computer's it would push the node
  off its own port for good.
- A removal reads the table only up to its own row (`read_table_until`).
- A read of the table ends only after 16 "no such entry" (713) answers in a row
  (`choice::TableRead`): the 2026-10-24 router keeps rows in fixed slots and answers 713 for an
  empty one, so after its slot 0 was deleted the listing read as empty while six rows remained —
  another computer's port would have read as free.
- Only a listen address on the router's network is mapped (the local address the router is
  reached from — miniupnpc's `lanaddr`); upstream also asked for WSL's 10.255.255.254.
  Addresses seen before the router answers wait in `waiting` and one per listener is picked
  when it does (`place`): the mapping table keeps ONE per listener and an insert keeps the
  first key, so a listener whose first address was WSL's would otherwise go unmapped. An
  expired listen address removes a mapping only when it is the address mapped.
- A fired renewal timer is reset when the renewal is sent: a fired `Delay` stays ready, so
  every poll until the reply queued another renewal.
- No more "mapping should exist" panic (libp2p #6377; 0.7 fixed it with PR 6459's rework, which
  needs libp2p-swarm 0.48). A mapping whose address went is `Removing`, and only the reply to
  that removal ends it; a reply that outlived its address changes nothing (a `Mapped` one hands
  the row back); the entry's key is replaced with its address (`MappingList::set` —
  `HashMap::insert` kept the old one). So an address that goes and comes back while its
  removal is in flight (an interface flap) is mapped again, and a listener closing with
  several addresses removes its one mapping once. Tests drive both sequences.
- One mapping per listener: another address of a mapped listener is not asked for (where the
  router-facing one is unknown it would take the first one's port on a router that replaces).
- A mapped address that goes is expired with the swarm; upstream left it confirmed.
- A removal the router fails is asked for 3 times in all, then left to the lease (upstream:
  again at once, for ever).
- Every reply waiting is handled in one poll; one that produced nothing left the receiver
  un-polled, so the next reply woke nobody.
- The router-facing address is read again for a listen address that is not it (a DHCP change,
  Wi-Fi to cable); it was fixed at the search.
- A listen address not mapped now — another of a mapped listener's, or one not facing the
  router yet — is kept until it goes, and mapped when its listener's mapping ends
  (`map_a_successor`): a new address can arrive BEFORE the old one goes (a cable plugged in
  while Wi-Fi is up), and upstream dropped it as "already mapped", so once the old one went
  the listener stayed unmapped.
- A router that does not ANSWER a read of its table (a timeout, a dropped connection; each
  index is asked twice) is not read as one that will not list: that falls back to asking for
  the internal port, another computer's on a router that replaces. The mapping or removal
  fails and is asked again later. Only an error the router answers with, at the first index,
  means it will not list (`choice::Table`).
- A gateway not found is searched for again after 2 minutes, doubling to 30 (FUTURE_WORK
  #329); listen addresses seen meanwhile are mapped when one is found.

## 0.6.0

- Change `Event::NewExternalAddr` and `Event::ExpiredExternalAddr` from tuple variants to struct variants
  that include both local and external addresses. This allows users to correlate which local listen
  address was mapped to which external address.
  - `Event::NewExternalAddr` now contains `local_addr` and `external_addr` fields
  - `Event::ExpiredExternalAddr` now contains `local_addr` and `external_addr` fields
  See [PR 6121](https://github.com/libp2p/rust-libp2p/pull/6121).

- Skip port mapping when an active port mapping is present.
  Previously, the behavior would skip creating new mappings if any mapping
  (active or inactive or pending) existed for the same port. Now it correctly only
  checks active mappings on the gateway.
  See [PR 6127](https://github.com/libp2p/rust-libp2p/pull/6127).

- Fix excessive retry attempts for failed port mappings by implementing exponential backoff.
  Failed mappings now retry up to 5 times with increasing delays (30s to 480s) before giving up.
  This prevents continuous retry loops.
  See [PR 6128](https://github.com/libp2p/rust-libp2p/pull/6128).

## 0.5.0

- update igd-next to 0.16.1
  See [PR 5944](https://github.com/libp2p/rust-libp2p/pull/5944).

- Fix panic during a shutdown process.
  See [PR 5998](https://github.com/libp2p/rust-libp2p/pull/5998).

<!-- Update to libp2p-swarm v0.47.0 -->

## 0.4.0

- update igd-next to 0.15.1.
  See [PR XXXX](https://github.com/libp2p/rust-libp2p/pull/XXXX).

<!-- Update to libp2p-core v0.43.0 -->

## 0.3.0

<!-- Update to libp2p-swarm v0.45.0 -->

## 0.2.2
- Fix a panic caused when `upnp::Gateway` is dropped and its events queue receiver is no longer
available.
  See [PR 5273](https://github.com/libp2p/rust-libp2p/pull/5273).

## 0.2.1
- Fix a panic caused when dropping `upnp::Behaviour` such as when used together with `Toggle`.
  See [PR 5096](https://github.com/libp2p/rust-libp2p/pull/5096).

## 0.2.0


## 0.1.1

- Fix high CPU usage due to repeated generation of failure events.
  See [PR 4569](https://github.com/libp2p/rust-libp2p/pull/4569).

- Fix port mapping protocol used for a UDP multiaddress.
  See [PR 4542](https://github.com/libp2p/rust-libp2p/pull/4542).

## 0.1.0

- Initial version
