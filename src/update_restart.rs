//! Draining and restarting into a freshly-applied binary.
//!
//! Replacing the binary on disk does not change the running process: it goes on
//! executing the old image, and `current_exe()` starts reporting
//! `".../swarmllm (deleted)"` (gotcha #188). Until v0.3.39 that also broke every
//! inference on the node, because worker spawning used that path — so a node
//! that had "updated" kept advertising its shards while being unable to serve
//! any of them. The fix made the daemon survive it; this module removes the
//! reason to be in that state at all.
//!
//! A tester hit the visible half of this on 2026-07-28: `swarmllm update`
//! replaced the binary, the daemon kept reporting the old version, and only a
//! manual SIGTERM and relaunch actually applied it. The dashboard button said
//! "Apply & Restart" while the code deliberately did not restart.

use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::daemon::SharedState;

/// How long to wait for in-flight work to finish before restarting anyway.
///
/// Generous because the work being waited on is somebody's answer: a long
/// prompt on a CPU node can legitimately take minutes (prefill is ~99% of a
/// long request), and cutting it off to install an update is a worse outcome
/// than installing a few minutes later. Bounded because a wedged request must
/// not defer an update forever.
pub const DRAIN_TIMEOUT: Duration = Duration::from_secs(600);

/// How often to re-check whether the node has gone idle.
const DRAIN_POLL: Duration = Duration::from_secs(2);

/// Wait until this node is neither running nor serving any inference.
///
/// Returns `true` if the node went idle, `false` if [`DRAIN_TIMEOUT`] expired
/// first (the caller decides whether that is still worth restarting for).
///
/// Consults BOTH `active_pipelines` and `serving_models`, and the second is the
/// one that is easy to forget: `active_pipelines` is the *coordinator's* map
/// and never contains work this node is doing on a peer's behalf, so a node
/// that does nothing but answer other people looks permanently idle through it
/// alone (gotcha #194 — that exact blind spot got a worker killed mid-answer).
pub async fn drain(state: &Arc<SharedState>) -> bool {
    let started = Instant::now();
    loop {
        let coordinating = state.active_pipelines.len();
        let serving = state.serving_models.len();
        if coordinating == 0 && serving == 0 {
            return true;
        }
        if started.elapsed() >= DRAIN_TIMEOUT {
            tracing::warn!(
                coordinating,
                serving,
                timeout_secs = DRAIN_TIMEOUT.as_secs(),
                "Still busy after the drain window — restarting into the update anyway"
            );
            return false;
        }
        tokio::time::sleep(DRAIN_POLL).await;
    }
}

/// Carries the process id of the build that spawned this one, so a replacement
/// can wait for its predecessor to let go before it tries to take over.
///
/// Only ever set by [`exec_into`] on Windows, and always to the CURRENT process
/// id — never inherited unchanged, or a node that updated twice would wait on
/// the id of a process two generations back (and, after id reuse, on whatever
/// holds it now).
pub const HANDOFF_PID_VAR: &str = "SWARMLLM_UPDATE_HANDOFF_PID";

/// Longest a replacement waits for its predecessor. Generous relative to what
/// it is waiting for — a process that has already called `exit` — and bounded
/// because waiting forever on a predecessor that will not die is worse than
/// starting and reporting the port conflict.
const HANDOFF_WAIT: Duration = Duration::from_secs(30);

/// How often to re-check.
const HANDOFF_POLL: Duration = Duration::from_millis(100);

/// Wait for the build that spawned this one during an update to exit.
///
/// **Unix does not need this and never calls it.** There, `exec_into` replaces
/// the process image: same pid, same open files, so nothing is ever held twice.
/// Windows has no `exec`, so the replacement is a SECOND process that starts
/// while the first is still exiting — and the two things it needs, the API port
/// and the redb lock, are both exclusive and neither is retried. Losing that
/// race costs the user their node: the predecessor has gone, the replacement
/// refuses to start, and nothing tries again.
///
/// Ordinary starts are unaffected — with no handoff variable set this returns
/// immediately, so the `Port … is already in use` message still means what it
/// has always meant.
pub fn await_predecessor_exit() {
    let Ok(raw) = std::env::var(HANDOFF_PID_VAR) else {
        return;
    };
    // Deliberately NOT removed from the environment afterwards. It is tempting
    // — worker subprocesses inherit it and it means nothing to them — but
    // `std::env::remove_var` mutates a global while every Tokio worker thread
    // is already running, which is the hazard that made it `unsafe` in edition
    // 2024. It buys nothing here either: `exec_into` writes this variable
    // afresh on every handoff, so a value from an earlier generation can never
    // be the one acted on.
    let Ok(pid) = raw.trim().parse::<u32>() else {
        tracing::warn!(value = %raw, "Ignoring an unreadable update handoff process id");
        return;
    };

    tracing::info!(
        predecessor_pid = pid,
        "Started by an update — waiting for the previous version to finish exiting"
    );
    let waited = wait_while(
        || process_is_running(pid),
        HANDOFF_WAIT,
        HANDOFF_POLL,
        std::time::Instant::now,
    );
    match waited {
        Some(elapsed) => tracing::info!(
            waited_ms = elapsed.as_millis() as u64,
            "The previous version has exited — starting"
        ),
        None => tracing::warn!(
            predecessor_pid = pid,
            timeout_secs = HANDOFF_WAIT.as_secs(),
            "The previous version is still running. Starting anyway; if it still holds the API \
             port or the database this node will say so and stop, and starting it again by hand \
             will work"
        ),
    }
}

/// A socket this node is about to claim, which the build it replaced held.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PortClaim {
    Tcp(std::net::SocketAddr),
    Udp(std::net::SocketAddr),
}

impl PortClaim {
    /// Could this node bind it right now? Binds and lets go at once.
    fn is_free(&self) -> bool {
        match self {
            PortClaim::Tcp(addr) => std::net::TcpListener::bind(addr).is_ok(),
            PortClaim::Udp(addr) => std::net::UdpSocket::bind(addr).is_ok(),
        }
    }
}

/// Set on a replacement that has already relaunched itself without inherited
/// handles, so it never does so twice.
#[cfg(windows)]
const CLEAN_RELAUNCH_VAR: &str = "SWARMLLM_UPDATE_CLEAN_RELAUNCH";

/// After an update handoff, make sure the ports the previous version held can be
/// bound — called once the configuration has said which ports those are.
///
/// **On Windows the replacement itself held the old version's QUIC port** (#769,
/// measured 2026-10-01): `exec_into` up to v0.3.217 started it with std's
/// `Command`, which hands the child every inheritable handle — and the old QUIC
/// socket was one. Nine seconds after the old process was gone, UDP 8950 was still
/// registered to it; killing the replacement freed it 99 ms later. So the
/// replacement could never bind it, waited or not: every automatic update on
/// Windows ended with the node stopped (v0.3.204 → .216 and → .217, 3 of 3).
/// (`Get-NetUDPEndpoint` names the process that CREATED a socket, which is why it
/// first read as "the port outlives the old process".)
///
/// So a replacement that finds a port held relaunches itself once with no
/// inherited handle and exits; its exit closes the inherited socket, and the
/// relaunched node waits for it as for any predecessor. That is what rescues a
/// swap done by an OLDER version — which still hands the socket down — and
/// `exec_into` no longer passes it at all. The bounded wait stays for anything
/// else holding a port.
pub fn await_ports_released(claims: &[PortClaim]) {
    if std::env::var_os(HANDOFF_PID_VAR).is_none() {
        return;
    }
    #[cfg(windows)]
    if std::env::var_os(CLEAN_RELAUNCH_VAR).is_none() && claims.iter().any(|c| !c.is_free()) {
        let args: Vec<std::ffi::OsString> = std::env::args_os().skip(1).collect();
        let me = std::process::id().to_string();
        match std::env::current_exe().and_then(|exe| {
            spawn_without_inherited_handles(
                &exe,
                &args,
                &[(HANDOFF_PID_VAR, me.as_str()), (CLEAN_RELAUNCH_VAR, "1")],
            )
        }) {
            Ok(pid) => {
                tracing::info!(
                    relaunched_pid = pid,
                    still_held = ?claims.iter().filter(|c| !c.is_free()).collect::<Vec<_>>(),
                    "A port is held by a handle this process inherited from the version it \
                     replaced — relaunching without inherited handles"
                );
                std::process::exit(0);
            }
            Err(e) => tracing::warn!(
                error = %e,
                "Could not relaunch without inherited handles — waiting for the ports instead"
            ),
        }
    }
    match wait_for_ports(claims, HANDOFF_WAIT, HANDOFF_POLL) {
        Some(elapsed) => tracing::info!(
            waited_ms = elapsed.as_millis() as u64,
            "The previous version's ports are free — starting"
        ),
        None => tracing::warn!(
            still_held = ?claims.iter().filter(|c| !c.is_free()).collect::<Vec<_>>(),
            timeout_secs = HANDOFF_WAIT.as_secs(),
            "A port this node needs is still taken after the update handoff. Starting anyway; if it \
             is still taken this node will say so and stop, and starting it again by hand will work"
        ),
    }
}

/// Wait until every claim can be bound; how long that took, or `None` at the
/// timeout. Apart from the handoff variable so it can be tested on real sockets.
fn wait_for_ports(claims: &[PortClaim], timeout: Duration, poll: Duration) -> Option<Duration> {
    wait_while(
        || claims.iter().any(|c| !c.is_free()),
        timeout,
        poll,
        Instant::now,
    )
}

/// Is a process with this id running?
fn process_is_running(pid: u32) -> bool {
    let mut sys = sysinfo::System::new();
    let pid = sysinfo::Pid::from_u32(pid);
    sys.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
    sys.process(pid).is_some()
}

/// Poll `busy` until it answers false, and say how long that took — or `None`
/// if it never did.
///
/// The clock is a parameter so this can be tested without one: what is being
/// asserted is the loop's shape, and a test that waits out real timeouts to
/// check a timeout is a test nobody runs twice.
fn wait_while(
    mut busy: impl FnMut() -> bool,
    timeout: Duration,
    poll: Duration,
    now: impl Fn() -> Instant,
) -> Option<Duration> {
    let started = now();
    loop {
        if !busy() {
            return Some(now().saturating_duration_since(started));
        }
        if now().saturating_duration_since(started) >= timeout {
            return None;
        }
        std::thread::sleep(poll);
    }
}

/// Replace this process with the binary at `exe`, keeping the original argv.
///
/// On Unix this is `execv`: same PID, same parent, same file descriptors, so it
/// works identically under systemd (no `Restart=` policy involved — the service
/// never exits), under a plain `swarmllm run` in a terminal, and under a
/// process supervisor. Exiting and hoping something restarts us would only work
/// for the first of those, and the packaged unit is `Restart=on-failure`, which
/// deliberately does not restart a clean exit.
///
/// Only returns on failure — success never returns.
pub fn exec_into(exe: &std::path::Path) -> std::io::Error {
    let args: Vec<std::ffi::OsString> = std::env::args_os().skip(1).collect();
    tracing::info!(exe = %exe.display(), "Restarting into the updated binary");

    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        std::process::Command::new(exe).args(&args).exec()
    }

    #[cfg(windows)]
    {
        // Windows has no exec: spawn a replacement and let this process exit.
        // The new process keeps the console, so an interactive user keeps their
        // window. A service wrapper sees the old process exit cleanly.
        //
        // Unlike `exec`, this leaves two processes alive for an instant, both
        // wanting the API port and the redb lock. `HANDOFF_PID_VAR` is how the
        // replacement knows to wait for this one — see `await_predecessor_exit`.
        // And it passes NO handle but stdio: std's `Command` handed the
        // replacement our QUIC socket, which it then could not bind (#769).
        let me = std::process::id().to_string();
        match spawn_without_inherited_handles(exe, &args, &[(HANDOFF_PID_VAR, me.as_str())]) {
            Ok(_) => {
                tracing::info!("Replacement process spawned — exiting");
                std::process::exit(0);
            }
            Err(e) => e,
        }
    }

    #[cfg(not(any(unix, windows)))]
    {
        let _ = args;
        std::io::Error::other("restarting into an update is not supported on this platform")
    }
}

/// Start `exe` with `args` and the given environment additions, passing it NO
/// handle of ours except standard input, output and error (#769).
///
/// std's `Command` always calls `CreateProcessW` with `bInheritHandles = TRUE`,
/// so the child receives every inheritable handle in the process; on Windows
/// that included the QUIC socket, despite `socket2` asking for a non-inheritable
/// one. std has no stable way to say otherwise: `CommandExt::inherit_handles`
/// and `spawn_with_attributes` are both still nightly-only
/// (`windows_process_extensions_inherit_handles`, rust-lang/rust#146407, and
/// `…_raw_attribute`, #114854 — checked against the stable docs 2026-10-04;
/// this comment used to say the first was stabilised, which it is not). This is the
/// approach Python's `subprocess` takes (`close_fds` + `handle_list`):
/// `bInheritHandles = TRUE` restricted by `PROC_THREAD_ATTRIBUTE_HANDLE_LIST` to
/// the stdio handles that are inheritable — logs redirected to a file keep
/// flowing — or `FALSE` when there are none, where the child simply attaches to
/// the same console. Returns the new process id.
#[cfg(windows)]
fn spawn_without_inherited_handles(
    exe: &std::path::Path,
    args: &[std::ffi::OsString],
    env: &[(&str, &str)],
) -> std::io::Result<u32> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Foundation::{
        CloseHandle, GetHandleInformation, HANDLE, HANDLE_FLAG_INHERIT, INVALID_HANDLE_VALUE,
    };
    use windows_sys::Win32::System::Console::{
        GetStdHandle, STD_ERROR_HANDLE, STD_INPUT_HANDLE, STD_OUTPUT_HANDLE,
    };
    use windows_sys::Win32::System::Threading::{
        CreateProcessW, DeleteProcThreadAttributeList, InitializeProcThreadAttributeList,
        UpdateProcThreadAttribute, CREATE_UNICODE_ENVIRONMENT, EXTENDED_STARTUPINFO_PRESENT,
        LPPROC_THREAD_ATTRIBUTE_LIST, PROCESS_INFORMATION, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
        STARTF_USESTDHANDLES, STARTUPINFOEXW,
    };

    // The stdio handles that can be passed on: valid, inheritable, each once
    // (a handle listed twice fails the whole call — stdout and stderr are often
    // the same file).
    let stdio = [STD_INPUT_HANDLE, STD_OUTPUT_HANDLE, STD_ERROR_HANDLE].map(|which| {
        // SAFETY: GetStdHandle has no preconditions.
        let h = unsafe { GetStdHandle(which) };
        let mut flags = 0u32;
        // SAFETY: `h` is a handle this process holds (or null/invalid, which
        // GetHandleInformation rejects); `flags` is a valid out-pointer.
        let ok = !h.is_null()
            && h != INVALID_HANDLE_VALUE
            && unsafe { GetHandleInformation(h, &mut flags) } != 0
            && flags & HANDLE_FLAG_INHERIT != 0;
        if ok {
            h
        } else {
            std::ptr::null_mut()
        }
    });
    let mut pass: Vec<HANDLE> = Vec::new();
    for h in stdio {
        if !h.is_null() && !pass.contains(&h) {
            pass.push(h);
        }
    }

    let mut command_line = windows_command_line(exe.as_os_str(), args);
    let mut block = Vec::<u16>::new();
    for (k, v) in std::env::vars_os() {
        if env.iter().any(|(name, _)| k.eq_ignore_ascii_case(name)) {
            continue;
        }
        block.extend(k.encode_wide());
        block.push(u16::from(b'='));
        block.extend(v.encode_wide());
        block.push(0);
    }
    for (k, v) in env {
        block.extend(std::ffi::OsStr::new(k).encode_wide());
        block.push(u16::from(b'='));
        block.extend(std::ffi::OsStr::new(v).encode_wide());
        block.push(0);
    }
    block.push(0);

    // SAFETY: zeroed STARTUPINFOEXW / PROCESS_INFORMATION are valid "empty"
    // values for these plain-data structs.
    let mut si: STARTUPINFOEXW = unsafe { std::mem::zeroed() };
    si.StartupInfo.cb = std::mem::size_of::<STARTUPINFOEXW>() as u32;
    let mut pi: PROCESS_INFORMATION = unsafe { std::mem::zeroed() };
    let mut flags = CREATE_UNICODE_ENVIRONMENT;
    let mut attributes: Vec<usize> = Vec::new();
    if !pass.is_empty() {
        let mut size = 0usize;
        // SAFETY: the documented size query — a null list with a size pointer;
        // it fails by design and writes the size needed.
        unsafe { InitializeProcThreadAttributeList(std::ptr::null_mut(), 1, 0, &mut size) };
        // usize-aligned storage for the opaque attribute list.
        attributes.resize(size.div_ceil(std::mem::size_of::<usize>()), 0);
        let list = attributes.as_mut_ptr() as LPPROC_THREAD_ATTRIBUTE_LIST;
        // SAFETY: `list` points at `size` writable bytes, kept alive in
        // `attributes` until after CreateProcessW; `pass` outlives the call too.
        unsafe {
            if InitializeProcThreadAttributeList(list, 1, 0, &mut size) == 0 {
                return Err(std::io::Error::last_os_error());
            }
            if UpdateProcThreadAttribute(
                list,
                0,
                PROC_THREAD_ATTRIBUTE_HANDLE_LIST as usize,
                pass.as_ptr() as *const core::ffi::c_void,
                pass.len() * std::mem::size_of::<HANDLE>(),
                std::ptr::null_mut(),
                std::ptr::null(),
            ) == 0
            {
                let e = std::io::Error::last_os_error();
                DeleteProcThreadAttributeList(list);
                return Err(e);
            }
        }
        si.lpAttributeList = list;
        si.StartupInfo.dwFlags |= STARTF_USESTDHANDLES;
        si.StartupInfo.hStdInput = stdio[0];
        si.StartupInfo.hStdOutput = stdio[1];
        si.StartupInfo.hStdError = stdio[2];
        flags |= EXTENDED_STARTUPINFO_PRESENT;
    }
    // SAFETY: every pointer is valid for the call: the NUL-terminated mutable
    // command line, the double-NUL-terminated UTF-16 environment block, the
    // startup info (with its attribute list, when present) and `pi`.
    let created = unsafe {
        CreateProcessW(
            std::ptr::null(),
            command_line.as_mut_ptr(),
            std::ptr::null(),
            std::ptr::null(),
            i32::from(!pass.is_empty()),
            flags,
            block.as_ptr() as *const core::ffi::c_void,
            std::ptr::null(),
            &si.StartupInfo,
            &mut pi,
        )
    };
    let error = std::io::Error::last_os_error();
    if !si.lpAttributeList.is_null() {
        // SAFETY: initialised above and not yet deleted.
        unsafe { DeleteProcThreadAttributeList(si.lpAttributeList) };
    }
    if created == 0 {
        return Err(error);
    }
    // SAFETY: both handles were just returned to us by CreateProcessW.
    unsafe {
        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
    }
    Ok(pi.dwProcessId)
}

/// A Windows command line for `exe` + `args`, NUL-terminated: the program name
/// in quotes, and each argument quoted by the rules `CommandLineToArgvW` and the
/// MSVC runtime parse with — the same quoting std's `Command` applies.
#[cfg_attr(not(windows), allow(dead_code))]
fn windows_command_line(exe: &std::ffi::OsStr, args: &[std::ffi::OsString]) -> Vec<u16> {
    let mut out: Vec<u16> = Vec::new();
    let quote = u16::from(b'"');
    let backslash = u16::from(b'\\');
    out.push(quote);
    out.extend(exe.to_string_lossy().encode_utf16());
    out.push(quote);
    for arg in args {
        out.push(u16::from(b' '));
        let units: Vec<u16> = arg.to_string_lossy().encode_utf16().collect();
        let plain = !units.is_empty()
            && !units
                .iter()
                .any(|&c| c == u16::from(b' ') || c == u16::from(b'\t') || c == quote);
        if plain {
            out.extend(units);
            continue;
        }
        out.push(quote);
        let mut backslashes = 0usize;
        for c in units {
            if c == backslash {
                backslashes += 1;
            } else if c == quote {
                out.extend(std::iter::repeat_n(backslash, backslashes * 2 + 1));
                out.push(quote);
                backslashes = 0;
            } else {
                out.extend(std::iter::repeat_n(backslash, backslashes));
                out.push(c);
                backslashes = 0;
            }
        }
        out.extend(std::iter::repeat_n(backslash, backslashes * 2));
        out.push(quote);
    }
    out.push(0);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The predecessor going away ends the wait, and the answer says how long
    /// it took rather than merely that it happened.
    #[test]
    fn the_wait_ends_when_the_previous_version_exits() {
        let mut polls = 0;
        let waited = wait_while(
            || {
                polls += 1;
                polls < 3
            },
            Duration::from_secs(30),
            Duration::from_millis(1),
            Instant::now,
        );
        assert!(
            waited.is_some(),
            "a predecessor that exits must end the wait"
        );
        assert_eq!(polls, 3);
    }

    /// A predecessor that never exits must not hold the replacement for ever.
    /// Starting and reporting the port conflict is the better failure: the node
    /// says something, and starting it again by hand then works.
    ///
    /// Driven by a fake clock — a test that waits out a 30-second timeout to
    /// check a 30-second timeout is one nobody runs twice.
    #[test]
    fn a_predecessor_that_never_exits_does_not_block_startup_for_ever() {
        let start = Instant::now();
        let ticks = std::cell::Cell::new(0u32);
        let waited = wait_while(
            || true,
            Duration::from_secs(30),
            Duration::from_millis(1),
            || {
                let t = ticks.get();
                ticks.set(t + 1);
                start + Duration::from_secs(u64::from(t) * 10)
            },
        );
        assert!(
            waited.is_none(),
            "the wait must give up rather than hang a node's startup on a stuck predecessor"
        );
    }

    /// An ordinary start — no update, no variable — must not wait at all, or
    /// every node start pays for a case that only arises on Windows after an
    /// update.
    #[test]
    fn an_ordinary_start_does_not_wait() {
        assert!(
            std::env::var(HANDOFF_PID_VAR).is_err(),
            "test environment must not carry a handoff id"
        );
        let before = Instant::now();
        await_predecessor_exit();
        // Nor for the ports, even when one of them is genuinely taken: outside a
        // handoff a taken port is reported by the bind, as it always was.
        let taken = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();
        await_ports_released(&[PortClaim::Udp(taken.local_addr().unwrap())]);
        assert!(before.elapsed() < Duration::from_secs(1));
    }

    /// #769's shape on real sockets: the previous version still holds the port
    /// when the replacement starts, and lets go a moment later. The wait must
    /// last until then — and a claim it can bind must cost nothing.
    #[test]
    fn the_replacement_waits_until_the_previous_versions_ports_are_free() {
        let udp = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();
        let tcp = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let claims = [
            PortClaim::Udp(udp.local_addr().unwrap()),
            PortClaim::Tcp(tcp.local_addr().unwrap()),
        ];
        // Held: the probe must see it — the null control for everything below.
        assert!(claims.iter().all(|c| !c.is_free()));

        let release_after = Duration::from_millis(300);
        let holder = std::thread::spawn(move || {
            std::thread::sleep(release_after);
            drop((udp, tcp));
        });
        let waited = wait_for_ports(&claims, Duration::from_secs(10), Duration::from_millis(10))
            .expect("the ports were released, so the wait must end");
        holder.join().unwrap();
        assert!(
            waited >= release_after.saturating_sub(Duration::from_millis(50)),
            "returned after {waited:?}, before the previous version let go"
        );

        // Free from the start: no wait at all.
        let free = wait_for_ports(&claims, Duration::from_secs(10), Duration::from_millis(10))
            .expect("free ports");
        assert!(free < Duration::from_millis(100));
    }

    /// The replacement's command line must parse back into the same arguments —
    /// a data directory with spaces, a trailing backslash before the closing
    /// quote, an embedded quote, an empty argument. These are the
    /// `CommandLineToArgvW` / MSVC rules std's `Command` quotes by.
    #[test]
    fn a_relaunch_command_line_quotes_like_windows_parses() {
        let line = |exe: &str, args: &[&str]| {
            let args: Vec<std::ffi::OsString> = args.iter().map(|a| a.into()).collect();
            let mut v = windows_command_line(std::ffi::OsStr::new(exe), &args);
            assert_eq!(v.pop(), Some(0), "NUL-terminated");
            String::from_utf16(&v).unwrap()
        };
        assert_eq!(
            line(r"C:\SwarmLLM\swarmllm.exe", &["-p", "8800", "run"]),
            r#""C:\SwarmLLM\swarmllm.exe" -p 8800 run"#
        );
        assert_eq!(
            line(r"C:\x.exe", &["-d", r"C:\Users\A B\data\", "run"]),
            r#""C:\x.exe" -d "C:\Users\A B\data\\" run"#
        );
        assert_eq!(
            line(r"C:\x.exe", &[r#"say "hi""#]),
            r#""C:\x.exe" "say \"hi\"""#
        );
        assert_eq!(line(r"C:\x.exe", &[""]), r#""C:\x.exe" """#);
        assert_eq!(line(r"C:\x.exe", &[r"a\b"]), r#""C:\x.exe" a\b"#);
    }

    #[test]
    fn drain_timeout_is_long_enough_for_a_real_request() {
        // Prefill on a modest CPU node has been measured at 285-320s for a
        // ~600-1300 token prompt (gotcha #181). A drain window shorter than
        // that would routinely cut off a healthy request to install an update.
        assert!(
            DRAIN_TIMEOUT >= Duration::from_secs(320),
            "drain window must outlast a legitimate long prefill"
        );
    }
}
