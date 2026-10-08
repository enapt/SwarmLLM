//! The node's log in a file as well as its window: on by default, in
//! `<data dir>/logs/swarmllm.log`, rotated by size so it can never fill a disk.
//!
//! **Why.** The log went only to the window the node ran in, and the advice was
//! `swarmllm run -vv 2>&1 | tee file.log` — easy to forget, and a problem that
//! shows up now and then (a memory refusal at 3 a.m.) was gone with the window.
//! A tester asked for a `logging.file` setting (2026-10-08); the key had been
//! accepted and documented as "not applied yet" since July. Ollama keeps
//! `server.log` in its data directory by default for the same reason, and
//! llama.cpp's server takes `--log-file`.
//!
//! **One writer.** The model processes (`swarmllm model-worker`) wrote to the
//! window they inherited. With a file, their output is piped to the daemon
//! ([`forward_worker_output`]), which writes each line to its own window and
//! the file — so their logs and their panics land in the same file, and only
//! one process ever writes it: rotating a file another process holds open is
//! refused on Windows and silently splits the log on Linux.
//!
//! **Bounded.** [`MAX_FILE_BYTES`] per file, [`OLDER_FILES_KEPT`] older ones
//! (`swarmllm.log.1` … `.3`): 200 MB at most. A rotation the filesystem refuses
//! (a viewer holding `.1` open on Windows) starts the current file over rather
//! than letting it grow.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

/// The file is rotated once a write would take it past this.
pub const MAX_FILE_BYTES: u64 = 50 * 1024 * 1024;

/// Rotated files kept beside the current one: `<name>.1` (newest) … `<name>.3`.
pub const OLDER_FILES_KEPT: usize = 3;

/// Where the log goes when nothing says otherwise: this file in this folder of
/// the data directory, joined as two parts so the path a Windows user is shown
/// has one kind of separator.
pub const DEFAULT_DIR: &str = "logs";
pub const DEFAULT_FILE: &str = "swarmllm.log";

/// The log file `setting` names — the `--log-file` flag, else `logging.file`.
/// Unset or empty: the default under `data_dir`. `off` / `false` / `none`: no
/// file. A relative path is under `data_dir`; an absolute one is taken as is.
pub fn log_file_path(setting: Option<&str>, data_dir: &Path) -> Option<PathBuf> {
    match setting.map(str::trim) {
        None | Some("") => Some(data_dir.join(DEFAULT_DIR).join(DEFAULT_FILE)),
        Some(s) if matches!(s.to_ascii_lowercase().as_str(), "off" | "false" | "none") => None,
        Some(s) => {
            let p = PathBuf::from(s);
            Some(if p.is_absolute() { p } else { data_dir.join(p) })
        }
    }
}

/// A log file that starts over in a new file at [`MAX_FILE_BYTES`], keeping
/// [`OLDER_FILES_KEPT`] older ones.
pub struct RotatingFile {
    path: PathBuf,
    file: Option<File>,
    written: u64,
    max_bytes: u64,
    keep: usize,
    /// Set when a rotation could not start the file over: try again only once
    /// it has grown past this, not on every line.
    retry_at: Option<u64>,
}

impl RotatingFile {
    /// Open `path` for appending — a restarted node continues its file —
    /// creating its directory.
    pub fn open(path: &Path, max_bytes: u64, keep: usize) -> std::io::Result<Self> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        let written = file.metadata().map(|m| m.len()).unwrap_or(0);
        Ok(Self {
            path: path.to_path_buf(),
            file: Some(file),
            written,
            max_bytes,
            keep,
            retry_at: None,
        })
    }

    /// Write one whole line (or event), rotating first if it would not fit.
    /// Errors are swallowed: a log that cannot be written must not take the
    /// node down with it.
    pub fn write_line(&mut self, bytes: &[u8]) {
        let limit = self.retry_at.unwrap_or(self.max_bytes);
        if self.written > 0 && self.written + bytes.len() as u64 > limit {
            self.rotate();
        }
        if let Some(file) = self.file.as_mut() {
            if file.write_all(bytes).is_ok() {
                self.written += bytes.len() as u64;
            }
        }
    }

    fn rotated(&self, n: usize) -> PathBuf {
        let mut name = self.path.as_os_str().to_owned();
        name.push(format!(".{n}"));
        PathBuf::from(name)
    }

    fn rotate(&mut self) {
        // Closed first: Windows refuses to rename a file this process holds.
        self.file = None;
        let _ = std::fs::remove_file(self.rotated(self.keep));
        for n in (1..self.keep).rev() {
            let _ = std::fs::rename(self.rotated(n), self.rotated(n + 1));
        }
        let renamed = self.keep > 0 && std::fs::rename(&self.path, self.rotated(1)).is_ok();
        let mut options = OpenOptions::new();
        options.create(true);
        if renamed {
            options.append(true);
        } else {
            // Kept within bounds when the rename is refused — a viewer holding
            // the file without sharing deletion (PowerShell's `Get-Content
            // -Wait` on Windows): start the file over instead.
            options.write(true).truncate(true);
        }
        // A viewer refusing even that must not stop the log: keep appending,
        // and try again once it has grown by a quarter of the bound.
        self.file = options
            .open(&self.path)
            .or_else(|_| {
                OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&self.path)
            })
            .ok();
        self.written = self
            .file
            .as_ref()
            .and_then(|f| f.metadata().ok())
            .map_or(0, |m| m.len());
        self.retry_at = (self.written >= self.max_bytes).then(|| self.written + self.max_bytes / 4);
    }
}

static SINK: OnceLock<(PathBuf, Mutex<RotatingFile>)> = OnceLock::new();

/// Start writing the log to `path` as well, once per process. Only the daemon
/// does: a model process's output reaches the file through the daemon.
pub fn install(path: &Path) -> std::io::Result<()> {
    let file = RotatingFile::open(path, MAX_FILE_BYTES, OLDER_FILES_KEPT)?;
    let _ = SINK.set((path.to_path_buf(), Mutex::new(file)));
    Ok(())
}

/// The file the log is being written to, if any.
pub fn file_path() -> Option<&'static Path> {
    SINK.get().map(|(p, _)| p.as_path())
}

fn write_to_file(bytes: &[u8]) {
    if let Some((_, file)) = SINK.get() {
        if let Ok(mut f) = file.lock() {
            f.write_line(bytes);
        }
    }
}

/// The `tracing` writer for the file: each event is formatted into a buffer and
/// written in ONE call when the event ends, so a rotation never cuts one.
#[derive(Clone, Copy, Default)]
pub struct FileWriter;

/// One event's text on its way to the file.
pub struct EventBuffer(Vec<u8>);

impl Write for EventBuffer {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Drop for EventBuffer {
    fn drop(&mut self) {
        if !self.0.is_empty() {
            write_to_file(&self.0);
        }
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for FileWriter {
    type Writer = EventBuffer;

    fn make_writer(&'a self) -> Self::Writer {
        EventBuffer(Vec::with_capacity(256))
    }
}

/// Whether a spawned model process's output should be piped through this
/// process ([`forward_worker_output`]) — only while the log has a file.
pub fn pipes_worker_output() -> bool {
    SINK.get().is_some()
}

/// Copy a model process's piped output, line by line, to this process's window
/// and the log file. Ends when the process closes the pipe.
pub fn forward_worker_output<R>(reader: R)
where
    R: tokio::io::AsyncRead + Unpin + Send + 'static,
{
    use tokio::io::AsyncBufReadExt;
    tokio::spawn(async move {
        let mut reader = tokio::io::BufReader::new(reader);
        let mut line = Vec::with_capacity(256);
        loop {
            line.clear();
            match reader.read_until(b'\n', &mut line).await {
                Ok(0) | Err(_) => break,
                Ok(_) => forward_line(&line),
            }
        }
    });
}

fn forward_line(line: &[u8]) {
    {
        let mut out = std::io::stdout().lock();
        let _ = out.write_all(line);
        if !line.ends_with(b"\n") {
            let _ = out.write_all(b"\n");
        }
    }
    if line.ends_with(b"\n") {
        write_to_file(line);
    } else {
        let mut whole = line.to_vec();
        whole.push(b'\n');
        write_to_file(&whole);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_setting_names_the_file_or_turns_it_off() {
        let data = Path::new("/data");
        assert_eq!(
            log_file_path(None, data),
            Some(PathBuf::from("/data/logs/swarmllm.log")),
            "on by default, under the data directory"
        );
        assert_eq!(log_file_path(Some(" "), data), log_file_path(None, data));
        for off in ["off", "OFF", "false", "none"] {
            assert_eq!(log_file_path(Some(off), data), None, "{off}");
        }
        assert_eq!(
            log_file_path(Some("node.log"), data),
            Some(PathBuf::from("/data/node.log")),
            "a relative path is under the data directory"
        );
        let absolute = std::env::temp_dir().join("x.log");
        assert_eq!(
            log_file_path(absolute.to_str(), data),
            Some(absolute.clone())
        );
    }

    /// The file never grows past its bound: a write that would cross it moves
    /// the file to `.1` (and `.1` to `.2` …) and starts a new one, keeping
    /// `keep` older files; no line is split, and a restart continues the file.
    #[test]
    fn the_log_file_rotates_at_its_size_and_keeps_a_few_older_ones() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("logs").join("swarmllm.log");
        let line = |n: usize| format!("line {n:04} {}\n", "x".repeat(40));
        let mut log = RotatingFile::open(&path, 200, 2).unwrap();
        for n in 0..20 {
            log.write_line(line(n).as_bytes());
        }
        drop(log);

        let read = |p: &Path| std::fs::read_to_string(p).unwrap_or_default();
        let current = read(&path);
        let one = read(&path.with_extension("log.1"));
        let two = read(&path.with_extension("log.2"));
        assert!(current.len() as u64 <= 200, "{}", current.len());
        assert!(!one.is_empty() && !two.is_empty());
        assert!(
            !path.with_extension("log.3").exists(),
            "only `keep` older files are kept"
        );
        assert!(
            current.contains("line 0019"),
            "the newest line is in the current file"
        );
        for text in [&current, &one, &two] {
            assert!(text
                .lines()
                .all(|l| l.starts_with("line ") && l.len() == 50));
        }
        let newest_in_one: usize = one.lines().last().unwrap()[5..9].parse().unwrap();
        let oldest_in_current: usize = current.lines().next().unwrap()[5..9].parse().unwrap();
        assert_eq!(
            newest_in_one + 1,
            oldest_in_current,
            "nothing lost between files"
        );

        // A restart appends to what is there.
        let before = current.len();
        let mut log = RotatingFile::open(&path, 200, 2).unwrap();
        log.write_line(b"after restart\n");
        drop(log);
        let after = read(&path);
        assert!(
            after.len() > before && after.ends_with("after restart\n")
                || after == "after restart\n"
        );
    }

    /// On Windows a reader that shares reading and writing but not deletion —
    /// PowerShell's `Get-Content -Wait` — makes renaming the file fail. The log
    /// then starts the file over in place, stays within its bound, and never
    /// stops being written.
    #[cfg(windows)]
    #[test]
    fn a_viewer_holding_the_log_open_on_windows_neither_stops_it_nor_unbounds_it() {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_SHARE_READ: u32 = 1;
        const FILE_SHARE_WRITE: u32 = 2;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("swarmllm.log");
        let mut log = RotatingFile::open(&path, 200, 2).unwrap();
        let viewer = OpenOptions::new()
            .read(true)
            .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE)
            .open(&path)
            .unwrap();
        for n in 0..20 {
            log.write_line(format!("line {n:04} {}\n", "x".repeat(40)).as_bytes());
        }
        let current = std::fs::read_to_string(&path).unwrap();
        assert!(
            !path.with_extension("log.1").exists(),
            "the rename was refused, so the file started over in place"
        );
        assert!(current.contains("line 0019"), "the log kept being written");
        assert!(
            current.len() as u64 <= 200,
            "and within its bound: {}",
            current.len()
        );
        drop(viewer);
        drop(log);
    }
}
