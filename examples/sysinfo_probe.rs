//! Prices what `GET /api/admin/stats` pays to describe this machine, and
//! checks the cheap path reports the SAME four facts as the expensive one.
//!
//! `detect_hardware` is the one site in the tree that builds a sysinfo
//! `System` with `new_all()` and then calls `refresh_all()` on it; every
//! other site uses `System::new()` plus a targeted refresh.
use std::time::Instant;

fn min_ms<F: FnMut()>(reps: usize, mut f: F) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64() * 1000.0);
    }
    best
}

type Facts = (u64, u64, u64, String, usize);

fn read_whole(pid: sysinfo::Pid) -> Facts {
    let mut sys = sysinfo::System::new_all();
    sys.refresh_all();
    facts(&sys, pid)
}

fn read_targeted(pid: sysinfo::Pid) -> Facts {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    sys.refresh_cpu_list(sysinfo::CpuRefreshKind::nothing());
    sys.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
    facts(&sys, pid)
}

fn facts(sys: &sysinfo::System, pid: sysinfo::Pid) -> Facts {
    (
        sys.total_memory() / (1024 * 1024),
        sys.used_memory() / (1024 * 1024),
        sys.process(pid)
            .map(|p| p.memory() / (1024 * 1024))
            .unwrap_or(0),
        sys.cpus()
            .first()
            .map(|c| c.brand().to_string())
            .unwrap_or_else(|| "Unknown".into()),
        sys.cpus().len(),
    )
}

fn main() {
    let pid = sysinfo::Pid::from_u32(std::process::id());
    let w = read_whole(pid);
    let t = read_targeted(pid);
    println!(
        "whole:    total={} used={} rss={} cpu={:?} cores={}",
        w.0, w.1, w.2, w.3, w.4
    );
    println!(
        "targeted: total={} used={} rss={} cpu={:?} cores={}",
        t.0, t.1, t.2, t.3, t.4
    );
    println!(
        "same total_ram={} cpu_name={} cores={} rss_nonzero={}",
        w.0 == t.0,
        w.3 == t.3,
        w.4 == t.4,
        t.2 > 0
    );
    println!();
    println!(
        "new_all()+refresh_all()      {:8.2} ms",
        min_ms(7, || {
            read_whole(pid);
        })
    );
    println!(
        "targeted                     {:8.2} ms",
        min_ms(7, || {
            read_targeted(pid);
        })
    );
    println!(
        "Disks::new_with_refreshed    {:8.2} ms",
        min_ms(7, || {
            std::hint::black_box(sysinfo::Disks::new_with_refreshed_list().list().len());
        })
    );
}
