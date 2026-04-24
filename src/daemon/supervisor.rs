//! Supervisor loop + shutdown drain for the daemon's subsystem JoinSet.
//!
//! The supervisor watches every spawned subsystem via a `JoinSet`. On task
//! exit, it decides whether to shut the daemon down (critical subsystem or
//! non-critical exceeding max restarts) or continue degraded. It also
//! listens for Ctrl+C / SIGTERM / API-triggered shutdown and then drains
//! the JoinSet with a timeout.

use std::sync::Arc;

use tokio::sync::watch;
use tokio::task::JoinSet;

use super::helpers::{SubsystemCriticality, MAX_NONCRITICAL_FAILURES};
use super::state::SharedState;

/// Run the supervisor loop until a shutdown condition is hit, then drain
/// the JoinSet. Consumes the JoinSet.
pub(super) async fn run(
    mut subsystems: JoinSet<(&'static str, SubsystemCriticality, Result<(), String>)>,
    mut shutdown_rx: watch::Receiver<bool>,
    shared_state: Arc<SharedState>,
) {
    // Track failure count per subsystem name. Counter only meaningfully
    // grows above 1 in a future world that re-spawns failed subsystems —
    // today each subsystem launches exactly once at startup.
    let mut failure_counts: std::collections::HashMap<&str, u32> = std::collections::HashMap::new();

    // Register SIGTERM handler ONCE before the loop to avoid re-registering
    // on each iteration. Windows has no SIGTERM — the select branch below
    // pends forever on that platform and Ctrl+C / API shutdown take over.
    #[cfg(unix)]
    let mut sigterm_stream: Option<tokio::signal::unix::Signal> =
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).ok();

    loop {
        tokio::select! {
            // Handle Ctrl+C
            _ = tokio::signal::ctrl_c() => {
                tracing::info!("Shutdown signal received (Ctrl+C)");
                break;
            }
            // SIGTERM on Unix; pends forever on Windows (no SIGTERM there).
            _ = async {
                #[cfg(unix)]
                {
                    match sigterm_stream.as_mut() {
                        Some(s) => { let _ = s.recv().await; }
                        None => { std::future::pending::<()>().await; }
                    }
                }
                #[cfg(not(unix))]
                {
                    std::future::pending::<()>().await;
                }
            } => {
                tracing::info!("Shutdown signal received (SIGTERM)");
                break;
            }
            // Handle API-triggered shutdown (watch channel)
            _ = shutdown_rx.changed() => {
                if *shutdown_rx.borrow() {
                    tracing::info!("Shutdown requested via API — draining subsystems");
                    break;
                }
            }
            // Handle subsystem task exits
            result = subsystems.join_next() => {
                match result {
                    None => {
                        tracing::error!(subsystem_count = subsystems.len(), "all subsystem tasks have exited unexpectedly");
                        break;
                    }
                    Some(Ok((name, criticality, task_result))) => {
                        if *shutdown_rx.borrow() {
                            tracing::debug!(subsystem = name, "Subsystem exited during shutdown");
                            continue;
                        }

                        match task_result {
                            Ok(()) => {
                                tracing::warn!(
                                    subsystem = name,
                                    "Subsystem exited unexpectedly with Ok"
                                );
                            }
                            Err(ref e) => {
                                tracing::error!(
                                    subsystem = name,
                                    error = %e,
                                    "Subsystem exited with error"
                                );
                            }
                        }

                        let count = failure_counts.entry(name).or_insert(0);
                        *count += 1;

                        if criticality == SubsystemCriticality::Critical {
                            tracing::error!(
                                subsystem = name,
                                "Critical subsystem failed — triggering graceful shutdown"
                            );
                            break;
                        } else if *count >= MAX_NONCRITICAL_FAILURES {
                            tracing::error!(
                                subsystem = name,
                                failure_count = *count,
                                max_failures = MAX_NONCRITICAL_FAILURES,
                                "Non-critical subsystem exceeded max failure count — triggering shutdown"
                            );
                            break;
                        } else {
                            tracing::warn!(
                                subsystem = name,
                                failure_count = *count,
                                max_failures = MAX_NONCRITICAL_FAILURES,
                                "Non-critical subsystem failed — daemon continues without it"
                            );
                        }
                    }
                    Some(Err(join_error)) => {
                        if join_error.is_panic() {
                            tracing::error!(
                                error = %join_error,
                                "Subsystem task panicked — triggering shutdown"
                            );
                            break;
                        } else {
                            tracing::warn!(
                                error = %join_error,
                                "Subsystem task cancelled"
                            );
                        }
                    }
                }
            }
        }
    }

    // Signal graceful shutdown to all subsystems
    shared_state.shutdown();

    // Drain the JoinSet with a timeout so subsystems can run their cleanup
    // (e.g., save peer cache, close connections, flush data).
    const SHUTDOWN_TIMEOUT_SECS: u64 = 10;
    tracing::info!(
        timeout_secs = SHUTDOWN_TIMEOUT_SECS,
        "Waiting for subsystems to shut down"
    );
    let drain_deadline = tokio::time::sleep(std::time::Duration::from_secs(SHUTDOWN_TIMEOUT_SECS));
    tokio::pin!(drain_deadline);
    loop {
        tokio::select! {
            _ = &mut drain_deadline => {
                tracing::warn!("Shutdown timeout — aborting remaining subsystems");
                break;
            }
            result = subsystems.join_next() => {
                match result {
                    Some(Ok((name, _, _))) => {
                        tracing::debug!(subsystem = name, "Subsystem exited cleanly");
                    }
                    Some(Err(e)) => {
                        tracing::debug!(error = %e, "Subsystem join error during shutdown");
                    }
                    None => {
                        tracing::info!("All subsystems shut down cleanly");
                        break;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn joinset_catches_task_panic() {
        let mut set: JoinSet<(&str, SubsystemCriticality, Result<(), String>)> = JoinSet::new();
        set.spawn(async {
            panic!("simulated subsystem panic");
        });

        let result = set.join_next().await.unwrap();
        assert!(result.is_err());
        assert!(result.unwrap_err().is_panic());
    }

    #[tokio::test]
    async fn joinset_returns_task_error() {
        let mut set: JoinSet<(&str, SubsystemCriticality, Result<(), String>)> = JoinSet::new();
        set.spawn(async {
            (
                "TestSubsystem",
                SubsystemCriticality::NonCritical,
                Err("boom".to_string()),
            )
        });

        let result = set.join_next().await.unwrap();
        let (name, crit, task_result) = result.unwrap();
        assert_eq!(name, "TestSubsystem");
        assert_eq!(crit, SubsystemCriticality::NonCritical);
        assert!(task_result.is_err());
        assert_eq!(task_result.unwrap_err(), "boom");
    }

    #[tokio::test]
    async fn joinset_returns_task_success() {
        let mut set: JoinSet<(&str, SubsystemCriticality, Result<(), String>)> = JoinSet::new();
        set.spawn(async { ("TestSubsystem", SubsystemCriticality::Critical, Ok(())) });

        let result = set.join_next().await.unwrap();
        let (name, crit, task_result) = result.unwrap();
        assert_eq!(name, "TestSubsystem");
        assert_eq!(crit, SubsystemCriticality::Critical);
        assert!(task_result.is_ok());
    }

    #[tokio::test]
    async fn supervisor_non_critical_failure_does_not_drain_set() {
        let mut set: JoinSet<(&str, SubsystemCriticality, Result<(), String>)> = JoinSet::new();

        set.spawn(async {
            (
                "HealthMonitor",
                SubsystemCriticality::NonCritical,
                Err("test error".to_string()),
            )
        });

        set.spawn(async {
            tokio::time::sleep(Duration::from_secs(3600)).await;
            ("ApiServer", SubsystemCriticality::Critical, Ok(()))
        });

        let result = set.join_next().await.unwrap();
        let (name, crit, _) = result.unwrap();
        assert_eq!(name, "HealthMonitor");
        assert_eq!(crit, SubsystemCriticality::NonCritical);

        assert_eq!(set.len(), 1);

        set.abort_all();
    }

    #[tokio::test]
    async fn supervisor_failure_counting() {
        let mut failure_counts: std::collections::HashMap<&str, u32> =
            std::collections::HashMap::new();

        for i in 1..=5 {
            let count = failure_counts.entry("HealthMonitor").or_insert(0);
            *count += 1;
            assert_eq!(*count, i);
        }

        assert_eq!(
            *failure_counts.get("HealthMonitor").unwrap(),
            MAX_NONCRITICAL_FAILURES
        );

        let count = failure_counts.entry("HealthMonitor").or_insert(0);
        *count += 1;
        assert!(*count > MAX_NONCRITICAL_FAILURES);
    }
}
