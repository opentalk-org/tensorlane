use std::{future::Future, time::Duration};

use anyhow::Result;
use axum::{Router, extract::State, http::StatusCode, routing::get};
use futures::TryFutureExt;
use tokio::fs;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::{runtime::Runtime, shared_cache::TemporaryFile};

const READINESS_TIMEOUT: Duration = Duration::from_secs(2);

pub fn router(runtime: Runtime) -> Router {
    Router::new()
        .route("/healthz", get(async || "ok\n"))
        .route("/readyz", get(ready))
        .with_state(runtime)
}

async fn ready(State(runtime): State<Runtime>) -> (StatusCode, &'static str) {
    let status = readiness_status(runtime.shutdown.clone(), check_dependencies(&runtime)).await;
    let body = if status == StatusCode::OK {
        "ready\n"
    } else {
        "not ready\n"
    };
    (status, body)
}

async fn readiness_status(
    shutdown: CancellationToken,
    checks: impl Future<Output = Result<()>>,
) -> StatusCode {
    tokio::select! {
        biased;
        _ = shutdown.cancelled() => StatusCode::SERVICE_UNAVAILABLE,
        result = tokio::time::timeout(READINESS_TIMEOUT, checks) => match result {
            Ok(Ok(())) => StatusCode::OK,
            _ => StatusCode::SERVICE_UNAVAILABLE,
        },
    }
}

async fn check_dependencies(runtime: &Runtime) -> Result<()> {
    let probe = TemporaryFile(runtime.cache.join(format!("{}.probe", Uuid::new_v4())));
    tokio::try_join!(
        crate::cache_limits::space(&runtime.cache, 0),
        fs::write(&probe.0, b"ready\n").err_into::<anyhow::Error>(),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn dependency_failure_removes_readiness() {
        for checks in [Ok(()), Err(anyhow::anyhow!("dependency unavailable"))] {
            let expected = if checks.is_ok() {
                StatusCode::OK
            } else {
                StatusCode::SERVICE_UNAVAILABLE
            };
            assert_eq!(
                readiness_status(CancellationToken::new(), async { checks }).await,
                expected
            );
        }
    }

    #[tokio::test(start_paused = true)]
    async fn stalled_dependencies_expire() {
        let started = tokio::time::Instant::now();
        assert_eq!(
            readiness_status(CancellationToken::new(), std::future::pending()).await,
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(started.elapsed(), READINESS_TIMEOUT);
    }

    #[tokio::test(start_paused = true)]
    async fn shutdown_removes_readiness_without_waiting_for_dependencies() {
        let shutdown = CancellationToken::new();
        let request = readiness_status(shutdown.clone(), std::future::pending());
        let cancel = async {
            tokio::task::yield_now().await;
            shutdown.cancel();
        };
        let (status, ()) = tokio::join!(request, cancel);
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            readiness_status(shutdown, async { Ok(()) }).await,
            StatusCode::SERVICE_UNAVAILABLE
        );
    }
}
