use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::{path::Path, time::Duration};
use tokio::{fs, time::Instant};

#[derive(Debug, Deserialize, Serialize)]
pub struct Failure {
    pub message: String,
    pub retryable: bool,
}
impl std::fmt::Display for Failure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}
impl std::error::Error for Failure {}

pub async fn failed(path: &Path, error: &anyhow::Error) {
    tracing::warn!(path = %path.display(), error = %error, "background job failed");
    let retryable = match error.downcast_ref::<clickhouse::error::Error>() {
        Some(clickhouse::error::Error::BadResponse(message)) => ![
            "SYNTAX_ERROR",
            "UNKNOWN_IDENTIFIER",
            "TIMEOUT_EXCEEDED",
            "UNKNOWN_FUNCTION",
            "UNKNOWN_TABLE",
            "UNKNOWN_DATABASE",
            "BAD_ARGUMENTS",
            "TYPE_MISMATCH",
            "ILLEGAL_TYPE_OF_ARGUMENT",
            "UNKNOWN_QUERY_PARAMETER",
            "CANNOT_PARSE",
            "CANNOT_CONVERT",
            "UNKNOWN_TYPE",
            "AUTHENTICATION_FAILED",
        ]
        .iter()
        .any(|code| message.contains(code)),
        Some(
            clickhouse::error::Error::SchemaMismatch(_)
            | clickhouse::error::Error::InvalidParams(_),
        ) => false,
        _ => ![
            "exceeds",
            "unordered",
            "strictly ordered",
            "unexpected size",
            "SHA256 does not match",
            "conflicting retry",
            "invalid",
            "missing parameter",
        ]
        .iter()
        .any(|text| error.to_string().contains(text)),
    };
    let failure = Failure {
        message: format!("{error:#}"),
        retryable,
    };
    let _ = crate::shared_cache::write_atomic(path, &serde_json::to_vec(&failure).unwrap()).await;
}

pub async fn check(path: &Path) -> Result<()> {
    let bytes = match fs::read(path).await {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    let failure: Failure = serde_json::from_slice(&bytes)?;
    if failure.retryable
        && fs::metadata(path)
            .await?
            .modified()?
            .elapsed()
            .unwrap_or_default()
            >= Duration::from_secs(2)
    {
        fs::remove_file(path).await?;
        return Ok(());
    }
    Err(failure.into())
}

pub async fn wait_for_file(path: &Path) -> Result<bool> {
    let started = Instant::now();
    loop {
        if fs::try_exists(path).await? {
            return Ok(true);
        }
        check(&path.with_extension("error")).await?;
        if started.elapsed() >= Duration::from_secs(2) {
            return Ok(false);
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn permanent_query_failures_are_not_retried() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let path = temp.path().join("plan.error");
        for error in [
            anyhow::anyhow!("query rows must be strictly ordered by batch_idx, sample_idx"),
            clickhouse::error::Error::BadResponse("UNKNOWN_FUNCTION".into()).into(),
        ] {
            failed(&path, &error).await;
            let failure: Failure = serde_json::from_slice(&fs::read(&path).await?)?;
            assert!(!failure.retryable);
            assert!(check(&path).await.is_err());
        }
        Ok(())
    }
}
