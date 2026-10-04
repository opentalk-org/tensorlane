use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;
use tokio::fs;

#[derive(Debug, Deserialize, Serialize)]
pub struct Failure {
    pub message: String,
    pub retryable: bool,
    pub timestamp: u64,
}
impl std::fmt::Display for Failure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}
impl std::error::Error for Failure {}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

pub async fn failed(path: &Path, error: &anyhow::Error) {
    let retryable = match error.downcast_ref::<clickhouse::error::Error>() {
        Some(clickhouse::error::Error::BadResponse(message)) => ![
            "SYNTAX_ERROR",
            "UNKNOWN_IDENTIFIER",
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
        timestamp: now(),
    };
    if let Ok(bytes) = serde_json::to_vec(&failure) {
        let _ = crate::shared_cache::write_atomic(path, &bytes).await;
    }
}

pub async fn check(path: &Path) -> Result<()> {
    let bytes = match fs::read(path).await {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    let failure: Failure = serde_json::from_slice(&bytes)?;
    if failure.retryable && now().saturating_sub(failure.timestamp) >= 2 {
        fs::remove_file(path).await?;
        return Ok(());
    }
    Err(failure.into())
}

pub async fn wait_for_file(path: &Path) -> Result<bool> {
    let started = tokio::time::Instant::now();
    loop {
        if fs::try_exists(path).await? {
            return Ok(true);
        }
        check(&path.with_extension("error")).await?;
        if started.elapsed() >= std::time::Duration::from_secs(2) {
            return Ok(false);
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
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
