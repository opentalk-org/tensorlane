use anyhow::Result;
use error_context::error_context;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use time::OffsetDateTime;
use uuid::Uuid;

use crate::run::Config;

const SELECT_RUNS: &str = "
select ?fields from runs
left join (
    select run_id,
           toNullable(argMax(toInt8(run_status.status), run_status.timestamp)) as status,
           toNullable(max(run_status.timestamp)) as status_timestamp
    from run_status
    group by run_id
) latest on latest.run_id = runs.id
";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[repr(i8)]
pub enum RunStatus {
    Running = 1,
    Succeeded = 2,
    Failed = 3,
    Cancelled = 4,
    Queued = 5,
}

impl TryFrom<i8> for RunStatus {
    type Error = anyhow::Error;

    fn try_from(value: i8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Running),
            2 => Ok(Self::Succeeded),
            3 => Ok(Self::Failed),
            4 => Ok(Self::Cancelled),
            5 => Ok(Self::Queued),
            _ => anyhow::bail!("unknown run status {value}"),
        }
    }
}

pub struct Run {
    pub id: Uuid,
    pub project_id: Uuid,
    pub name: String,
    pub config: Map<String, Value>,
    pub status: Option<RunStatus>,
    pub status_timestamp: Option<OffsetDateTime>,
}

#[derive(clickhouse::Row, Serialize)]
struct RunConfigRow {
    #[serde(with = "clickhouse::serde::uuid")]
    id: Uuid,
    #[serde(with = "clickhouse::serde::uuid")]
    project_id: Uuid,
    name: String,
    config: String,
}

#[derive(clickhouse::Row, Deserialize)]
struct RunRow {
    #[serde(with = "clickhouse::serde::uuid")]
    id: Uuid,
    #[serde(with = "clickhouse::serde::uuid")]
    project_id: Uuid,
    name: String,
    config: String,
    status: Option<i8>,
    #[serde(with = "clickhouse::serde::time::datetime64::nanos::option")]
    status_timestamp: Option<OffsetDateTime>,
}

impl TryFrom<RunRow> for Run {
    type Error = anyhow::Error;

    fn try_from(row: RunRow) -> Result<Self, Self::Error> {
        Ok(Self {
            id: row.id,
            project_id: row.project_id,
            name: row.name,
            config: serde_json::from_str(&row.config)?,
            status: row.status.map(RunStatus::try_from).transpose()?,
            status_timestamp: row.status_timestamp,
        })
    }
}

#[derive(Clone)]
pub struct RunRepo {
    client: clickhouse::Client,
}

impl RunRepo {
    pub fn new(client: clickhouse::Client) -> Self {
        Self { client }
    }

    pub async fn create(
        &self,
        project_id: Uuid,
        name: &str,
        config: &Map<String, Value>,
    ) -> Result<Option<Uuid>> {
        Config::parse(config)?;
        let project = self
            .client
            .query("SELECT 1 FROM projects WHERE id = ? LIMIT 1")
            .bind(project_id)
            .fetch_optional::<u8>()
            .await?;
        if project.is_none() {
            return Ok(None);
        }
        let row = RunConfigRow {
            id: Uuid::new_v4(),
            project_id,
            name: name.to_owned(),
            config: serde_json::to_string(config)?,
        };
        let mut insert = self.client.insert::<RunConfigRow>("runs").await?;
        insert.write(&row).await?;
        insert.end().await?;
        self.append_status(row.id, RunStatus::Queued).await?;
        Ok(Some(row.id))
    }

    pub fn assets(&self) -> crate::asset_repo::AssetRepo {
        crate::asset_repo::AssetRepo::new(self.client.clone())
    }

    pub async fn get(&self, id: Uuid) -> Result<Option<Run>> {
        self.current(id).await?.map(Run::try_from).transpose()
    }

    pub async fn list(&self) -> Result<Vec<Run>> {
        self.client
            .query(&format!("{SELECT_RUNS} order by project_id, id"))
            .fetch_all::<RunRow>()
            .await?
            .into_iter()
            .map(Run::try_from)
            .collect()
    }

    async fn current(&self, id: Uuid) -> Result<Option<RunRow>> {
        self.client
            .query(&format!("{SELECT_RUNS} where id = ?"))
            .bind(id.to_string())
            .fetch_optional()
            .await
            .map_err(Into::into)
    }

    #[error_context("failed to append a run status")]
    pub async fn append_status(&self, run_id: Uuid, status: RunStatus) -> Result<()> {
        self.client
            .query("INSERT INTO run_status (timestamp, run_id, status) SELECT greatest(now64(9), max(timestamp) + INTERVAL 1 NANOSECOND), ?, ? FROM run_status WHERE run_id = ?")
            .bind(run_id)
            .bind(status as i8)
            .bind(run_id)
            .execute()
            .await?;
        Ok(())
    }

    pub async fn session(&self, id: Uuid) -> Result<Option<Session>> {
        Ok(self
            .client
            .query("SELECT ?fields FROM run_sessions FINAL WHERE run_id = ?")
            .bind(id)
            .fetch_optional()
            .await?)
    }

    pub async fn renew_session(&self, id: Uuid, session: Uuid) -> Result<()> {
        self.client
            .query(
                "INSERT INTO run_sessions (run_id, session_id, updated_at) VALUES (?, ?, now64(9))",
            )
            .bind(id)
            .bind(session)
            .with_setting("async_insert", "1")
            .with_setting("wait_for_async_insert", "1")
            .execute()
            .await?;
        Ok(())
    }
}

#[derive(clickhouse::Row, Deserialize)]
pub struct Session {
    #[serde(with = "clickhouse::serde::uuid")]
    pub session_id: Uuid,
}
