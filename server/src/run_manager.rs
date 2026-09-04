use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use time::OffsetDateTime;
use uuid::Uuid;

use crate::run::DataConfig;

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
    pub data_config: DataConfig,
    pub train_config: Map<String, Value>,
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
    data_config: String,
    train_config: String,
}

#[derive(clickhouse::Row, Serialize)]
struct StatusRow {
    #[serde(with = "clickhouse::serde::time::datetime64::nanos")]
    timestamp: OffsetDateTime,
    #[serde(with = "clickhouse::serde::uuid")]
    run_id: Uuid,
    status: i8,
}

#[derive(clickhouse::Row, Deserialize)]
struct RunRow {
    #[serde(with = "clickhouse::serde::uuid")]
    id: Uuid,
    #[serde(with = "clickhouse::serde::uuid")]
    project_id: Uuid,
    name: String,
    data_config: String,
    train_config: String,
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
            data_config: serde_json::from_str(&row.data_config)?,
            train_config: serde_json::from_str(&row.train_config)?,
            status: row.status.map(RunStatus::try_from).transpose()?,
            status_timestamp: row.status_timestamp,
        })
    }
}

#[derive(Clone)]
pub struct RunManager {
    client: clickhouse::Client,
}

impl RunManager {
    pub fn new(client: clickhouse::Client) -> Self {
        Self { client }
    }

    pub async fn create(
        &self,
        project_id: Uuid,
        name: &str,
        data_config: &DataConfig,
        train_config: &Map<String, Value>,
    ) -> anyhow::Result<Uuid> {
        let row = RunConfigRow {
            id: Uuid::new_v4(),
            project_id,
            name: name.to_owned(),
            data_config: serde_json::to_string(data_config)?,
            train_config: serde_json::to_string(train_config)?,
        };
        let mut insert = self.client.insert::<RunConfigRow>("runs").await?;
        insert.write(&row).await?;
        insert.end().await?;
        self.append_status(row.id, RunStatus::Queued).await?;
        Ok(row.id)
    }

    pub async fn get(&self, id: Uuid) -> anyhow::Result<Option<Run>> {
        self.current(id).await?.map(Run::try_from).transpose()
    }

    pub async fn list(&self) -> anyhow::Result<Vec<Run>> {
        self.client
            .query(&format!("{SELECT_RUNS} order by project_id, id"))
            .fetch_all::<RunRow>()
            .await?
            .into_iter()
            .map(Run::try_from)
            .collect()
    }

    async fn current(&self, id: Uuid) -> anyhow::Result<Option<RunRow>> {
        self.client
            .query(&format!("{SELECT_RUNS} where id = ?"))
            .bind(id.to_string())
            .fetch_optional()
            .await
            .map_err(Into::into)
    }

    pub async fn append_status(&self, run_id: Uuid, status: RunStatus) -> anyhow::Result<()> {
        let row = StatusRow {
            timestamp: OffsetDateTime::now_utc(),
            run_id,
            status: status as i8,
        };
        let mut insert = self.client.insert::<StatusRow>("run_status").await?;
        insert.write(&row).await?;
        insert.end().await?;
        Ok(())
    }
}
