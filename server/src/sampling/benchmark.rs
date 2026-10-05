use super::plan::QuerySampler;
use crate::{
    db::{SampleRow, stream_samples},
    run_config::Config,
};
use std::{env, time::Instant};

#[tokio::test]
#[ignore = "requires live ClickHouse and TENSORLANE_BENCHMARK_CONFIG"]
async fn live_plan_memory() -> anyhow::Result<()> {
    let database = clickhouse::Client::default()
        .with_url(env::var("CLICKHOUSE_URL")?)
        .with_user(env::var("CLICKHOUSE_USER")?)
        .with_password(env::var("CLICKHOUSE_PASSWORD")?);
    let document: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(env::var(
        "TENSORLANE_BENCHMARK_CONFIG",
    )?)?)?;
    let config = Config::parse(document["config"].as_object().unwrap())?;
    let dir = tempfile::tempdir()?;
    for (index, (name, query)) in config.queries.into_iter().enumerate() {
        let started = Instant::now();
        let rows = stream_samples(&database, &query.sql, &query.params);
        let path = dir.path().join(format!("{index}.plan"));
        let mut sampler = QuerySampler::create(&name, rows, &path, false).await?;
        let disk_bytes = std::fs::metadata(&path)?.len();
        let (mut count, mut batches) = (0, 0);
        while let Some(batch) = sampler.batch_at(batches).await? {
            count += batch.samples.len();
            batches += 1;
        }
        println!(
            "{}",
            serde_json::json!({"stream":name,"samples":count,"batches":batches,
            "disk_bytes":disk_bytes,"seconds":started.elapsed().as_secs_f64()})
        );
    }
    Ok(())
}

#[tokio::test]
#[ignore = "writes and replays two large synthetic plans to measure peak RSS"]
async fn disk_plan_memory() -> anyhow::Result<()> {
    let rows: u64 = env::var("TENSORLANE_BENCHMARK_ROWS")
        .unwrap_or("500000".into())
        .parse()?;
    let dir = tempfile::tempdir()?;
    let started = Instant::now();
    let mut plans = Vec::new();
    let mut disk_bytes = 0;
    for (index, (count, text_len)) in [(rows, 650), (rows * 594265 / 2560000, 1150)]
        .into_iter()
        .enumerate()
    {
        let metadata = serde_json::json!({"text":"a".repeat(text_len),"weight":4.5}).to_string();
        let blobs = r#"{"payload":{"object":"objects/01234567-0123-0123-0123-012345678901.tar","byte_offset":1024,"byte_length":65536}}"#;
        let source = futures::stream::iter((0..count).map(|i| {
            Ok(SampleRow {
                sample_id: "01234567-0123-0123-0123-012345678901".into(),
                batch_idx: i / 32,
                sample_idx: i,
                metadata_json: metadata.clone(),
                blobs_json: blobs.into(),
            })
        }));
        let path = dir.path().join(format!("{index}.plan"));
        plans.push(QuerySampler::create("benchmark", source, &path, false).await?);
        disk_bytes += std::fs::metadata(path)?.len();
    }
    let mut samples = 0;
    for plan in &mut plans {
        let mut sequence = 0;
        while let Some(batch) = plan.batch_at(sequence).await? {
            samples += batch.samples.len();
            sequence += 1;
        }
    }
    println!(
        "{}",
        serde_json::json!({"samples":samples,"disk_bytes":disk_bytes,
        "files":std::fs::read_dir(dir.path())?.count(),"seconds":started.elapsed().as_secs_f64(),
        "memory":std::fs::read_to_string("/proc/self/status")?.lines()
            .filter(|line| line.starts_with("VmHWM:") || line.starts_with("VmRSS:"))
            .collect::<Vec<_>>()})
    );
    Ok(())
}
