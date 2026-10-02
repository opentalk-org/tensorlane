use super::QuerySampler;
use crate::{db::fetch_samples, run::Config};
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
    for (name, query) in config.queries {
        let started = Instant::now();
        let rows = fetch_samples(&database, &query.sql, &query.params).await?;
        let count = rows.len();
        let sampler = QuerySampler::new(rows)?;
        println!(
            "{}",
            serde_json::json!({"stream":name,"samples":count,"batches":sampler.len(),"seconds":started.elapsed().as_secs_f64()})
        );
    }
    Ok(())
}
