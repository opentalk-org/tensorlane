use super::{
    setup::{KEY, TestEnv},
    stress::TrainingClient,
};
use anyhow::{Context, Result, ensure};
use bytes::Bytes;
use futures::{StreamExt, TryStreamExt, future::try_join_all, stream};
use serde_json::json;
use std::{
    path::Path,
    time::{Duration, Instant},
};

const STREAMS: [&str; 3] = ["images", "audio", "labels"];

fn config(
    samples: usize,
    blob_bytes: usize,
    batch_size: usize,
    limit: usize,
    repeat: bool,
) -> serde_json::Value {
    let sql = "SELECT concat('sample-',toString(number)) AS sample_id, intDiv(number,{batch_size:UInt64}) AS batch_idx, modulo(number,{batch_size:UInt64}) AS sample_idx, concat('{\"position\":',toString(number),'}') AS metadata_json, concat('{\"payload\":{\"object\":\"performance/data\",\"byte_offset\":',toString(number*{blob_bytes:UInt64}),',\"byte_length\":',toString({blob_bytes:UInt64}),'}}') AS blobs_json FROM numbers({samples:UInt64}) ORDER BY batch_idx,sample_idx";
    let query = json!({"sql":sql,"params":{"samples":samples,"batch_size":batch_size,"blob_bytes":blob_bytes},"repeat":repeat});
    let queries: serde_json::Map<_, _> = STREAMS
        .into_iter()
        .map(|name| (name.to_owned(), query.clone()))
        .collect();
    json!({"queries":queries,"tensorlane":{"max_load_memory_bytes":limit}})
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "live four-run, three-stream throughput sweep"]
async fn throughput_at_memory_and_request_limits() -> Result<()> {
    const BATCHES: usize = 32;
    const BATCH_SIZE: usize = 8;
    const BLOB_BYTES: usize = 64 * 1024;
    let env = TestEnv::start().await?;
    env.put_object(
        "performance/data",
        Bytes::from(vec![23; (BATCHES + 1) * BATCH_SIZE * BLOB_BYTES]),
    )
    .await?;
    for limit in [1024, 6 * 1024 * 1024, 96 * 1024 * 1024] {
        for concurrency in [1, 4, 8] {
            let mut runs = Vec::new();
            for _ in 0..4 {
                let run = env
                    .create_run(config(
                        (BATCHES + 1) * BATCH_SIZE,
                        BLOB_BYTES,
                        BATCH_SIZE,
                        limit,
                        false,
                    ))
                    .await?;
                env.init(&run).await?;
                runs.push(run);
            }
            // Prepare query snapshots and warm S3 connections. Timed batches
            // have distinct cache keys and still fetch every payload from S3.
            let mut warm = Vec::new();
            for run in &runs {
                for name in STREAMS {
                    warm.push(env.batch(run, name, 0));
                }
            }
            try_join_all(warm).await?;
            let mut consumers = Vec::new();
            for run in &runs {
                for name in STREAMS {
                    let env = &env;
                    consumers.push(async move {
                        stream::iter(1..=BATCHES)
                            .map(|sequence| async move {
                                let started = Instant::now();
                                let batch = env
                                    .batch(run, name, sequence as u64)
                                    .await?
                                    .context("unexpected EOF")?;
                                ensure!(
                                    batch.stream == name
                                        && batch.batch_id == sequence as u64
                                        && batch.query_batch_idx == sequence as u64,
                                    "wrong batch identity"
                                );
                                ensure!(batch.batch.len() == BATCH_SIZE, "wrong sample count");
                                for (index, sample) in batch.batch.iter().enumerate() {
                                    ensure!(
                                        sample.sample_id
                                            == format!("sample-{}", sequence * BATCH_SIZE + index),
                                        "wrong sample order"
                                    );
                                    let payload = &sample.blobs["payload"];
                                    ensure!(
                                        payload.len() == BLOB_BYTES
                                            && payload.iter().all(|byte| *byte == 23),
                                        "corrupt payload"
                                    );
                                }
                                anyhow::Ok(started.elapsed().as_secs_f64())
                            })
                            .buffer_unordered(concurrency)
                            .try_collect::<Vec<_>>()
                            .await
                    });
                }
            }
            let started = Instant::now();
            let mut latencies: Vec<_> =
                tokio::time::timeout(Duration::from_secs(120), try_join_all(consumers))
                    .await??
                    .into_iter()
                    .flatten()
                    .collect();
            let seconds = started.elapsed().as_secs_f64();
            latencies.sort_by(f64::total_cmp);
            let batches = latencies.len();
            assert_eq!(batches, 4 * STREAMS.len() * BATCHES);
            for run in &runs {
                for name in STREAMS {
                    assert!(env.batch(run, name, (BATCHES + 1) as u64).await?.is_none());
                }
            }
            println!(
                "{}",
                json!({"run_limit_bytes":limit,"per_stream_limit_bytes":(limit / STREAMS.len()).max(1),"requests_per_stream":concurrency,"runs":4,"streams_per_run":STREAMS.len(),"batches":batches,"seconds":seconds,"batches_per_second":batches as f64 / seconds,"payload_mib_per_second":(batches * BATCH_SIZE * BLOB_BYTES) as f64 / 1048576.0 / seconds,"latency_p95_seconds":latencies[(batches - 1) * 95 / 100],"latency_max_seconds":latencies[batches - 1]})
            );
        }
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "one uninterrupted five-minute Python client/server soak"]
async fn five_minute_single_shot_soak() -> Result<()> {
    let python =
        std::env::var_os("TENSORLANE_TEST_PYTHON").context("set TENSORLANE_TEST_PYTHON")?;
    let env = TestEnv::start().await?;
    env.put_object("performance/data", Bytes::from(vec![23; 512 * 32768]))
        .await?;
    let run = env.create_run(config(512, 32768, 8, 1024, true)).await?;
    let root = tempfile::Builder::new()
        .prefix("tl-soak-")
        .tempdir_in("/tmp")?;
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("../client/tests/server_soak.py");
    let process = tokio::process::Command::new(python)
        .arg(script)
        .args([&env.url, &run])
        .arg(root.path())
        .arg("300")
        .env("TENSORLANE_API_KEY", KEY)
        .env("OMP_NUM_THREADS", "1")
        .env("MKL_NUM_THREADS", "1")
        .env("OPENBLAS_NUM_THREADS", "1")
        .process_group(0)
        .kill_on_drop(true)
        .spawn()?;
    let mut client = TrainingClient {
        group: process.id().unwrap() as i32,
        process,
    };
    let status = tokio::time::timeout(Duration::from_secs(360), client.process.wait()).await??;
    ensure!(status.success(), "single-shot soak failed: {status}");
    assert_eq!(env.status(&run).await?, "succeeded");
    env.ready().await?;
    Ok(())
}
