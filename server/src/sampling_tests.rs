use super::QuerySampler;
use crate::db::SampleRow;
use futures::stream;

fn row(batch: u64, index: u64, id: &str) -> SampleRow {
    SampleRow {
        sample_id: id.into(),
        batch_idx: batch,
        sample_idx: index,
        metadata_json: r#"{"text":"hello","nested":{"label":3}}"#.into(),
        blobs_json: "{}".into(),
    }
}

async fn sampler(
    rows: Vec<SampleRow>,
    repeat: bool,
) -> anyhow::Result<(tempfile::TempDir, QuerySampler)> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("plan");
    QuerySampler::create("training", stream::iter(rows.into_iter().map(Ok)), &path).await?;
    let sampler = QuerySampler::open(&path, repeat).await?;
    Ok((dir, sampler))
}

#[tokio::test]
async fn groups_sparse_batches_and_preserves_repeated_samples() {
    let (_dir, mut sampler) = sampler(vec![row(0, 0, "a"), row(0, 1, "b"), row(4, 0, "a")], false)
        .await
        .unwrap();
    let first = sampler.batch_at(0).await.unwrap().unwrap();
    assert_eq!(
        first
            .samples
            .iter()
            .map(|s| s.sample_id.as_str())
            .collect::<Vec<_>>(),
        vec!["a", "b"]
    );
    assert_eq!(
        sampler.batch_at(1).await.unwrap().unwrap().query_batch_idx,
        4
    );
    assert!(sampler.batch_at(2).await.unwrap().is_none());
}

#[tokio::test]
async fn rejects_unordered_rows_and_skips_invalid_rows() {
    for positions in [[(2, 0), (1, 0)], [(0, 2), (0, 1)], [(0, 1), (0, 1)]] {
        let rows = positions.into_iter().map(|(b, i)| row(b, i, "a")).collect();
        assert!(sampler(rows, false).await.is_err());
    }
    for (meta, blobs) in [
        ("[]", "{}"),
        ("{}", "[]"),
        ("{}", r#"{"x":{"object":"a","byte_offset":0}}"#),
        (
            "{}",
            r#"{"x":{"object":"a","byte_offset":0,"byte_length":0}}"#,
        ),
        (
            "{}",
            r#"{"x":{"object":"a","byte_offset":18446744073709551615,"byte_length":2}}"#,
        ),
    ] {
        let mut sample = row(0, 0, "a");
        sample.metadata_json = meta.into();
        sample.blobs_json = blobs.into();
        let (_dir, mut plan) = sampler(vec![sample, row(1, 0, "b")], false).await.unwrap();
        assert_eq!(plan.batch_at(0).await.unwrap().unwrap().query_batch_idx, 1);
    }
}

#[tokio::test]
async fn repeats_identical_plans_and_empty_repetition_ends() {
    let (dir, mut repeated) = sampler(vec![row(7, 0, "a"), row(9, 0, "b")], true)
        .await
        .unwrap();
    let original = std::fs::read(dir.path().join("plan")).unwrap();
    for index in 0..1000 {
        let batch = repeated.batch_at(index).await.unwrap().unwrap();
        assert_eq!(batch.query_batch_idx, if index % 2 == 0 { 7 } else { 9 });
        assert_eq!(
            batch.samples[0].sample_id,
            if index % 2 == 0 { "a" } else { "b" }
        );
    }
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 3);
    assert_eq!(std::fs::read(dir.path().join("plan")).unwrap(), original);
    let (_dir, mut empty) = sampler(vec![], true).await.unwrap();
    assert!(empty.batch_at(0).await.unwrap().is_none());
}

#[tokio::test]
async fn plan_can_be_reopened_and_replayed_by_sequence() {
    let (dir, _) = sampler(vec![row(7, 0, "a"), row(9, 0, "b")], true)
        .await
        .unwrap();
    let mut first = QuerySampler::open(&dir.path().join("plan"), true)
        .await
        .unwrap();
    let mut second = QuerySampler::open(&dir.path().join("plan"), true)
        .await
        .unwrap();
    for sequence in [0, 9, 1, 8, 9] {
        let a = first.batch_at(sequence).await.unwrap().unwrap();
        let b = second.batch_at(sequence).await.unwrap().unwrap();
        assert_eq!(a.query_batch_idx, b.query_batch_idx);
        assert_eq!(a.samples[0].sample_id, b.samples[0].sample_id);
        assert_eq!(
            a.samples[0].sample_id,
            if sequence % 2 == 0 { "a" } else { "b" }
        );
    }
    let mut finite = QuerySampler::open(&dir.path().join("plan"), false)
        .await
        .unwrap();
    assert!(finite.batch_at(2).await.unwrap().is_none());
}

#[tokio::test]
async fn query_errors_and_cancellation_remove_partial_plans() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("plan");
    let rows = stream::iter([Ok(row(0, 0, "a")), Err(anyhow::anyhow!("query failed"))]);
    let error = QuerySampler::create("training", rows, &path)
        .await
        .err()
        .unwrap();
    assert!(error.to_string().contains("query failed"));
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    let rows = futures::StreamExt::chain(stream::iter([Ok(row(0, 0, "a"))]), stream::pending());
    assert!(
        tokio::time::timeout(
            std::time::Duration::from_millis(100),
            QuerySampler::create("training", rows, &path)
        )
        .await
        .is_err()
    );
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
}

#[tokio::test]
async fn large_batches_are_allowed() {
    let rows = (0..65_537).map(|i| row(0, i, "a")).collect();
    let (dir, mut large) = sampler(rows, false).await.unwrap();
    assert_eq!(
        large.batch_at(0).await.unwrap().unwrap().samples.len(),
        65_537
    );
    assert!(!dir.path().join("plan.part").exists());
}
