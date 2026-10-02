use super::{BlobRef, QuerySampler, Sampler};
use crate::db::SampleRow;
fn row(batch: u64, index: u64, id: &str) -> SampleRow {
    SampleRow {
        sample_id: id.into(),
        batch_idx: batch,
        sample_idx: index,
        metadata_json: r#"{"text":"hello","nested":{"label":3}}"#.into(),
        blobs_json: "{}".into(),
    }
}
#[test]
fn groups_sparse_batches_and_preserves_repeated_samples() {
    let mut sampler =
        QuerySampler::new(vec![row(0, 0, "a"), row(0, 1, "b"), row(4, 0, "a")]).unwrap();
    assert_eq!(sampler.len(), 2);
    let first = sampler.next_batch().unwrap().unwrap();
    assert_eq!(
        first
            .samples
            .iter()
            .map(|s| s.sample_id.as_str())
            .collect::<Vec<_>>(),
        vec!["a", "b"]
    );
    assert_eq!(sampler.next_batch().unwrap().unwrap().query_batch_idx, 4);
    assert!(sampler.next_batch().unwrap().is_none());
}
#[test]
fn rejects_unordered_duplicate_or_invalid_rows() {
    for positions in [[(2, 0), (1, 0)], [(0, 2), (0, 1)], [(0, 1), (0, 1)]] {
        assert!(
            QuerySampler::new(positions.into_iter().map(|(b, i)| row(b, i, "a")).collect())
                .is_err()
        );
    }
    for (meta, blobs) in [
        ("[]", "{}"),
        ("{}", "[]"),
        ("{}", r#"{"x":{"object":"a","byte_offset":0}}"#),
    ] {
        let mut sample = row(0, 0, "a");
        sample.metadata_json = meta.into();
        sample.blobs_json = blobs.into();
        assert!(QuerySampler::new(vec![sample]).is_err());
    }
}
#[test]
fn repeats_identical_plans_and_empty_repetition_ends() {
    let mut sampler = QuerySampler::new(vec![row(7, 0, "a")]).unwrap().repeat();
    for _ in 0..3 {
        assert_eq!(sampler.next_batch().unwrap().unwrap().query_batch_idx, 7);
    }
    assert!(
        QuerySampler::new(vec![])
            .unwrap()
            .repeat()
            .next_batch()
            .unwrap()
            .is_none()
    );
}
#[test]
fn validates_ranges() {
    for (offset, length, valid) in [
        (None, None, true),
        (Some(0), Some(1), true),
        (Some(1), None, false),
        (None, Some(1), false),
        (Some(0), Some(0), false),
        (Some(u64::MAX), Some(2), false),
    ] {
        assert_eq!(
            BlobRef {
                object: "pack".into(),
                byte_offset: offset,
                byte_length: length
            }
            .validate()
            .is_ok(),
            valid
        );
    }
}
