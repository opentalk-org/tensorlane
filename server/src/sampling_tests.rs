use super::{QuerySampler, Sampler};
use crate::db::SampleRow;
use uuid::Uuid;

fn row(batch_idx: u64, sample_idx: u64, id: u128) -> SampleRow {
    SampleRow {
        audio_id: Uuid::from_u128(id),
        duration: 2.0,
        language: Some("en".into()),
        speaker_id: None,
        text: Some("tˈuː".into()),
        batch_idx,
        sample_idx,
        object_path: "audio.tar".into(),
        byte_offset: 0,
        byte_length: 100,
    }
}

#[test]
fn groups_query_rows_and_preserves_repeated_samples() {
    let rows = vec![row(0, 0, 1), row(0, 1, 2), row(4, 0, 1)];
    let mut sampler = QuerySampler::new(rows, &[]).unwrap();
    assert_eq!(sampler.len(), 2);
    let first = sampler.next_batch().unwrap().unwrap();
    assert_eq!(
        first
            .iter()
            .map(|s| s.audio_id.as_u128())
            .collect::<Vec<_>>(),
        vec![1, 2]
    );
    assert_eq!(sampler.len(), 1);
    let last = sampler.next_batch().unwrap().unwrap();
    assert_eq!(last.len(), 1);
    assert_eq!(last[0].audio_id.as_u128(), 1);
    assert!(sampler.next_batch().unwrap().is_none());
    assert!(sampler.next_batch().unwrap().is_none());
}

#[test]
fn rejects_unordered_or_duplicate_positions() {
    for positions in [[(2, 0), (1, 0)], [(0, 2), (0, 1)], [(0, 1), (0, 1)]] {
        let rows = positions
            .into_iter()
            .map(|(batch, sample)| row(batch, sample, 1))
            .collect();
        assert!(QuerySampler::new(rows, &[]).is_err());
    }
}

#[test]
fn empty_query_ends_and_missing_metadata_fails() {
    let mut sampler = QuerySampler::new(vec![], &[]).unwrap();
    assert!(sampler.next_batch().unwrap().is_none());
    let mut missing_text = row(0, 0, 1);
    missing_text.text = None;
    assert!(QuerySampler::new(vec![missing_text], &[]).is_err());
    let mut missing_language = row(0, 0, 1);
    missing_language.language = None;
    assert!(QuerySampler::new(vec![missing_language], &[]).is_err());
}

#[test]
fn validation_replays_identical_batches_across_passes() {
    let mut sampler = QuerySampler::new(vec![row(0, 0, 1), row(0, 1, 2), row(1, 2, 3)], &[])
        .unwrap()
        .repeat();
    for _ in 0..3 {
        let first = sampler.next_batch().unwrap().unwrap();
        assert_eq!(
            first
                .iter()
                .map(|s| s.audio_id.as_u128())
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert_eq!(
            sampler.next_batch().unwrap().unwrap()[0].audio_id.as_u128(),
            3
        );
    }
    let mut empty = QuerySampler::new(vec![], &[]).unwrap().repeat();
    assert!(empty.next_batch().unwrap().is_none());
}
