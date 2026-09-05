use std::sync::{Arc, Mutex};

use anyhow::Context;
use flume::Sender;
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;

use crate::audio::AudioProcessor;
use crate::data::{NativeBatch, process_response};
use crate::proto::tensor_lane_client::TensorLaneClient;
use crate::proto::{DataRequest, Split};

pub struct Pipeline {
    pub run_id: String,
    pub validation: bool,
    pub prefetch: usize,
    pub modality_id: i64,
    pub pin_memory: bool,
    pub output: Sender<anyhow::Result<NativeBatch>>,
    pub cancellation: CancellationToken,
}

impl Pipeline {
    pub async fn run(
        self,
        client: &mut TensorLaneClient<tonic::transport::Channel>,
    ) -> anyhow::Result<()> {
        let Self {
            run_id,
            validation,
            prefetch,
            modality_id,
            pin_memory,
            output,
            cancellation,
        } = self;
        let split = if validation {
            Split::Validation
        } else {
            Split::Training
        };
        let request = DataRequest {
            run_id,
            split: split as i32,
        };
        let (request_tx, request_rx) = tokio::sync::mpsc::channel(prefetch);
        for _ in 0..prefetch {
            request_tx.send(request.clone()).await?;
        }
        let mut responses = client
            .data(ReceiverStream::new(request_rx))
            .await?
            .into_inner();
        let (response_tx, mut response_rx) = tokio::sync::mpsc::channel(prefetch);
        let receive_cancellation = cancellation.clone();

        let receive = async move {
            loop {
                let response = tokio::select! {
                    _ = receive_cancellation.cancelled() => return anyhow::Ok(()),
                    response = responses.message() => response?,
                };
                let Some(response) = response else {
                    return anyhow::Ok(());
                };
                let sent = tokio::select! {
                    _ = receive_cancellation.cancelled() => return anyhow::Ok(()),
                    sent = response_tx.send(response) => sent,
                };
                if sent.is_err() {
                    return anyhow::Ok(());
                }
            }
        };

        let process = async {
            let processor = Arc::new(Mutex::new(AudioProcessor::new()?));
            let mut requests_open = true;
            loop {
                let response = tokio::select! {
                    _ = cancellation.cancelled() => return anyhow::Ok(()),
                    response = response_rx.recv() => response,
                };
                let Some(response) = response else {
                    return anyhow::Ok(());
                };
                let processor = processor.clone();
                let batch = tokio::task::spawn_blocking(move || {
                    let processor = processor
                        .lock()
                        .map_err(|_| anyhow::anyhow!("audio processor lock is poisoned"))?;
                    process_response(&processor, response, modality_id, pin_memory)
                })
                .await
                .context("joining batch processor")?;
                let sent = tokio::select! {
                    _ = cancellation.cancelled() => return anyhow::Ok(()),
                    sent = output.send_async(batch) => sent,
                };
                if sent.is_err() {
                    return anyhow::Ok(());
                }
                if requests_open && request_tx.send(request.clone()).await.is_err() {
                    requests_open = false;
                }
            }
        };

        let (receive_result, process_result) = tokio::join!(receive, process);
        receive_result?;
        process_result
    }
}
