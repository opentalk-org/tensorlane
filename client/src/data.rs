use anyhow::{Context, anyhow};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Streaming, transport::Channel};

use crate::proto::{DataRequest, DataResponse, tensor_lane_client::TensorLaneClient};

#[derive(Default)]
pub struct DataStream {
    connection: Option<(mpsc::Sender<DataRequest>, Streaming<DataResponse>)>,
    exhausted: bool,
    failure: Option<String>,
}

impl DataStream {
    pub async fn next(
        &mut self,
        grpc: &mut TensorLaneClient<Channel>,
        request: DataRequest,
    ) -> anyhow::Result<Option<DataResponse>> {
        if let Some(error) = &self.failure {
            return Err(anyhow!(error.clone()));
        }
        if self.exhausted {
            return Ok(None);
        }
        // Bound waits so an unavailable service cannot keep close() waiting forever.
        let result = match tokio::time::timeout(
            std::time::Duration::from_secs(30),
            self.fetch(grpc, request),
        )
        .await
        {
            Ok(result) => result,
            Err(_) => Err(anyhow!("TensorLane Data request timed out")),
        };
        match &result {
            Ok(None) => {
                self.exhausted = true;
                self.connection = None;
            }
            Err(error) => {
                // Never retry implicitly: the server may already have advanced its cursor.
                self.failure = Some(format!("{error:#}"));
                self.connection = None;
            }
            _ => {}
        }
        result
    }

    async fn fetch(
        &mut self,
        grpc: &mut TensorLaneClient<Channel>,
        request: DataRequest,
    ) -> anyhow::Result<Option<DataResponse>> {
        if let Some((sender, _)) = &self.connection {
            sender
                .send(request)
                .await
                .context("TensorLane Data request stream closed")?;
        } else {
            let (sender, receiver) = mpsc::channel(1);
            // Queue the first request before opening the bidirectional RPC.
            sender
                .send(request)
                .await
                .context("sending first Data request")?;
            let response = grpc
                .data(ReceiverStream::new(receiver))
                .await
                .context("opening TensorLane Data stream")?
                .into_inner();
            self.connection = Some((sender, response));
        }
        self.connection
            .as_mut()
            .unwrap()
            .1
            .message()
            .await
            .context("receiving TensorLane batch")
    }
}
