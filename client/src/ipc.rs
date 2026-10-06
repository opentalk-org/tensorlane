use std::marker::PhantomData;

use anyhow::{Result, ensure};
use futures_util::StreamExt;
use serde::{Serialize, de::DeserializeOwned};
use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt};
use tokio::net::UnixStream;
use tokio_util::codec::{FramedRead, LengthDelimitedCodec};

fn codec() -> LengthDelimitedCodec {
    LengthDelimitedCodec::builder()
        .length_field_length(8)
        .max_frame_length(usize::MAX)
        .new_codec()
}

pub struct Sender<T, Socket = UnixStream> {
    socket: Socket,
    marker: PhantomData<T>,
}

impl<T: Serialize, Socket: AsyncWrite + Unpin> Sender<T, Socket> {
    pub fn new(socket: Socket) -> Self {
        Self {
            socket,
            marker: PhantomData,
        }
    }

    pub async fn send(&mut self, value: &T) -> Result<()> {
        let bytes = postcard::to_allocvec(value)?;
        self.socket
            .write_all(&u64::try_from(bytes.len())?.to_be_bytes())
            .await?;
        self.socket.write_all(&bytes).await?;
        self.socket.flush().await?;
        Ok(())
    }
}

pub struct Receiver<T, Socket = UnixStream> {
    framed: FramedRead<Socket, LengthDelimitedCodec>,
    marker: PhantomData<T>,
}

impl<T: DeserializeOwned, Socket: AsyncRead + Unpin> Receiver<T, Socket> {
    pub fn new(socket: Socket) -> Self {
        Self {
            framed: FramedRead::new(socket, codec()),
            marker: PhantomData,
        }
    }

    pub async fn recv(&mut self) -> Result<Option<T>> {
        let Some(frame) = self.framed.next().await else {
            return Ok(None);
        };
        let frame = frame?;
        let (value, remaining) = postcard::take_from_bytes(&frame)?;
        ensure!(remaining.is_empty(), "trailing bytes in IPC message");
        Ok(Some(value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Work;
    use futures_util::SinkExt;
    use tokio::{
        io::AsyncWriteExt,
        time::{Duration, timeout},
    };
    use tokio_util::codec::FramedWrite;

    #[test]
    fn sample_blob_encoding_preserves_wire_format() -> Result<()> {
        let blobs = std::collections::HashMap::from([("audio".to_owned(), vec![23; 100_000])]);
        let sample = tensorlane_protocol::Sample {
            sample_id: "audio".into(),
            metadata_json: "{}".into(),
            blobs: blobs
                .iter()
                .map(|(name, bytes)| (name.clone(), bytes.clone().into()))
                .collect(),
        };
        let legacy = postcard::to_allocvec(&(&sample.sample_id, &sample.metadata_json, &blobs))?;
        assert_eq!(postcard::to_allocvec(&sample)?, legacy);
        assert_eq!(
            postcard::from_bytes::<tensorlane_protocol::Sample>(&legacy)?,
            sample
        );
        assert_eq!(
            serde_json::to_value(&sample)?["blobs"]["audio"]
                .as_array()
                .unwrap()
                .len(),
            100_000
        );
        Ok(())
    }

    #[tokio::test]
    async fn work_delivery_without_readiness_messages() -> Result<()> {
        let (daemon, worker) = UnixStream::pair()?;
        let mut data = Sender::new(daemon);
        let mut batches = Receiver::<Work>::new(worker);
        data.send(&Work::Batch {
            stream: "evaluation".into(),
            batch: (0, 1),
            query_batch_idx: 4,
            timings: [0.0; 3],
            memory_units: 0,
            samples: vec![(
                0,
                tensorlane_protocol::Sample {
                    sample_id: "sample".into(),
                    metadata_json: "{}".into(),
                    blobs: Default::default(),
                },
            )],
        })
        .await?;
        assert!(matches!(
            batches.recv().await?,
            Some(Work::Batch {
                batch: (0, 1),
                query_batch_idx: 4,
                ..
            })
        ));
        data.send(&Work::End {
            stream: "evaluation".into(),
        })
        .await?;
        assert!(matches!(batches.recv().await?, Some(Work::End { .. })));
        Ok(())
    }

    #[tokio::test]
    async fn dynamic_values_and_clean_eof() -> Result<()> {
        let (sender, receiver) = UnixStream::pair()?;
        let producer = tokio::spawn(async move {
            let mut sender = Sender::new(sender);
            sender.send(&vec![1u8; 100_000]).await?;
            sender.send(&Vec::<u8>::new()).await
        });
        let mut receiver = Receiver::<Vec<u8>>::new(receiver);
        let first = receiver.recv().await?.unwrap();
        assert_eq!(receiver.recv().await?, Some(vec![]));
        assert_eq!(receiver.recv().await?, None);
        producer.await??;
        assert_eq!(first, vec![1; 100_000]);
        Ok(())
    }

    #[tokio::test]
    async fn truncated_frames_are_errors() -> Result<()> {
        for bytes in [vec![0, 0], vec![0, 0, 0, 0, 0, 0, 0, 3, 1]] {
            let (mut sender, receiver) = UnixStream::pair()?;
            sender.write_all(&bytes).await?;
            drop(sender);
            assert!(Receiver::<Vec<u8>>::new(receiver).recv().await.is_err());
        }
        Ok(())
    }

    #[tokio::test]
    async fn invalid_postcard_payloads_are_errors() -> Result<()> {
        for payload in [vec![0x80], vec![0, 42]] {
            let (socket, receiver) = UnixStream::pair()?;
            let mut sender = FramedWrite::new(socket, codec());
            sender.send(payload.into()).await?;
            assert!(Receiver::<Vec<u8>>::new(receiver).recv().await.is_err());
        }
        Ok(())
    }

    #[tokio::test]
    async fn messages_larger_than_sixty_four_mib_are_delivered() -> Result<()> {
        let (sender, receiver) = UnixStream::pair()?;
        let size = 65 * 1024 * 1024;
        let producer = tokio::spawn(async move {
            Sender::new(sender)
                .send(&bytes::Bytes::from(vec![23; size]))
                .await
        });
        let bytes = Receiver::<bytes::Bytes>::new(receiver)
            .recv()
            .await?
            .unwrap();
        producer.await??;
        assert_eq!(bytes.len(), size);
        assert!(bytes.iter().all(|byte| *byte == 23));
        Ok(())
    }

    #[tokio::test]
    async fn receiver_disconnect_fails_sender() -> Result<()> {
        let (sender, receiver) = UnixStream::pair()?;
        drop(receiver);
        assert!(Sender::new(sender).send(&vec![1u8; 1024]).await.is_err());
        Ok(())
    }

    #[tokio::test]
    async fn zero_sized_serialized_value() -> Result<()> {
        let (sender, receiver) = UnixStream::pair()?;
        let mut sender = Sender::new(sender);
        sender.send(&()).await?;
        drop(sender);
        let mut receiver = Receiver::<()>::new(receiver);
        assert_eq!(receiver.recv().await?, Some(()));
        assert_eq!(receiver.recv().await?, None);
        Ok(())
    }

    #[tokio::test]
    async fn messages_larger_than_default_codec_limit() -> Result<()> {
        let (sender, receiver) = UnixStream::pair()?;
        let producer =
            tokio::spawn(
                async move { Sender::new(sender).send(&vec![1u8; 9 * 1024 * 1024]).await },
            );
        let mut receiver = Receiver::<Vec<u8>>::new(receiver);
        assert_eq!(receiver.recv().await?.unwrap(), vec![1; 9 * 1024 * 1024]);
        producer.await??;
        Ok(())
    }

    #[tokio::test]
    async fn blocked_sender_can_be_cancelled() -> Result<()> {
        let (sender, receiver) = UnixStream::pair()?;
        let mut sender = Sender::new(sender);
        assert!(
            timeout(
                Duration::from_millis(100),
                sender.send(&vec![1u8; 9 * 1024 * 1024])
            )
            .await
            .is_err()
        );
        drop(sender);
        drop(receiver);
        Ok(())
    }
}
