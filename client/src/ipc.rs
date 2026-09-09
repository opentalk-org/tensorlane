use std::marker::PhantomData;

use anyhow::{Result, ensure};
use futures_util::{SinkExt, StreamExt};
use serde::{Serialize, de::DeserializeOwned};
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::UnixStream;
use tokio_util::codec::{FramedRead, FramedWrite, LengthDelimitedCodec};

const MAX_FRAME_LENGTH: usize = 256 * 1024 * 1024;

fn codec() -> LengthDelimitedCodec {
    LengthDelimitedCodec::builder()
        .max_frame_length(MAX_FRAME_LENGTH)
        .new_codec()
}

pub struct Sender<T, Socket = UnixStream> {
    framed: FramedWrite<Socket, LengthDelimitedCodec>,
    marker: PhantomData<T>,
}

impl<T: Serialize, Socket: AsyncWrite + Unpin> Sender<T, Socket> {
    pub fn new(socket: Socket) -> Self {
        Self {
            framed: FramedWrite::new(socket, codec()),
            marker: PhantomData,
        }
    }

    pub async fn send(&mut self, value: &T) -> Result<()> {
        self.framed
            .send(postcard::to_allocvec(value)?.into())
            .await?;
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
    use tokio::{
        io::AsyncWriteExt,
        time::{Duration, timeout},
    };

    #[tokio::test]
    async fn work_delivery_without_readiness_messages() -> Result<()> {
        let (daemon, worker) = UnixStream::pair()?;
        let mut data = Sender::new(daemon);
        let mut batches = Receiver::<Work>::new(worker);
        data.send(&Work::Sample {
            batch: (0, 1),
            index: 0,
            wave: vec![0, 0],
            text: vec![],
            duration: 0.5,
            speaker_id: 0,
            language_id: 0,
        })
        .await?;
        assert!(matches!(
            batches.recv().await?,
            Some(Work::Sample {
                batch: (0, 1),
                index: 0,
                ..
            })
        ));
        data.send(&Work::End).await?;
        assert!(matches!(batches.recv().await?, Some(Work::End)));
        drop(data);
        assert!(batches.recv().await?.is_none());
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
        for bytes in [vec![0, 0], vec![0, 0, 0, 3, 1]] {
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
    async fn oversized_header_is_rejected() -> Result<()> {
        let (mut sender, receiver) = UnixStream::pair()?;
        sender
            .write_all(&((MAX_FRAME_LENGTH + 1) as u32).to_be_bytes())
            .await?;
        assert!(Receiver::<Vec<u8>>::new(receiver).recv().await.is_err());
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
