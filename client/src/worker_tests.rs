use super::*;
use crate::proto::tensor_lane_server::{TensorLane, TensorLaneServer};
use crate::proto::*;
use std::pin::Pin;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use tokio_stream::{Stream, wrappers::TcpListenerStream};
use tonic::{Request, Response, Status, Streaming};

type ReplyStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send>>;

struct Service {
    requests: Arc<Mutex<Vec<DataRequest>>>,
    fail_data: Arc<AtomicBool>,
}

#[tonic::async_trait]
impl TensorLane for Service {
    async fn init(&self, request: Request<InitRequest>) -> Result<Response<InitResponse>, Status> {
        if request.into_inner().run_id != "requested-run" {
            return Err(Status::invalid_argument("unknown run"));
        }
        Ok(Response::new(InitResponse {
            run_id: "server-run".into(),
            train_config: "{\"batch_size\":8}".into(),
        }))
    }
    type DataStream = ReplyStream<DataResponse>;
    async fn data(
        &self,
        request: Request<Streaming<DataRequest>>,
    ) -> Result<Response<Self::DataStream>, Status> {
        let mut input = request.into_inner();
        let requests = self.requests.clone();
        let fail_data = self.fail_data.clone();
        let (sender, receiver) = mpsc::channel(1);
        tokio::spawn(async move {
            let mut served = false;
            while let Some(request) = input.message().await.unwrap() {
                requests.lock().unwrap().push(request.clone());
                if fail_data.load(Ordering::SeqCst) {
                    let _ = sender
                        .send(Err(Status::internal("fixture data error")))
                        .await;
                    break;
                }
                if served {
                    break;
                }
                served = true;
                let _ = sender
                    .send(Ok(DataResponse {
                        batch: vec![Sample {
                            wave: vec![0, 255, 1],
                            duration: 0.5,
                            speaker_id: 42,
                            language_id: request.split,
                            text: vec![2, 0, 3],
                        }],
                    }))
                    .await;
            }
        });
        Ok(Response::new(Box::pin(
            tokio_stream::wrappers::ReceiverStream::new(receiver),
        )))
    }
    type AssetStream = ReplyStream<AssetResponse>;
    async fn asset(&self, _: Request<AssetRequest>) -> Result<Response<Self::AssetStream>, Status> {
        Err(Status::unimplemented("not in this test"))
    }
    async fn checkpoint(
        &self,
        _: Request<Streaming<CheckpointRequest>>,
    ) -> Result<Response<CheckpointResponse>, Status> {
        Err(Status::unimplemented("not in this test"))
    }
    async fn metrics(
        &self,
        _: Request<Streaming<MetricsRequest>>,
    ) -> Result<Response<MetricsResponse>, Status> {
        Err(Status::unimplemented("not in this test"))
    }
    async fn end(&self, _: Request<EndRequest>) -> Result<Response<EndResponse>, Status> {
        Err(Status::unimplemented("not in this test"))
    }
}

struct Server {
    requests: Arc<Mutex<Vec<DataRequest>>>,
    fail_data: Arc<AtomicBool>,
    addr: String,
    stop: Option<oneshot::Sender<()>>,
    thread: Option<JoinHandle<()>>,
}

impl Server {
    fn start() -> Self {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        listener.set_nonblocking(true).unwrap();
        let (stop, stopped) = oneshot::channel();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let fail_data = Arc::new(AtomicBool::new(false));
        let service = Service {
            requests: requests.clone(),
            fail_data: fail_data.clone(),
        };
        let thread = thread::spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(async {
                    let listener = tokio::net::TcpListener::from_std(listener).unwrap();
                    tonic::transport::Server::builder()
                        .add_service(TensorLaneServer::new(service))
                        .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async {
                            let _ = stopped.await;
                        })
                        .await
                        .unwrap();
                });
        });
        Self {
            requests,
            fail_data,
            addr,
            stop: Some(stop),
            thread: Some(thread),
        }
    }
}

fn batch(client: &Worker, validation: bool) -> anyhow::Result<Option<DataResponse>> {
    let (reply, response) = oneshot::channel();
    client.send(Command::NextBatch { validation, reply })?;
    response.blocking_recv().unwrap()
}

#[test]
fn data_is_lazy_on_demand_and_exhaustion_is_cached_per_split() {
    let server = Server::start();
    let (mut client, _) = Worker::start("requested-run".into(), server.addr.clone()).unwrap();
    assert!(server.requests.lock().unwrap().is_empty());
    let first = batch(&client, false).unwrap().unwrap();
    assert_eq!(first.batch[0].wave, vec![0, 255, 1]);
    assert_eq!(first.batch[0].text, vec![2, 0, 3]);
    assert_eq!(server.requests.lock().unwrap().len(), 1);
    assert!(batch(&client, false).unwrap().is_none());
    assert!(batch(&client, false).unwrap().is_none());
    assert_eq!(server.requests.lock().unwrap().len(), 2);
    assert_eq!(
        batch(&client, true).unwrap().unwrap().batch[0].language_id,
        1
    );
    let requests = server.requests.lock().unwrap();
    assert_eq!(requests.len(), 3);
    assert!(requests.iter().all(|r| r.run_id == "server-run"));
    drop(requests);
    client.shutdown().unwrap();
    assert!(batch(&client, false).is_err());
}

#[test]
fn data_errors_propagate_without_implicit_retry() {
    let server = Server::start();
    server.fail_data.store(true, Ordering::SeqCst);
    let (client, _) = Worker::start("requested-run".into(), server.addr.clone()).unwrap();
    for _ in 0..2 {
        assert!(format!("{:#}", batch(&client, false).unwrap_err()).contains("fixture data error"));
    }
    assert_eq!(server.requests.lock().unwrap().len(), 1);
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.stop.take().unwrap().send(());
        self.thread.take().unwrap().join().unwrap();
    }
}

#[test]
fn initializes_before_returning_and_drains_individual_replies() {
    let server = Server::start();
    let (mut client, initialized) =
        Worker::start("requested-run".into(), server.addr.clone()).unwrap();
    assert_eq!(initialized.run_id, "server-run");
    assert_eq!(initialized.train_config, "{\"batch_size\":8}");
    let replies: Vec<_> = [false, true]
        .into_iter()
        .map(|validation| {
            let (reply, response) = oneshot::channel();
            client
                .send(Command::NextBatch { validation, reply })
                .unwrap();
            response
        })
        .collect();
    client.shutdown().unwrap();
    for (i, reply) in replies.into_iter().enumerate() {
        assert_eq!(
            reply.blocking_recv().unwrap().unwrap().unwrap().batch[0].language_id,
            i as i32
        );
    }
    client.shutdown().unwrap();
    assert!(batch(&client, false).is_err());
}

#[test]
fn rejected_init_returns_server_error() {
    let server = Server::start();
    let error = Worker::start("wrong-run".into(), server.addr.clone())
        .err()
        .unwrap();
    assert!(format!("{error:#}").contains("unknown run"));
}

#[test]
fn invalid_address_returns_error() {
    assert!(Worker::start("requested-run".into(), "http://[invalid".into()).is_err());
}

#[test]
fn abandoned_reply_and_drop_still_work_after_init() {
    let server = Server::start();
    let (client, _) =
        Worker::start("requested-run".into(), format!("http://{}", server.addr)).unwrap();
    let (reply, abandoned) = oneshot::channel();
    client
        .send(Command::NextBatch {
            validation: false,
            reply,
        })
        .unwrap();
    drop(abandoned);
    let (reply, response) = oneshot::channel();
    client
        .send(Command::NextBatch {
            validation: true,
            reply,
        })
        .unwrap();
    drop(client);
    assert_eq!(
        response.blocking_recv().unwrap().unwrap().unwrap().batch[0].language_id,
        1
    );
}
