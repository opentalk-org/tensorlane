use crate::{
    proto,
    setup::{TestEnv, eventually},
};
use tonic::{Code, Request};

const KEY: &str = "0123456789abcdef0123456789abcdef";

#[tokio::test]
async fn protected_http_and_every_grpc_method_reject_anonymous_requests() -> anyhow::Result<()> {
    let mut env = TestEnv::start().await?;
    env.server.kill()?;
    env.server.wait()?;
    env.server_command.env("TENSORLANE_API_KEY", KEY);
    env.server = env.server_command.spawn()?;
    eventually("authenticated server startup", || async {
        Ok(env
            .http
            .get(format!("{}/runs", env.http_url))
            .bearer_auth(KEY)
            .send()
            .await
            .ok()
            .filter(|response| response.status().is_success())
            .map(|_| ()))
    })
    .await?;

    let response = env
        .http
        .get(format!("{}/runs", env.http_url))
        .send()
        .await?;
    assert_eq!(response.status(), reqwest::StatusCode::UNAUTHORIZED);
    let response = env
        .http
        .get(format!("{}/runs", env.http_url))
        .bearer_auth("wrong")
        .send()
        .await?;
    assert_eq!(response.status(), reqwest::StatusCode::UNAUTHORIZED);

    let channel = tonic::transport::Endpoint::from_shared(env.grpc_url.clone())?
        .connect()
        .await?;
    let mut grpc = proto::tensor_lane_client::TensorLaneClient::new(channel);
    assert_eq!(
        grpc.init(proto::InitRequest::default())
            .await
            .unwrap_err()
            .code(),
        Code::Unauthenticated
    );
    assert_eq!(
        grpc.data(tokio_stream::empty::<proto::DataRequest>())
            .await
            .unwrap_err()
            .code(),
        Code::Unauthenticated
    );
    assert_eq!(
        grpc.asset(proto::AssetRequest::default())
            .await
            .unwrap_err()
            .code(),
        Code::Unauthenticated
    );
    assert_eq!(
        grpc.save_asset(tokio_stream::empty::<proto::SaveAssetRequest>())
            .await
            .unwrap_err()
            .code(),
        Code::Unauthenticated
    );
    assert_eq!(
        grpc.metrics(tokio_stream::empty::<proto::MetricsRequest>())
            .await
            .unwrap_err()
            .code(),
        Code::Unauthenticated
    );
    assert_eq!(
        grpc.heartbeat(proto::HeartbeatRequest::default())
            .await
            .unwrap_err()
            .code(),
        Code::Unauthenticated
    );
    assert_eq!(
        grpc.end(proto::EndRequest::default())
            .await
            .unwrap_err()
            .code(),
        Code::Unauthenticated
    );

    let mut request = Request::new(proto::InitRequest::default());
    request
        .metadata_mut()
        .insert("authorization", "Bearer wrong".parse()?);
    assert_eq!(
        grpc.init(request).await.unwrap_err().code(),
        Code::Unauthenticated
    );
    let mut request = Request::new(proto::InitRequest::default());
    request
        .metadata_mut()
        .insert("authorization", format!("Bearer {KEY}").parse()?);
    assert_eq!(
        grpc.init(request).await.unwrap_err().code(),
        Code::InvalidArgument
    );
    Ok(())
}

#[tokio::test]
async fn grpc_tls_verifies_the_certificate_and_keeps_authentication_required() -> anyhow::Result<()>
{
    let mut env = TestEnv::start().await?;
    let files = tempfile::tempdir()?;
    let cert = files.path().join("cert.pem");
    let key = files.path().join("key.pem");
    let status = std::process::Command::new("openssl")
        .args([
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-noenc",
            "-days",
            "1",
            "-subj",
            "/CN=localhost",
            "-addext",
            "subjectAltName=DNS:localhost",
            "-addext",
            "basicConstraints=critical,CA:FALSE",
            "-addext",
            "extendedKeyUsage=serverAuth",
            "-keyout",
        ])
        .arg(&key)
        .arg("-out")
        .arg(&cert)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()?;
    anyhow::ensure!(status.success(), "generating local test certificate failed");
    env.server.kill()?;
    env.server.wait()?;
    env.server_command
        .env("TENSORLANE_API_KEY", KEY)
        .env("GRPC_TLS_CERT_FILE", &cert)
        .env("GRPC_TLS_KEY_FILE", &key);
    env.server = env.server_command.spawn()?;
    eventually("TLS server startup", || async {
        Ok(env
            .http
            .get(format!("{}/runs", env.http_url))
            .bearer_auth(KEY)
            .send()
            .await
            .ok()
            .filter(|response| response.status().is_success())
            .map(|_| ()))
    })
    .await?;
    let url = env.grpc_url.replace("http://", "https://");
    let endpoint = tonic::transport::Endpoint::from_shared(url)?;
    let roots = tonic::transport::Certificate::from_pem(std::fs::read(&cert)?);
    let channel = endpoint
        .clone()
        .tls_config(
            tonic::transport::ClientTlsConfig::new()
                .domain_name("localhost")
                .ca_certificate(roots.clone()),
        )?
        .connect()
        .await?;
    let mut grpc = proto::tensor_lane_client::TensorLaneClient::new(channel);
    assert_eq!(
        grpc.init(proto::InitRequest::default())
            .await
            .unwrap_err()
            .code(),
        Code::Unauthenticated
    );
    let mut request = Request::new(proto::InitRequest::default());
    request
        .metadata_mut()
        .insert("authorization", format!("Bearer {KEY}").parse()?);
    assert_eq!(
        grpc.init(request).await.unwrap_err().code(),
        Code::InvalidArgument
    );
    assert!(
        endpoint
            .clone()
            .tls_config(
                tonic::transport::ClientTlsConfig::new()
                    .domain_name("other.example")
                    .ca_certificate(roots)
            )?
            .connect()
            .await
            .is_err()
    );
    assert!(
        endpoint
            .tls_config(tonic::transport::ClientTlsConfig::new().domain_name("localhost"))?
            .connect()
            .await
            .is_err()
    );
    Ok(())
}
