use anyhow::{Context, ensure};
use tonic::{
    Request, Status,
    metadata::{Ascii, MetadataValue},
    service::{Interceptor, interceptor::InterceptedService},
    transport::{Channel, ClientTlsConfig, Endpoint},
};

use crate::proto::tensor_lane_client::TensorLaneClient;

pub type GrpcClient = TensorLaneClient<InterceptedService<Channel, Authorization>>;

#[derive(Clone)]
pub struct Authorization {
    value: Option<MetadataValue<Ascii>>,
}

impl Authorization {
    fn new(key: Option<&str>) -> anyhow::Result<Self> {
        let value = key
            .map(|key| {
                ensure!(
                    key.len() >= 32 && key.bytes().all(|byte| byte.is_ascii_graphic()),
                    "API key must contain at least 32 printable ASCII characters without spaces"
                );
                let mut value: MetadataValue<Ascii> =
                    format!("Bearer {key}").parse().context("invalid API key")?;
                value.set_sensitive(true);
                anyhow::Ok(value)
            })
            .transpose()?;
        Ok(Self { value })
    }
}

impl Interceptor for Authorization {
    fn call(&mut self, mut request: Request<()>) -> Result<Request<()>, Status> {
        if let Some(value) = &self.value {
            request
                .metadata_mut()
                .insert("authorization", value.clone());
        }
        Ok(request)
    }
}

fn endpoint(addr: &str) -> anyhow::Result<Endpoint> {
    let url = if addr.contains("://") {
        addr.to_owned()
    } else {
        format!("http://{addr}")
    };
    let mut endpoint = Endpoint::from_shared(url)?;
    let tls = endpoint.uri().scheme_str() == Some("https");
    ensure!(
        tls || endpoint.uri().scheme_str() == Some("http"),
        "use an http:// or https:// server address"
    );
    if tls {
        static PROVIDER: std::sync::Once = std::sync::Once::new();
        PROVIDER.call_once(|| {
            let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        });
        endpoint = endpoint.tls_config(ClientTlsConfig::new().with_native_roots())?;
    }
    Ok(endpoint.connect_timeout(std::time::Duration::from_secs(10)))
}

pub async fn connect(addr: &str, key: Option<&str>) -> anyhow::Result<GrpcClient> {
    let auth = Authorization::new(key)?;
    let channel = endpoint(addr)?.connect().await?;
    Ok(TensorLaneClient::with_interceptor(channel, auth)
        .max_decoding_message_size(crate::MAX_BATCH_BYTES)
        .max_encoding_message_size(crate::MAX_BATCH_BYTES))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn credentials_are_sensitive_and_follow_cloned_clients() {
        let mut auth = Authorization::new(Some("0123456789abcdef0123456789abcdef"))
            .unwrap()
            .clone();
        for _ in 0..3 {
            let request = auth.call(Request::new(())).unwrap();
            let value = request.metadata().get("authorization").unwrap();
            assert!(value.is_sensitive());
            assert_eq!(value, "Bearer 0123456789abcdef0123456789abcdef");
        }
        assert!(
            Authorization::new(None)
                .unwrap()
                .call(Request::new(()))
                .unwrap()
                .metadata()
                .is_empty()
        );
        assert!(Authorization::new(Some("")).is_err());
    }

    #[tokio::test]
    async fn supports_http_and_https_addresses() {
        for addr in [
            "https://example.com:443",
            "http://example.com:8181",
            "example.com:8181",
            "localhost:8181",
            "http://127.0.0.1:8181",
            "http://[::1]:8181",
        ] {
            assert!(endpoint(addr).is_ok(), "{addr}");
        }
        assert!(endpoint("ftp://example.com:8181").is_err());
    }
}
