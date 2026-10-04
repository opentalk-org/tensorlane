use axum::{
    extract::{Request, State},
    http::{StatusCode, header},
    middleware::Next,
    response::{IntoResponse, Response},
};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;
use tonic::{Request as GrpcRequest, Status, service::Interceptor};

#[derive(Clone)]
pub struct Auth {
    digest: Option<[u8; 32]>,
}

impl Auth {
    pub fn new(key: Option<&str>, allow_unauthenticated: bool) -> anyhow::Result<Self> {
        let digest = match key {
            Some(key) => {
                anyhow::ensure!(
                    key.len() >= 32 && key.bytes().all(|byte| byte.is_ascii_graphic()),
                    "TENSORLANE_API_KEY must contain at least 32 printable ASCII characters without spaces"
                );
                Some(Sha256::digest(key.as_bytes()).into())
            }
            None => {
                anyhow::ensure!(
                    allow_unauthenticated,
                    "set TENSORLANE_API_KEY, or explicitly use --allow-unauthenticated for local development"
                );
                None
            }
        };
        Ok(Self { digest })
    }

    fn accepts(&self, authorization: Option<&str>) -> bool {
        let Some(expected) = self.digest else {
            return true;
        };
        let Some((scheme, key)) = authorization.and_then(|value| value.split_once(' ')) else {
            return false;
        };
        if !scheme.eq_ignore_ascii_case("Bearer") || key.is_empty() {
            return false;
        }
        let received: [u8; 32] = Sha256::digest(key.as_bytes()).into();
        bool::from(expected.ct_eq(&received))
    }
}

pub async fn http(State(auth): State<Auth>, request: Request, next: Next) -> Response {
    let mut values = request.headers().get_all(header::AUTHORIZATION).iter();
    let value = values.next().and_then(|value| value.to_str().ok());
    if values.next().is_some() || !auth.accepts(value) {
        return (
            StatusCode::UNAUTHORIZED,
            [(header::WWW_AUTHENTICATE, "Bearer")],
            "Unauthorized",
        )
            .into_response();
    }
    next.run(request).await
}

impl Interceptor for Auth {
    fn call(&mut self, request: GrpcRequest<()>) -> Result<GrpcRequest<()>, Status> {
        let mut values = request.metadata().get_all("authorization").iter();
        let value = values.next().and_then(|value| value.to_str().ok());
        if values.next().is_some() || !self.accepts(value) {
            return Err(Status::unauthenticated("Unauthorized"));
        }
        Ok(request)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const KEY: &str = "0123456789abcdef0123456789abcdef";

    #[test]
    fn startup_requires_a_key_or_explicit_local_opt_out() {
        assert!(Auth::new(None, false).is_err());
        assert!(Auth::new(None, true).unwrap().accepts(None));
        for key in ["", "short", "0123456789abcdef0123456789abcdef\n"] {
            assert!(Auth::new(Some(key), true).is_err());
        }
        assert!(!Auth::new(Some(KEY), true).unwrap().accepts(None));
    }

    #[test]
    fn grpc_rejects_missing_wrong_and_duplicate_credentials() {
        let mut auth = Auth::new(Some(KEY), false).unwrap();
        for value in [None, Some("Basic ignored"), Some("Bearer wrong")] {
            let mut request = GrpcRequest::new(());
            if let Some(value) = value {
                request
                    .metadata_mut()
                    .insert("authorization", value.parse().unwrap());
            }
            assert_eq!(
                auth.call(request).unwrap_err().code(),
                tonic::Code::Unauthenticated
            );
        }
        let mut request = GrpcRequest::new(());
        request
            .metadata_mut()
            .insert("authorization", format!("bearer {KEY}").parse().unwrap());
        assert!(auth.call(request).is_ok());
        let mut request = GrpcRequest::new(());
        request
            .metadata_mut()
            .insert("authorization", format!("Bearer {KEY}").parse().unwrap());
        request
            .metadata_mut()
            .append("authorization", "Bearer wrong".parse().unwrap());
        assert_eq!(
            auth.call(request).unwrap_err().code(),
            tonic::Code::Unauthenticated
        );
    }
}
