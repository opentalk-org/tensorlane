use axum::{
    extract::{Request, State},
    http::{StatusCode, header},
    middleware::Next,
    response::{IntoResponse, Response},
};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

#[derive(Clone)]
pub struct Auth {
    digest: Option<[u8; 32]>,
}

impl Auth {
    pub fn new(key: Option<&str>) -> anyhow::Result<Self> {
        let digest = match key {
            Some(key) => {
                anyhow::ensure!(
                    key.len() >= 32 && key.bytes().all(|byte| byte.is_ascii_graphic()),
                    "TENSORLANE_API_KEY must contain at least 32 printable ASCII characters without spaces"
                );
                Some(Sha256::digest(key.as_bytes()).into())
            }
            None => None,
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

#[cfg(test)]
mod tests {
    use super::*;

    const KEY: &str = "0123456789abcdef0123456789abcdef";

    #[test]
    fn authentication_is_optional_but_configured_keys_are_enforced() {
        let anonymous = Auth::new(None).unwrap();
        assert!(anonymous.accepts(None));
        for key in ["", "short", "0123456789abcdef0123456789abcdef\n"] {
            assert!(Auth::new(Some(key)).is_err());
        }
        assert!(!Auth::new(Some(KEY)).unwrap().accepts(None));
    }

    #[test]
    fn accepts_only_matching_bearer_credentials() {
        let auth = Auth::new(Some(KEY)).unwrap();
        for value in [None, Some("Basic ignored"), Some("Bearer wrong")] {
            assert!(!auth.accepts(value));
        }
        assert!(auth.accepts(Some(&format!("bearer {KEY}"))));
    }
}
