use anyhow::{Context, Result};
use aws_sdk_s3::{error::ProvideErrorMetadata, primitives::ByteStream};
use serde::{Serialize, de::DeserializeOwned};

use crate::runtime::Runtime;

impl Runtime {
    pub async fn read_state<T: DeserializeOwned>(&self, key: &str) -> Result<Option<T>> {
        let key = format!("{}/.tensorlane/{key}", self.checkpoint_prefix);
        match self
            .s3
            .get_object()
            .bucket(self.bucket)
            .key(key)
            .send()
            .await
        {
            Ok(object) => Ok(Some(serde_json::from_slice(
                &object.body.collect().await?.into_bytes(),
            )?)),
            Err(error) if error.as_service_error().is_some_and(|e| e.is_no_such_key()) => Ok(None),
            Err(error) => Err(error.into()),
        }
    }

    pub async fn create_state<T: Serialize + DeserializeOwned + Clone>(
        &self,
        key: &str,
        value: &T,
    ) -> Result<T> {
        let object = format!("{}/.tensorlane/{key}", self.checkpoint_prefix);
        match self
            .s3
            .put_object()
            .bucket(self.bucket)
            .key(object)
            .if_none_match("*")
            .content_type("application/json")
            .body(ByteStream::from(serde_json::to_vec(value)?))
            .send()
            .await
        {
            Ok(_) => Ok(value.clone()),
            Err(error)
                if error.as_service_error().is_some_and(|e| {
                    matches!(
                        e.code(),
                        Some("PreconditionFailed" | "ConditionalRequestConflict")
                    )
                }) =>
            {
                self.read_state(key)
                    .await?
                    .context("state object disappeared")
            }
            Err(error) => Err(error.into()),
        }
    }
}
