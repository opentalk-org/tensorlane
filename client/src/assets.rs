use crate::proto::{
    AssetRequest, InitResponse, asset_response::Payload, tensor_lane_client::TensorLaneClient,
};
use anyhow::Context;
use futures_util::{StreamExt, TryStreamExt, stream};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};
use tokio::{fs, io::AsyncWriteExt};
use tonic::transport::Channel;

pub async fn prefetch(
    grpc: &TensorLaneClient<Channel>,
    initialized: &InitResponse,
    root: &Path,
) -> anyhow::Result<(HashMap<String, PathBuf>, HashMap<String, String>)> {
    let downloads: Vec<_> = stream::iter(initialized.assets.iter().enumerate())
        .map(|(index, name)| {
            let grpc = grpc.clone();
            let destination = root.join("assets").join(index.to_string());
            async move {
                let (path, metadata) = download(grpc, &initialized.run_id, name, destination)
                    .await
                    .with_context(|| format!("downloading asset {name:?}"))?;
                Ok::<_, anyhow::Error>((name.clone(), path, metadata))
            }
        })
        .buffer_unordered(4)
        .try_collect()
        .await?;
    let mut paths = HashMap::new();
    let mut metadata = HashMap::new();
    for (name, path, info) in downloads {
        paths.insert(name.clone(), path);
        metadata.insert(name, info);
    }
    Ok((paths, metadata))
}
async fn download(
    mut grpc: TensorLaneClient<Channel>,
    run_id: &str,
    name: &str,
    destination: PathBuf,
) -> anyhow::Result<(PathBuf, String)> {
    let mut responses = grpc
        .asset(AssetRequest {
            run_id: run_id.to_owned(),
            name: name.to_owned(),
        })
        .await?
        .into_inner();
    let metadata = match responses.message().await?.and_then(|message| message.payload) {
        Some(Payload::Metadata(metadata)) => serde_json::json!({"asset_id":metadata.asset_id,"entrypoint":metadata.entrypoint,
            "metadata":serde_json::from_str::<serde_json::Value>(&metadata.metadata_json)?,"kind":metadata.kind,"asset_type":metadata.asset_type}).to_string(),
        _ => anyhow::bail!("asset stream must start with metadata"),
    };
    fs::create_dir_all(&destination).await?;
    let partial = destination.join("download.part");
    let path = destination.join("data");
    let mut file = fs::File::create(&partial).await?;
    while let Some(message) = responses.message().await? {
        match message.payload {
            Some(Payload::Chunk(chunk)) => file.write_all(&chunk).await?,
            _ => anyhow::bail!("expected an asset chunk"),
        }
    }
    file.flush().await?;
    drop(file);
    fs::rename(partial, &path).await?;
    Ok((fs::canonicalize(path).await?, metadata))
}
