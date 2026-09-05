use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, ensure};
use tokio::io::AsyncWriteExt;

use crate::proto::tensor_lane_client::TensorLaneClient;
use crate::proto::{AssetRequest, asset_response};

pub async fn download(
    mut client: TensorLaneClient<tonic::transport::Channel>,
    run_id: String,
    name: String,
    destination: PathBuf,
) -> anyhow::Result<PathBuf> {
    let asset_dir = destination.join(&name);
    let marker = destination.join(format!("{name}.done"));
    if tokio::fs::try_exists(&marker).await? {
        let relative = tokio::fs::read_to_string(&marker).await?;
        return resolve(&asset_dir, relative.trim());
    }

    tokio::fs::create_dir_all(&destination).await?;
    let part = destination.join(format!("{name}.part"));
    let mut stream = client
        .asset(AssetRequest {
            run_id,
            name: name.clone(),
        })
        .await?
        .into_inner();
    let metadata = match stream.message().await?.and_then(|message| message.payload) {
        Some(asset_response::Payload::Metadata(metadata)) => metadata,
        _ => anyhow::bail!("asset {name:?} stream did not start with metadata"),
    };
    let mut file = tokio::fs::File::create(&part).await?;
    while let Some(message) = stream.message().await? {
        match message.payload {
            Some(asset_response::Payload::Chunk(chunk)) => file.write_all(&chunk).await?,
            _ => anyhow::bail!("asset {name:?} stream contained repeated metadata"),
        }
    }
    file.sync_all().await?;
    drop(file);

    let part_for_extract = part.clone();
    let asset_dir_for_extract = asset_dir.clone();
    tokio::task::spawn_blocking(move || extract(&part_for_extract, &asset_dir_for_extract))
        .await
        .context("joining asset extractor")??;
    let relative = metadata.entrypoint.as_deref().unwrap_or(".");
    let resolved = resolve(&asset_dir, relative)?;
    ensure!(
        resolved.exists(),
        "asset {name:?} entrypoint {relative:?} does not exist"
    );
    tokio::fs::write(marker, relative).await?;
    Ok(resolved)
}

fn extract(part: &Path, asset_dir: &Path) -> anyhow::Result<()> {
    if asset_dir.exists() {
        std::fs::remove_dir_all(asset_dir)?;
    }
    std::fs::create_dir_all(asset_dir)?;
    if is_tar(part)? {
        tar::Archive::new(File::open(part)?).unpack(asset_dir)?;
        std::fs::remove_file(part)?;
    } else {
        let name = asset_dir
            .file_name()
            .context("asset directory has no filename")?;
        std::fs::rename(part, asset_dir.join(name))?;
    }
    Ok(())
}

fn is_tar(path: &Path) -> anyhow::Result<bool> {
    let mut file = File::open(path)?;
    let mut header = [0_u8; 512];
    let read = file.read(&mut header)?;
    Ok(read == header.len() && &header[257..262] == b"ustar")
}

fn resolve(asset_dir: &Path, relative: &str) -> anyhow::Result<PathBuf> {
    ensure!(!relative.is_empty(), "asset marker is empty");
    Ok(if relative == "." {
        asset_dir.to_path_buf()
    } else {
        asset_dir.join(relative)
    })
}
