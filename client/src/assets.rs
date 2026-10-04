use crate::proto::{AssetRequest, InitResponse, asset_response::Payload};
use anyhow::Context;
use futures_util::{StreamExt, TryStreamExt, stream};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};
use tokio::{fs, io::AsyncWriteExt};

pub async fn prefetch(
    grpc: &crate::transport::GrpcClient,
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
    mut grpc: crate::transport::GrpcClient,
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
    let mut entrypoint = None;
    let metadata = match responses
        .message()
        .await?
        .and_then(|message| message.payload)
    {
        Some(Payload::Metadata(metadata)) => {
            entrypoint.clone_from(&metadata.entrypoint);
            serde_json::json!({"asset_id":metadata.asset_id,"entrypoint":metadata.entrypoint,
            "metadata":serde_json::from_str::<serde_json::Value>(&metadata.metadata_json)?,"kind":metadata.kind,"asset_type":metadata.asset_type}).to_string()
        }
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
    let path = tokio::task::spawn_blocking(move || materialize(&path, entrypoint.as_deref()))
        .await
        .context("asset extraction task failed")??;
    Ok((fs::canonicalize(path).await?, metadata))
}

fn materialize(path: &Path, entrypoint: Option<&str>) -> anyhow::Result<PathBuf> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = std::fs::File::open(path)?;
    let mut block = [0u8; 512];
    let count = file.read(&mut block)?;
    let header = tar::Header::from_byte_slice(&block);
    let checksum = block[..148]
        .iter()
        .chain(&block[156..])
        .map(|byte| u32::from(*byte))
        .sum::<u32>()
        + 8 * 32;
    let is_tar = count == 512
        && (header.cksum().ok() == Some(checksum)
            || (block == [0; 512] && file.metadata()?.len() >= 1024));
    if !is_tar {
        return Ok(path.to_owned());
    }
    file.seek(SeekFrom::Start(0))?;
    let extracted = path.with_file_name("extracted");
    std::fs::create_dir_all(&extracted)?;
    for entry in tar::Archive::new(file).entries()? {
        let mut entry = entry?;
        anyhow::ensure!(
            entry.header().entry_type().is_file() || entry.header().entry_type().is_dir(),
            "asset archive entries must be regular files or directories"
        );
        anyhow::ensure!(
            entry.unpack_in(&extracted)?,
            "asset archive path escapes extraction directory"
        );
    }
    let result = if let Some(entrypoint) = entrypoint {
        let selected = extracted
            .join(entrypoint)
            .canonicalize()
            .context("asset entrypoint does not exist")?;
        anyhow::ensure!(
            selected.starts_with(extracted.canonicalize()?),
            "asset entrypoint escapes extraction directory"
        );
        selected
    } else {
        let children = std::fs::read_dir(&extracted)?.collect::<Result<Vec<_>, _>>()?;
        if children.len() == 1 {
            children[0].path()
        } else {
            extracted
        }
    };
    std::fs::remove_file(path)?;
    Ok(result)
}
