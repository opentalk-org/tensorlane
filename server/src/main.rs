use std::{io, path::PathBuf};

use anyhow::Context;
use aws_config::BehaviorVersion;
use aws_sdk_s3::config::Credentials;
use clap::{
    CommandFactory, FromArgMatches, Parser, Subcommand,
    builder::{
        Styles,
        styling::{AnsiColor, Effects},
    },
};
use clap_complete::{Shell, generate};
use tokio::fs;
use tokio_util::sync::CancellationToken;
use tracing::{error, info};

mod audio;
mod db;
mod grpc;
mod grpc_support;
mod http;
mod loader;
mod metrics;
mod prefetch;
mod run;
mod run_manager;
mod sampling;
mod symbols;

mod proto {
    tonic::include_proto!("_");
}

const STYLES: Styles = Styles::styled()
    .header(AnsiColor::Green.on_default().effects(Effects::BOLD))
    .usage(AnsiColor::Green.on_default().effects(Effects::BOLD))
    .literal(AnsiColor::Cyan.on_default().effects(Effects::BOLD))
    .placeholder(AnsiColor::Cyan.on_default())
    .error(AnsiColor::Red.on_default().effects(Effects::BOLD))
    .valid(AnsiColor::Cyan.on_default().effects(Effects::BOLD))
    .invalid(AnsiColor::Yellow.on_default().effects(Effects::BOLD));

#[derive(Parser)]
#[command(
    name = "givemedata",
    styles = STYLES,
    args_conflicts_with_subcommands = true,
    subcommand_negates_reqs = true
)]
#[command(about = "GIVE ME DATA!", long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Option<Command>,
    #[arg(
        short,
        long,
        env = "GRPC_PORT",
        help = "gRPC server binding port.",
        default_value = "8181"
    )]
    grpc_port: u16,
    #[arg(
        long,
        env = "CLICKHOUSE_URL",
        help = "ClickHouse HTTP endpoint for runs and training data metadata."
    )]
    clickhouse_url: String,
    #[arg(
        long,
        env = "CLICKHOUSE_USER",
        help = "ClickHouse user for runs and training data metadata."
    )]
    clickhouse_user: String,
    #[arg(
        long,
        env = "CLICKHOUSE_PASSWORD",
        help = "ClickHouse password for runs and training data metadata."
    )]
    clickhouse_password: String,
    #[arg(
        long,
        env = "HTTP_PORT",
        help = "HTTP server binding port.",
        default_value = "8180"
    )]
    http_port: u16,
    #[arg(long, env = "AWS_ENDPOINT_URL", help = "Endpoint to S3 bucket.")]
    s3_endpoint: String,
    #[arg(long, env = "AWS_ACCESS_KEY_ID", help = "S3 access key ID.")]
    s3_key: String,
    #[arg(long, env = "AWS_SECRET_ACCESS_KEY", help = "S3 secret key.")]
    s3_secret: String,
    #[arg(long, env = "S3_BUCKET", help = "S3 bucket name.")]
    bucket: String,
    #[arg(
        long,
        env = "CACHE_DIR",
        help = "Directory for spilling prefetched audio to disk; omit to keep it in memory."
    )]
    cache_dir: PathBuf,
    #[arg(
        long,
        env = "SYNTHETIC",
        help = "Serve synthetic runs: fabricated samples instead of database rows and bucket audio."
    )]
    synthetic: bool,
    #[arg(
        long,
        env = "ASSETS_DIR",
        help = "Directory caching training assets downloaded from the bucket."
    )]
    assets_dir: PathBuf,
    #[arg(
        long,
        env = "CHECKPOINT_DIR",
        help = "Directory storing checkpoints uploaded by trainings."
    )]
    checkpoint_dir: PathBuf,
    #[arg(
        long,
        env = "METRICS_DIR",
        help = "Directory storing streamed metric artifacts."
    )]
    metrics_dir: PathBuf,
}

#[derive(Subcommand)]
enum Command {
    /// Generate shell completion definitions.
    Completions { shell: Shell },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("debug,h2=off,hyper=off,tower=off,aws_runtime=off,aws_sdk_s3=off,aws_smithy_runtime_api=off,aws_smithy_runtime=off")
        .init();

    let matches = Args::command().get_matches();
    if let Some(("completions", completion_matches)) = matches.subcommand() {
        let shell = completion_matches
            .get_one::<Shell>("shell")
            .copied()
            .context("shell is required by clap")?;
        generate(shell, &mut Args::command(), "givemedata", &mut io::stdout());
        return Ok(());
    }
    let args = Args::from_arg_matches(&matches)?;

    let database = clickhouse::Client::default()
        .with_url(&args.clickhouse_url)
        .with_user(&args.clickhouse_user)
        .with_password(&args.clickhouse_password)
        .with_setting("allow_experimental_json_type", "1")
        .with_setting("input_format_binary_read_json_as_string", "1")
        .with_setting("output_format_binary_write_json_as_string", "1");

    let s3_config = aws_config::defaults(BehaviorVersion::latest())
        .endpoint_url(&args.s3_endpoint)
        .credentials_provider(Credentials::new(
            &args.s3_key,
            &args.s3_secret,
            None,
            None,
            "r2",
        ))
        .load()
        .await;
    let s3_client = aws_sdk_s3::Client::from_conf(
        aws_sdk_s3::config::Builder::from(&s3_config)
            .force_path_style(true)
            .build(),
    );

    fs::create_dir_all(&args.cache_dir).await?;
    fs::create_dir_all(&args.assets_dir).await?;
    fs::create_dir_all(&args.checkpoint_dir).await?;
    fs::create_dir_all(&args.metrics_dir).await?;

    let run_manager = run_manager::RunManager::new(database.clone());
    let shutdown = CancellationToken::new();
    tokio::spawn(watch_shutdown_signals(shutdown.clone()));
    let mut http_server = tokio::spawn(http::serve(
        args.http_port,
        run_manager.clone(),
        shutdown.clone(),
    ));
    let mut grpc_server = tokio::spawn(grpc::serve(
        args.grpc_port,
        s3_client,
        database,
        run_manager,
        args.bucket.leak(),
        Box::leak(args.cache_dir.into_boxed_path()),
        Box::leak(args.assets_dir.into_boxed_path()),
        Box::leak(args.checkpoint_dir.into_boxed_path()),
        Box::leak(args.metrics_dir.into_boxed_path()),
        args.synthetic,
        shutdown.clone(),
    ));
    tokio::select! {
        result = &mut http_server => {
            shutdown.cancel();
            result??;
            grpc_server.await??;
        }
        result = &mut grpc_server => {
            shutdown.cancel();
            result??;
            http_server.await??;
        }
    }
    Ok(())
}

async fn watch_shutdown_signals(shutdown: CancellationToken) {
    if let Err(err) = receive_shutdown_signals(shutdown).await {
        error!(error = format!("{err:#}"), "shutdown signal handler failed");
    }
}

async fn receive_shutdown_signals(shutdown: CancellationToken) -> anyhow::Result<()> {
    use tokio::signal::unix::{SignalKind, signal};

    let mut interrupt = signal(SignalKind::interrupt())?;
    let mut terminate = signal(SignalKind::terminate())?;
    for received in 1.. {
        tokio::select! {
            _ = interrupt.recv() => {}
            _ = terminate.recv() => {}
        }
        handle_shutdown_signal(&shutdown, received);
    }
    Ok(())
}

fn handle_shutdown_signal(shutdown: &CancellationToken, received: usize) {
    match received {
        1 => {
            info!("shutdown requested");
            shutdown.cancel();
        }
        2 => info!("shutdown still in progress; one more signal will force exit"),
        _ => {
            info!("received 3 signals, forcing shutdown");
            std::process::exit(130);
        }
    }
}
