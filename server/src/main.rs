use std::{
    io,
    path::PathBuf,
    time::{Duration, Instant},
};

use anyhow::Context;
use aws_config::{BehaviorVersion, retry::RetryConfig};
use aws_sdk_s3::config::{Credentials, Region, timeout::TimeoutConfig};
use clap::{
    CommandFactory, FromArgMatches, Parser, Subcommand,
    builder::{
        Styles,
        styling::{AnsiColor, Effects},
    },
};
use clap_complete::{Shell, generate};
use tokio::{
    fs,
    signal::unix::{Signal, SignalKind, signal},
};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

mod asset_commit;
mod asset_http;
mod asset_repo;
mod auth;
mod batch_http;
mod cache_limits;
mod db;
mod health;
mod http;
mod job;
mod loader;
mod metric_http;
mod object_state;
mod query_params;
mod run_config;
mod run_repo;
mod runtime;
mod s3_upload;
mod sampling;
mod shared_cache;
mod upload_http;

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
    name = "tensorlane",
    styles = STYLES,
    args_conflicts_with_subcommands = true,
    subcommand_negates_reqs = true
)]
#[command(about = "TensorLane", long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Option<Command>,
    #[arg(long, env = "TENSORLANE_API_KEY", hide_env_values = true)]
    api_key: Option<String>,
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
    #[arg(
        long,
        env = "AWS_REGION",
        default_value = "auto",
        help = "S3 signing region."
    )]
    s3_region: String,
    #[arg(long, env = "AWS_ACCESS_KEY_ID", help = "S3 access key ID.")]
    s3_key: String,
    #[arg(long, env = "AWS_SECRET_ACCESS_KEY", help = "S3 secret key.")]
    s3_secret: String,
    #[arg(long, env = "S3_BUCKET", help = "S3 bucket name.")]
    bucket: String,
    #[arg(
        long,
        env = "CACHE_DIR",
        help = "Directory for disposable query, batch, and input asset caches."
    )]
    cache_dir: PathBuf,
    #[arg(long, env = "CACHE_BYTES", default_value_t = 50 * 1024 * 1024 * 1024u64)]
    cache_bytes: u64,
    #[arg(
        long,
        env = "CHECKPOINT_PREFIX",
        help = "S3 object-key prefix for saved assets."
    )]
    checkpoint_prefix: String,
    #[arg(
        long,
        env = "METRICS_PREFIX",
        help = "S3 object-key prefix for uploaded metric artifacts."
    )]
    metrics_prefix: String,
}

#[derive(Subcommand)]
enum Command {
    /// Generate shell completion definitions.
    Completions { shell: Shell },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let matches = Args::command().get_matches();
    if let Some(("completions", completion)) = matches.subcommand() {
        let shell = *completion
            .get_one::<Shell>("shell")
            .context("shell is required by clap")?;
        generate(shell, &mut Args::command(), "tensorlane", &mut io::stdout());
        return Ok(());
    }
    let args = Args::from_arg_matches(&matches)?;
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
    let auth = auth::Auth::new(args.api_key.as_deref())?;
    let database = clickhouse::Client::default()
        .with_url(&args.clickhouse_url)
        .with_user(&args.clickhouse_user)
        .with_password(&args.clickhouse_password)
        .with_setting("allow_experimental_json_type", "1")
        .with_setting("input_format_binary_read_json_as_string", "1")
        .with_setting("output_format_binary_write_json_as_string", "1");
    let s3_config = aws_config::defaults(BehaviorVersion::latest())
        .endpoint_url(&args.s3_endpoint)
        .retry_config(RetryConfig::standard().with_max_attempts(5))
        .timeout_config(
            TimeoutConfig::builder()
                .connect_timeout(Duration::from_secs(10))
                .operation_attempt_timeout(Duration::from_secs(600))
                .build(),
        )
        .credentials_provider(Credentials::new(
            &args.s3_key,
            &args.s3_secret,
            None,
            None,
            "r2",
        ))
        .region(Region::new(args.s3_region))
        .load()
        .await;
    let s3_client = aws_sdk_s3::Client::from_conf(
        aws_sdk_s3::config::Builder::from(&s3_config)
            .force_path_style(true)
            .build(),
    );

    fs::create_dir_all(&args.cache_dir).await?;
    let shutdown = CancellationToken::new();
    tokio::spawn(shutdown_on_signals(
        shutdown.clone(),
        signal(SignalKind::interrupt())?,
        signal(SignalKind::terminate())?,
    ));
    let runtime = runtime::Runtime::new(
        database,
        s3_client,
        args.bucket.leak(),
        args.cache_dir,
        args.cache_bytes,
        shutdown.clone(),
        args.checkpoint_prefix.leak(),
        args.metrics_prefix.leak(),
    )?;
    runtime.tasks.spawn(maintain(runtime.clone()));
    let serving = http::serve(args.http_port, auth, runtime.clone());
    tokio::pin!(serving);
    tokio::select! {
        result = &mut serving => result?,
        _ = shutdown.cancelled() => {
            let _ = tokio::time::timeout(Duration::from_secs(30), &mut serving).await;
        }
    }
    runtime.tasks.close();
    let _ = tokio::time::timeout(Duration::from_secs(30), runtime.tasks.wait()).await;
    Ok(())
}

async fn maintain(runtime: runtime::Runtime) {
    let started = Instant::now();
    loop {
        if let Err(error) =
            cache_limits::prune(&runtime.cache, runtime.cache_bytes, &runtime.cache_written).await
        {
            warn!(error = %error, "cache cleanup failed");
        }
        if started.elapsed() >= Duration::from_secs(120)
            && let Err(error) = runtime.repo.fail_silent_runs().await
        {
            warn!(error = %error, "failing silent runs failed");
        }
        tokio::select! {
            _ = runtime.shutdown.cancelled() => break,
            _ = tokio::time::sleep(Duration::from_secs(60)) => {}
        }
    }
}

async fn shutdown_on_signals(
    shutdown: CancellationToken,
    mut interrupt: Signal,
    mut terminate: Signal,
) {
    for received in 1.. {
        tokio::select! {
            _ = interrupt.recv() => {}
            _ = terminate.recv() => {}
        }
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
}
