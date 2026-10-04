use anyhow::Result;
use std::{
    net::TcpListener,
    path::Path,
    process::{Child, Command, Stdio},
};
use testcontainers_modules::{
    clickhouse::ClickHouse,
    minio::MinIO,
    testcontainers::{
        ContainerAsync, ImageExt,
        core::{IntoContainerPort, WaitFor, wait::HttpWaitStrategy},
        runners::AsyncRunner,
    },
};
pub fn port() -> Result<u16> {
    Ok(TcpListener::bind("127.0.0.1:0")?.local_addr()?.port())
}
#[derive(Default)]
pub struct Services {
    pub children: Vec<Child>,
    clickhouse: Option<ContainerAsync<ClickHouse>>,
    s3: Option<ContainerAsync<MinIO>>,
}
impl Drop for Services {
    fn drop(&mut self) {
        for child in &mut self.children {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}
impl Services {
    pub async fn start(root: &Path) -> Result<(Self, String, String)> {
        let mut services = Self::default();
        let urls = if std::env::var_os("TENSORLANE_TEST_LOCAL").is_some() {
            let http_port = port()?;
            let tcp_port = port()?;
            let s3_port = port()?;
            let config_path = root.join("clickhouse.xml");
            std::fs::write(
                &config_path,
                format!(
                    "<clickhouse><logger><level>error</level><log>{root}/clickhouse.log</log><errorlog>{root}/clickhouse-error.log</errorlog></logger><path>{root}/ch/</path><tmp_path>{root}/tmp/</tmp_path><listen_host>127.0.0.1</listen_host><http_port>{http_port}</http_port><tcp_port>{tcp_port}</tcp_port><background_pool_size>16</background_pool_size><background_schedule_pool_size>4</background_schedule_pool_size><background_message_broker_schedule_pool_size>2</background_message_broker_schedule_pool_size><background_distributed_schedule_pool_size>2</background_distributed_schedule_pool_size><profiles><default><max_threads>4</max_threads><max_memory_usage>1000000000</max_memory_usage></default></profiles><users><default><password>test</password><networks><ip>127.0.0.1</ip></networks><profile>default</profile><quota>default</quota><access_management>1</access_management></default></users><quotas><default/></quotas></clickhouse>",
                    root = root.display()
                ),
            )?;
            let clickhouse =
                std::env::var("TENSORLANE_TEST_CLICKHOUSE").unwrap_or_else(|_| "clickhouse".into());
            services.children.push(
                Command::new(clickhouse)
                    .args(["server", "--config-file"])
                    .arg(&config_path)
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .spawn()?,
            );
            let minio = std::env::var("TENSORLANE_TEST_MINIO").unwrap_or_else(|_| "minio".into());
            services.children.push(
                Command::new(minio)
                    .arg("server")
                    .arg(root.join("minio"))
                    .args([
                        "--address",
                        &format!("127.0.0.1:{s3_port}"),
                        "--console-address",
                        "127.0.0.1:0",
                    ])
                    .env("MINIO_ROOT_USER", "minioadmin")
                    .env("MINIO_ROOT_PASSWORD", "minioadmin")
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .spawn()?,
            );
            (
                format!("http://127.0.0.1:{http_port}"),
                format!("http://127.0.0.1:{s3_port}"),
            )
        } else {
            let ch = ClickHouse::default()
                .with_tag("26.7.4.58")
                .with_env_var("CLICKHOUSE_USER", "default")
                .with_env_var("CLICKHOUSE_PASSWORD", "test")
                .with_ready_conditions(vec![WaitFor::http(
                    HttpWaitStrategy::new("/ping")
                        .with_port(8123.tcp())
                        .with_expected_status_code(200_u16),
                )])
                .start()
                .await?;
            let s3 = MinIO::default()
                .with_name("quay.io/minio/minio")
                .with_env_var("MINIO_ROOT_USER", "minioadmin")
                .with_env_var("MINIO_ROOT_PASSWORD", "minioadmin")
                .start()
                .await?;
            let urls = (
                format!(
                    "http://{}:{}",
                    ch.get_host().await?,
                    ch.get_host_port_ipv4(8123).await?
                ),
                format!(
                    "http://{}:{}",
                    s3.get_host().await?,
                    s3.get_host_port_ipv4(9000).await?
                ),
            );
            services.clickhouse = Some(ch);
            services.s3 = Some(s3);
            urls
        };
        Ok((services, urls.0, urls.1))
    }
}
