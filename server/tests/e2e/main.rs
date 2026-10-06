mod cases;
mod performance;
mod recovery;
mod services;
mod setup;
mod stress;
mod training;
mod uploads;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn http_training_survives_retries_and_restarts() -> anyhow::Result<()> {
    let mut env = setup::TestEnv::start().await?;
    assert!(
        env.http
            .get(format!("{}/healthz", env.url))
            .send()
            .await?
            .status()
            .is_success()
    );
    cases::check(&env).await?;
    uploads::check(&mut env).await?;
    recovery::check(&mut env).await?;
    training::check(&env).await
}
