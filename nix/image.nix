{
  pkgs,
  server,
  nix2container,
}:
nix2container.buildImage {
  name = "ghcr.io/opentalk-org/tensorlane";
  copyToRoot = pkgs.buildEnv {
    name = "tensorlane-root";
    paths = [
      server
      pkgs.dockerTools.caCertificates
    ];
    pathsToLink = [
      "/bin"
      "/etc"
    ];
  };
  config = {
    entrypoint = [ "/bin/tensorlane" ];
    user = "10001:10001";
    env = [
      "CACHE_DIR=/cache"
      "CHECKPOINT_PREFIX=tensorlane/dev/checkpoints"
      "METRICS_PREFIX=tensorlane/dev/metrics"
      "RUST_LOG=tensorlane=info"
      "SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt"
    ];
    Labels."org.opencontainers.image.source" = "https://github.com/opentalk-org/tensorlane";
  };
}
