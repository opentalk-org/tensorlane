{
  description = "TensorLane";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-26.05";
    flake-parts.url = "github:hercules-ci/flake-parts";
    dnvr.url = "github:dialohq/dnvr";
    dnvr.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-parts,
      dnvr,
    }@inputs:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [ dnvr.flakeModule ];
      systems = [
        "aarch64-darwin"
        "aarch64-linux"
        "x86_64-darwin"
        "x86_64-linux"
      ];

      perSystem =
        {
          self',
          pkgs,
          ...
        }:
        {
          formatter = pkgs.nixfmt;

          packages = {
            default = self'.packages.server;
            server =
              let
                manifest = (pkgs.lib.importTOML ./server/Cargo.toml).package;
              in
              pkgs.rustPlatform.buildRustPackage {
                pname = manifest.name;
                version = manifest.version;
                cargoLock = {
                  lockFile = ./Cargo.lock;
                };
                src = ./.;
                buildAndTestSubdir = "server";
              };
          };

          dnvr.shells.default = { ... }: {
            packages = [
              pkgs.python312
              pkgs.uv
              pkgs.ruff
              pkgs.pyright

              pkgs.cargo
              pkgs.clippy
              pkgs.rustc
              pkgs.rust-analyzer
              pkgs.rustfmt

              pkgs.protobuf
            ];

            env = {
              CLICKHOUSE_URL = "http://127.0.0.1:8123";
              CLICKHOUSE_USER = "default";
              CLICKHOUSE_PASSWORD = "";
              GRPC_PORT = "8181";
              HTTP_PORT = "8180";
              SYNTHETIC = "true";

              AWS_ENDPOINT_URL = "http://127.0.0.1:9001";
              AWS_ACCESS_KEY_ID = "tensorlane";
              AWS_SECRET_ACCESS_KEY = "tensorlane";
              S3_BUCKET = "tensorlane";

              CACHE_DIR = "$DNVR_ROOT/.tensorlane/cache";
              ASSETS_DIR = "$DNVR_ROOT/.tensorlane/assets";
              CHECKPOINT_DIR = "$DNVR_ROOT/.tensorlane/checkpoints";
              METRICS_DIR = "$DNVR_ROOT/.tensorlane/artifacts";

              UV_PYTHON_PREFERENCE = "only-system";
              UV_PYTHON_DOWNLOADS = "never";

              LIBTORCH_USE_PYTORCH = "1";
            };

            shellHook = ''
              if [ -f .env ]; then
                set -a
                . ./.env
                set +a
              fi
              unset NIX_CFLAGS_COMPILE CFLAGS CXXFLAGS
              if [ -e .venv/bin/activate ]; then
                . .venv/bin/activate
              fi
            '';

            scripts.server = {
              description = "The main server.";
              runtimeInputs = [ pkgs.cargo ];
              text = ''
                cargo run --bin tensorlane
              '';
            };
          };
        };
    };
}
