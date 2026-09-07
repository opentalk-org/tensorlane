{
  dnvrState,
  inputs,
  pkgs,
  presets,
  system,
  ...
}:
let
  atlas = inputs.dialo-overlays.packages.${system}.atlas;
in
{
  processes.clickhouse = {
    imports = [ presets.clickhouse ];
    database = "default";
    logLevel = "warning";
  };

  processes.clickhouse-atlas-dev = {
    imports = [ presets.clickhouse ];
    database = "default";
    dataDir = ".dnvr/runtime/clickhouse-atlas-dev/data";
    httpPort = null;
    tcpPort = null;
    logLevel = "warning";
  };

  processes.clickhouse-atlas-watch = {
    env = {
      LOCAL_CLICKHOUSE_HOST = "dnvr://clickhouse/host";
      LOCAL_CLICKHOUSE_PORT = "dnvr://clickhouse/tcpPort";
      LOCAL_CLICKHOUSE_DATABASE = "dnvr://clickhouse/database";
      ATLAS_CLICKHOUSE_HOST = "dnvr://clickhouse-atlas-dev/host";
      ATLAS_CLICKHOUSE_PORT = "dnvr://clickhouse-atlas-dev/tcpPort";
      ATLAS_CLICKHOUSE_DATABASE = "dnvr://clickhouse-atlas-dev/database";
    };
    command = pkgs.writeShellApplication {
      name = "clickhouse-atlas-watch";
      runtimeInputs = [
        atlas
        pkgs.watchexec
      ];
      text = ''
        cd "$DNVR_ROOT"
        export CLICKHOUSE_URL="clickhouse://$LOCAL_CLICKHOUSE_HOST:$LOCAL_CLICKHOUSE_PORT/$LOCAL_CLICKHOUSE_DATABASE"
        export CLICKHOUSE_DEV_URL="clickhouse://$ATLAS_CLICKHOUSE_HOST:$ATLAS_CLICKHOUSE_PORT/$ATLAS_CLICKHOUSE_DATABASE"

        echo "[clickhouse-atlas] applying declared schema"
        atlas schema apply --env local --auto-approve
        echo "[clickhouse-atlas] watching db/schema"
        exec watchexec --postpone -e hcl,sql -w db/schema -- \
          atlas schema apply --env local --auto-approve
      '';
    };
  };

  scripts.atlas-migrate-diff = {
    description = "Generate a ClickHouse SQL migration from the declared HCL schema";
    runtimeInputs = [
      atlas
      dnvrState
    ];
    text = ''
      set -euo pipefail
      cd "$DNVR_ROOT"

      migration_name="''${1:-}"
      if [ -z "$migration_name" ]; then
        read -r -p "Migration name: " migration_name
      fi
      if [ -z "$migration_name" ]; then
        echo "Migration name is required" >&2
        exit 2
      fi

      dev_host="$(dnvr-state get clickhouse-atlas-dev.host)"
      dev_port="$(dnvr-state get clickhouse-atlas-dev.tcpPort)"
      dev_database="$(dnvr-state get clickhouse-atlas-dev.database)"
      export CLICKHOUSE_DEV_URL="clickhouse://$dev_host:$dev_port/$dev_database"

      exec atlas migrate diff "$migration_name" --env migration
    '';
  };

  scripts.atlas-migrate-prod = {
    description = "Apply versioned ClickHouse migrations to production";
    runtimeInputs = [ atlas ];
    text = ''
      set -euo pipefail
      cd "$DNVR_ROOT"

      read -r -s -p "Production ClickHouse URL: " CLICKHOUSE_URL
      printf '\n'
      if [ -z "$CLICKHOUSE_URL" ]; then
        echo "Production ClickHouse URL is required" >&2
        exit 2
      fi

      read -r -p "Apply pending production migrations? [y/N] " confirmation
      if [ "$confirmation" != "y" ] && [ "$confirmation" != "Y" ]; then
        echo "Cancelled"
        exit 0
      fi

      export CLICKHOUSE_URL
      exec atlas migrate apply --env prod "$@"
    '';
  };
}
