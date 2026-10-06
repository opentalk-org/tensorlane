# ClickHouse schema management

The desired ClickHouse schema is declared in `schema/*.ch.hcl`. Atlas-generated
SQL migrations and the migration directory checksum belong in `migrations/`.
Keep cloud-inspected engines in their OSS form here: use `Atomic` instead of
`Shared`, and the corresponding `MergeTree` engine instead of `SharedMergeTree`.
ClickHouse Cloud converts these engines when migrations are applied.

Run the persistent local ClickHouse target, disposable Atlas planning server,
and schema watcher with the rest of the development stack:

```sh
nix develop -c dnvr up
```

The `clickhouse-atlas-watch` process applies the schema once both databases are
ready and reapplies it whenever an HCL or SQL schema file changes.

Generate a versioned SQL migration after editing the HCL schema:

```sh
nix develop -c atlas-migrate-diff migration_name
```

This command uses the running `clickhouse-atlas-dev` process. If the migration
name argument is omitted, it prompts for one.

Apply committed migrations to production:

```sh
nix develop -c atlas-migrate-prod
```

The production command prompts for the database URL without echoing it, asks
for confirmation, and then applies the committed migration directory. Atlas
flags such as `--dry-run` or `--baseline <version>` can be appended.

Run configurations use the single `runs.config` string column. Apply the ordered
`20261002120000`, `20261002120100`, and `20261002120200` migrations before starting
the updated server. Pause run creation while backfilling and retiring the legacy
columns. The backfill retains both legacy JSON sections in a lossless snapshot;
it does not make historical multi-stage runs executable. The retirement migration
checks completion before dropping either old column.

Application dataset tables are excluded from active schema management. Historical
migrations remain intact and no dataset contents are dropped. Automatic schema
application skips table and column drops; the configuration retirement must run
through the committed, ordered migrations.

Historical migration files still describe the original application dataset tables.
Their checksums and compatibility exclusions are preserved for existing installs;
the current declared schema and runtime do not depend on those tables.

Client session ownership and heartbeat tracking have been removed. The server
uses the existing `runs` and `run_status` tables and does not need `run_sessions`.
The Atlas-generated `20261006115544_remove_run_sessions` migration drops that
unused table. Historical migrations and their checksums remain unchanged.
The new server can run before this cleanup migration is applied.
