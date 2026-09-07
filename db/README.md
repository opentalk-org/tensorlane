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
