# driftctl Command Reference

`driftctl` is the administrator command-line tool. Run it on the server as root or as the
`driftbox` user.

| Command | What it does |
|---|---|
| `driftctl init` | Complete first-time setup and create the first administrator. |
| `driftctl status` | Show server version, uptime, connected clients and storage usage. |
| `driftctl vault create <name>` | Create a vault. Add `--e2e` for an end-to-end encrypted vault. |
| `driftctl user add <email>` | Invite a user. |
| `driftctl snapshot list --vault <name>` | List snapshots of a vault. |
| `driftctl restore --vault <name> --snapshot <id>` | Restore a vault or path from a snapshot. |
| `driftctl doctor` | Check configuration, ports, disk space and database health. |
| `driftctl backup export` | Export server configuration and metadata for disaster recovery. |

## Global flags

- `--config <path>` – use a different config file (default `/etc/driftbox/driftd.toml`).
- `--json` – print machine-readable output.
- `--verbose` – print debug logs.
