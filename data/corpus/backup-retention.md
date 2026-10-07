# Snapshots, Retention and Restore

Every vault is protected by automatic snapshots. A snapshot is a point-in-time, read-only view of
the whole vault.

## Default retention policy ("standard")

| Snapshot type | Kept for |
|---|---|
| Hourly | 48 hours |
| Daily | 30 days |
| Weekly | 12 weeks |
| Monthly | 12 months |

Administrators can define custom policies per vault.

## Trash

Deleted files go to the vault trash before they are purged. Files stay in the trash for 30 days
on Team and Enterprise, and for 7 days on Community.

## Restoring

Restore individual files or whole folders from the web UI, or run
`driftctl restore --vault <name> --snapshot <id>` on the server.

## Immutable snapshots

On Enterprise, snapshots replicated to S3-compatible storage can use object lock so that they
cannot be deleted or modified before their retention period ends, even by an administrator.
