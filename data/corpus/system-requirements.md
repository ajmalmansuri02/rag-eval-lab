# System Requirements

## Hardware

- **CPU:** at least 2 cores.
- **Memory:** at least 2 GB of RAM. We recommend 4 GB or more for deployments with more than
  25 users.
- **Architecture:** x86-64 and ARM64. ARM64 builds have been available since version 3.0.

## Operating systems

The server is supported on Debian 12, Ubuntu 22.04, Ubuntu 24.04 and RHEL 9. Windows is not a
supported server platform.

## Storage

Vault storage must live on a local filesystem. ext4, XFS and ZFS are fully supported. Btrfs is
supported experimentally. **NFS and SMB network shares are not supported** for vault storage,
because file locking over network filesystems is unreliable.

## Metadata database

Driftbox keeps metadata (users, shares, file index) in a database:

- An embedded SQLite database is used by default and is suitable for up to 10 users.
- PostgreSQL 14 or newer is required above 10 users and for all Enterprise deployments.
