# Driftbox Overview

Driftbox is a self-hosted file sync and backup platform built by Quillfeather Labs. It was first
released in 2021 and the current major version is 3.2. Teams run the Driftbox Server on their own
Linux hardware, and people connect to it from desktop and mobile clients.

## Components

- **Driftbox Server** runs as a single daemon called `driftd`. It stores files in *vaults*, which
  are isolated storage areas with their own permissions, retention policy and encryption keys.
- **Driftbox Desktop** clients are available for Windows, macOS and Linux.
- **Driftbox Mobile** apps are available for iOS and Android.
- **driftctl** is the command-line tool administrators use to manage a server.

## Network ports

The web UI and REST API listen on port 8470 by default. The sync protocol used by desktop and
mobile clients uses port 8471. Both ports can be changed in `/etc/driftbox/driftd.toml`.

## Hosted option

Organizations that do not want to run their own server can use **Driftbox Cloud**, a hosted
version operated by Quillfeather Labs. Driftbox Cloud is available in two regions: Frankfurt
(eu-central) and Oregon (us-west).
