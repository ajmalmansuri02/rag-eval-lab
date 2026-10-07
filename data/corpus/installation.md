# Installing Driftbox Server

There are two supported ways to install the server: the official container image and the Debian
package.

## Container image

The official image is published as `quillfeather/driftbox:3.2`. A minimal run command:

```
docker run -d --name driftbox -p 8470:8470 -p 8471:8471 \
  -v /srv/driftbox:/var/lib/driftbox quillfeather/driftbox:3.2
```

Always mount a volume at `/var/lib/driftbox`; without it, data is lost when the container is
removed.

## Debian package

On Debian and Ubuntu, install the `driftbox-server` package from the Quillfeather apt repository.
The package creates a `driftbox` system user and a systemd unit named `driftd.service`. The
default data directory is `/var/lib/driftbox`.

## First-time setup

After the server starts for the first time it prints a one-time **setup token** to its log. Open
`http://<host>:8470/setup` and paste the token to create the first administrator account. The
setup token is valid for 30 minutes; if it expires, restart the server to generate a new one.

Alternatively, run `driftctl init` on the server to complete setup from the terminal.
