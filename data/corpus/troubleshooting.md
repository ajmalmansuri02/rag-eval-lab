# Troubleshooting

Start with `driftctl doctor`; it detects most configuration and environment problems.

## Error DBX-1042: vault locked

Another process holds the vault lock, usually after a crash. Run `driftctl doctor --fix-locks` to
clear stale locks. Do not delete lock files by hand.

## Sync stuck on "Scanning" (Linux)

On Linux clients with very many files, the client can run out of inotify watches and stay in the
"Scanning" state. Increase the limit, for example to 524288:

```
sudo sysctl fs.inotify.max_user_watches=524288
```

## Login or token errors after the clock changes

Authentication fails if the client or server clock is off by more than 5 minutes. Enable NTP on
both machines.

## Clients cannot connect

Check that port 8471 (sync) and port 8470 (web/API) are reachable through any firewall.
