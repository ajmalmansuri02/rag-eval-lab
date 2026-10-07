# How Sync Works

## Block-level delta sync

Driftbox splits files into 4 MiB blocks and only transfers blocks that changed. Editing a few
bytes in a large file therefore uploads one block rather than the whole file.

## Change detection

Desktop clients watch the filesystem for changes in real time. As a safety net, clients also run
a full rescan every 60 minutes. The interval can be changed with the `rescan_interval` setting.

## Conflict handling

Driftbox never silently overwrites changes. When two devices modify the same file before syncing,
the server keeps the version that arrived first and saves the other as a **conflict copy** next
to it, named `filename (conflict from <device> YYYY-MM-DD).ext`. Users resolve the conflict by
keeping the version they want and deleting the copy.

## LAN sync

When two clients are on the same local network, they can exchange blocks directly instead of
going through the server. Peers discover each other with mDNS. LAN sync is enabled by default.

## Selective sync and bandwidth

Users can choose which folders sync to each device (selective sync). Upload and download
bandwidth limits can be set per client.
