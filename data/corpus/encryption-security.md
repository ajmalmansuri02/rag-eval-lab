# Encryption and Security

## Encryption at rest and in transit

All vault data is encrypted at rest with AES-256-GCM using a separate key for each vault.
Connections between clients and the server use TLS 1.3, with TLS 1.2 as the minimum accepted
version.

## End-to-end encrypted vaults

A vault can optionally be created as **end-to-end encrypted (E2E)**. In an E2E vault, files are
encrypted on the client and the server never sees the keys (zero-knowledge). Keys are derived from
the user's passphrase with Argon2id.

Trade-offs of E2E vaults:

- The web UI cannot preview files or search file contents in an E2E vault.
- When the vault is created, Driftbox shows a 24-word recovery key. If a user loses both the
  passphrase and the recovery key, the data in the vault is unrecoverable. Quillfeather Labs
  cannot reset it.

## Reporting vulnerabilities

Report security issues to security@quillfeather.example. Quillfeather follows a 90-day
coordinated disclosure policy and runs a bug bounty with rewards up to $5,000.
