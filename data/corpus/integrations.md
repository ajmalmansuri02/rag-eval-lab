# Integrations

## Off-site replication

Vaults can replicate snapshots to S3-compatible object storage, such as AWS S3, Backblaze B2 or a
self-hosted MinIO cluster.

## Webhooks

Driftbox can send webhooks for events such as `file.created`, `file.deleted` and `share.created`.
Each request is signed with HMAC-SHA256 and the signature is sent in the `X-Driftbox-Signature`
header. Failed deliveries are retried 5 times with exponential backoff.

## Directory services

- **LDAP / Active Directory:** users and groups are synchronized every 4 hours. Available on Team
  and Enterprise.
- **SAML SSO:** available on Enterprise only. Tested with Okta, Microsoft Entra ID and Keycloak.

## Notifications

A Slack integration can post a message when a share is created or when a vault nears its quota.
