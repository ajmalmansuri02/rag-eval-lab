# REST API and Rate Limits

## Base URL and versions

The REST API is served at `/api/v2` on the web port. API v1 is deprecated.

## Authentication

Authenticate with a personal access token in the `Authorization: Bearer` header. Tokens start
with the prefix `dbx_pat_`. By default, tokens expire after 90 days; administrators can shorten
or extend this.

## Rate limits

- Each token can make 600 requests per minute, with bursts of up to 100 requests.
- When the limit is exceeded the API returns HTTP 429 with a `Retry-After` header that says how
  many seconds to wait.

## Uploads

A single upload request can be at most 5 GiB. Larger files must use the multipart upload API,
which accepts chunks between 8 MiB and 64 MiB.
