# Northwind Analytics — Integrations and SDKs

## Ingestion SDKs
Northwind provides server-side SDKs for the following languages:

- Python
- Node.js
- Go
- Ruby

There is also a browser JavaScript snippet for client-side event tracking. There
is no native mobile SDK; mobile apps are expected to send events through their
own backend using one of the server-side SDKs.

## Data warehouse export
On the Pro and Enterprise plans, raw events can be exported on a daily schedule
to one of the supported warehouses:

- Snowflake
- Google BigQuery
- Amazon Redshift

Export runs once per day at 02:00 UTC and cannot currently be scheduled more
frequently.

## Outbound integrations
Northwind can forward computed alerts to Slack and to a generic webhook. The
Slack integration is available on all paid plans. PagerDuty is not supported.
