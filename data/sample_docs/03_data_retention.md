# Northwind Analytics — Data Retention

Northwind Analytics distinguishes between raw event data and aggregated reports.

## Raw event data
- Starter: raw events are retained for 30 days, then permanently deleted.
- Pro: raw events are retained for 12 months.
- Enterprise: raw event retention is configurable from 12 to 60 months.

## Aggregated reports
Aggregated reports (funnels, retention curves, cohort tables) are retained for
the lifetime of the account on all plans, even after the underlying raw events
have been deleted.

## Deletion requests
Customers can request deletion of an individual end user's data through the
Privacy API. Northwind processes these deletion requests within 30 days. Deleted
end-user data is removed from both raw storage and any aggregated reports that
can still be recomputed.

## Backups
Encrypted backups of raw event data are kept for an additional 14 days beyond the
plan retention window, after which they are rotated out and destroyed.
