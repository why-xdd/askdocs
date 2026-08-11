# Architecture overview

## Shape of the system

Requests enter through the API gateway, which handles authentication and rate
limiting. Beyond it sit three services: the payments API, the ledger, and an
asynchronous worker fleet.

The payments API is stateless and horizontally scaled. The ledger is the only
component that writes to the primary database, which is what keeps
double-entry invariants enforceable in one place.

## Why the ledger is separate

Every money movement is two entries that must both land or neither. Keeping that
in a dedicated service means the invariant is enforced by one codebase with one
set of tests, rather than by convention across every service that touches money.

## Asynchronous work

Anything that can be done after the customer has their response is done by the
worker fleet: emails, webhooks, reconciliation, reporting. The queue is Redis
Streams with a dead-letter stream for messages that fail five times.
