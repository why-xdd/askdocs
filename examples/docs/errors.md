# Error codes

## Payment errors

- `PAY_1001` — card declined by the issuer. Not retryable. Show the customer the
  issuer's message verbatim; paraphrasing it causes support tickets.
- `PAY_1002` — insufficient funds. Not retryable within the same authorisation.
- `PAY_1004` — the idempotency key was reused with a different request body.
  This is a client bug, not a transient failure.
- `PAY_1009` — the acquirer timed out. Retryable with exponential backoff, but
  only with the same idempotency key.

## Infrastructure errors

- `INF_2001` — the database connection pool is exhausted. Usually a slow query
  holding connections rather than genuine load.
- `INF_2003` — Redis unreachable. The service degrades to in-memory idempotency,
  which is correct per-instance but not across instances.
