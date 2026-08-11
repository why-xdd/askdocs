# Deployment

## Environment variables

The service reads configuration from the environment at boot. Nothing is read
from disk except the TLS bundle.

| variable | default | notes |
|---|---|---|
| `PAYMENTS_DB_URL` | none | Required. Postgres DSN. |
| `PAYMENTS_REDIS_URL` | `redis://localhost:6379/0` | Used for idempotency keys. |
| `PAYMENTS_LOG_LEVEL` | `info` | One of `debug`, `info`, `warn`, `error`. |
| `PAYMENTS_SHUTDOWN_GRACE` | `30s` | How long to drain before exiting. |

## Rolling out a new version

Deployments are blue-green. The new colour is brought up alongside the old one,
health-checked, then the load balancer is flipped in a single operation. The old
colour stays running for fifteen minutes so a rollback is a second flip rather
than a redeploy.

Never scale to zero during a flip. The connection pool warms lazily, and an
empty pool on a cold instance produces a latency spike large enough to trip the
downstream circuit breaker.

## Draining

On `SIGTERM` the service stops accepting new connections, finishes in-flight
requests, and exits once the last one completes or the grace period expires.
