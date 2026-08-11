# On-call runbook

## What being on-call means

You are the first responder for alerts, not the person who must fix everything.
Escalating early is correct behaviour, not a failure.

## Common alerts

### `HighP99Latency`

Check the database first. In roughly nine cases out of ten this is a slow query
holding connections, not traffic. Look at `pg_stat_statements` ordered by total
time before anything else.

### `QueueBackingUp`

The worker fleet is not keeping up. Scale workers first, then find out why.
Diagnosing a backlog while it grows is harder than diagnosing a stable one.

### `PaymentFailureRate`

Compare failure rates per acquirer before assuming it is us. A single acquirer
having a bad afternoon looks identical to a deploy regression on the top-line
graph.

## Escalation

If you have not made progress in twenty minutes, page the secondary. There is no
prize for struggling alone at three in the morning.
