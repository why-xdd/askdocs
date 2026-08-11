# Security practices

## Secrets

Secrets live in the secret manager and are injected as environment variables at
boot. Nothing sensitive belongs in the repository, in a container image, or in a
log line.

## Logging

Card numbers, CVVs and full names are never logged. Log the payment id and look
it up if you need more. A log aggregator is a database with weaker access
controls than your actual database, and should be treated that way.

## Access

Production access is granted for a fixed window and expires automatically.
Nobody holds standing production credentials, including the people who built the
system.

## Dependencies

Dependencies are pinned and updated on a schedule rather than opportunistically.
An urgent security patch is the exception, not the pattern.
