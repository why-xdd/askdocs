# API reference

## Creating a payment

`POST /v1/payments`

Every request must carry an `Idempotency-Key` header. Reusing a key with an
identical body returns the original response; reusing it with a different body
returns `PAY_1004`.

```json
{
  "amount": 1250,
  "currency": "GBP",
  "reference": "order-8891"
}
```

Amounts are integers in the currency's minor unit. There are no floats anywhere
in the API, because binary floating point cannot represent 0.10 exactly and
money that does not add up destroys trust faster than any outage.

## Retrieving a payment

`GET /v1/payments/{id}` returns the current state. States are `pending`,
`authorised`, `captured`, `failed` and `refunded`, and they only move forward.

## Rate limits

Sixty requests per second per API key, burst of one hundred. Exceeding it
returns `429` with a `Retry-After` header.
