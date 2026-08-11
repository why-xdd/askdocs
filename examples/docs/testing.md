# Testing

## What we test

Behaviour, not implementation. A test that breaks when a private method is
renamed is a maintenance cost with no safety benefit.

## The pyramid, roughly

Most coverage comes from fast unit tests. Integration tests cover the seams
between services, and there is a small set of end-to-end tests that exercise a
real payment against the acquirer sandbox.

## Flaky tests

A flaky test is deleted or fixed within a week. Tolerating one teaches everyone
to ignore red builds, which costs more than the test was ever worth.

## Fixtures

Prefer building test data in the test over shared fixtures. A fixture used by
forty tests cannot be changed by any of them.
