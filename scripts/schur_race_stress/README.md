# Schur-solve race reproduction (manual)

Standalone reproduction of the Accelerate legacy-LAPACK failure that the native helper
no longer has: `cgesv_` returns NaN when the coupled infinite-baffle Schur sequence
(`zgesv`, `cgemm`, `cgesv`) runs on 6 threads at once. No helper code is involved.

Run `scripts/schur_race_stress/run.sh [ITERS] [PROCESSES]` on an Apple Silicon Mac. It
builds `schur_race.swift` (legacy `__CLPK_*` entry points) and `schur_race_new.swift`
(`$NEWLAPACK` entry points via `shim.c`) and runs each in several processes. The legacy
build is expected to report `nonfinite` above 0 in most processes; the new build must
report 0. The failure is probabilistic (roughly 5-10 % of solves under load), and it needs
a few dozen iterations per process. This is a manual tool: it is not part of the test
suite or CI.
