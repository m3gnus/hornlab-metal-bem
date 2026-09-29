#!/bin/bash
# Manual stress reproduction of the legacy Accelerate LAPACK failure. Not run in CI.
# usage: run.sh [ITERS] [PROCESSES]
set -eu
here="$(cd "$(dirname "$0")" && pwd)"
iters="${1:-30}"; procs="${2:-6}"
out="$(mktemp -d)"
swiftc -O "$here/schur_race.swift" -o "$out/legacy"
clang -c -O2 "$here/shim.c" -o "$out/shim.o"
swiftc -O -import-objc-header "$here/shim.h" "$here/schur_race_new.swift" "$out/shim.o" \
    -framework Accelerate -o "$out/new"
for variant in legacy new; do
    echo "== $variant: $procs processes x $iters iterations (1081 RHS, 1487 aperture unknowns, 6 threads)"
    for i in $(seq 1 "$procs"); do "$out/$variant" 1081 1487 6 "$iters" > "$out/$variant.$i.txt" 2>&1 & done
    wait
    grep -h "nonfinite" "$out/$variant".*.txt
done
echo "legacy is expected to show nonfinite > 0 in most processes; new must show 0 everywhere."
