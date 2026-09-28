# Changelog

## Unreleased

- Breaking change (minor before 1.0): remove the CircSym solver, meridian API, CPU kernels, and native Metal ring kernels. Full-3D Metal remains the supported solve path.
- Remove `SolveConfig.circsym_baffle_z`, `SolveConfig.circsym_aperture_tag`, and `SolveConfig.should_continue`; passing any of these now raises `TypeError`.
- Keep coupled infinite-baffle and pulsating-sphere validation against analytic references.
