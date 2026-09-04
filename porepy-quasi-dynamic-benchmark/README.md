# BP6-QD-A PorePy Benchmark Mixin

`quasi_dynamic_PE_mainFile.py` revises the original quasi-dynamic poromechanics
mixin to simulate **BP6-QD-A**, the simplest of the three 2-D quasi-dynamic
fluid-induced aseismic slip benchmarks from:

Lambert, V. R., Erickson, B. A., Jiang, J., Dunham, E. M., Kim, T., Ampuero, J.-P.,
et al. (2025). *Community-driven code comparisons for simulations of
fluid-induced aseismic slip.* Journal of Geophysical Research: Solid Earth, 130,
e2024JB030601. https://doi.org/10.1029/2024JB030601

BP6-QD-A considers a single planar fault governed by rate-and-state friction
with the aging law for state evolution, subject to an aseismic slip transient
driven by pore-pressure changes from fluid injection at the fault center, with
a quasi-dynamic radiation-damping term (mu / 2 c_s) standing in for full
elastodynamic wave propagation.

Only the required sections of the original mixin were revised (material
constants, rate-and-state parameters, mechanical boundary tractions, injection
location/timing, and time managers) — the overall code structure (geometry,
class layout, two-stage-plus-relaxation run pattern) is unchanged. See the
in-file comments for the mapping from the paper's Table 3 parameters to this
model, and the documented scaling assumptions used to fit the published
km/days-to-years benchmark onto the file's existing 100 m demonstration domain.
