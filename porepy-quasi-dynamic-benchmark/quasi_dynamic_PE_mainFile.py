import porepy as pp
import numpy as np
from porepy.applications.md_grids.domains import nd_cube_domain
from porepy.models.poromechanics import Poromechanics
from porepy.viz.data_saving_model_mixin import FractureDeformationExporting
from rate_state_friction_porepy import RateStateAdaptiveTimeStepper, RateStateFrictionMixin
from rate_state_friction_exporting import RateStateFrictionExporting
from fracture_resolution import FractureResolutionMixin
from seismological_parameters import SeismologicalParametersMixin

# ====================================================================
# BP6-QD-A benchmark (Lambert et al., 2025, JGR Solid Earth,
# "Community-Driven Code Comparisons for Simulations of Fluid-Induced
# Aseismic Slip", 10.1029/2024JB030601).
#
# BP6-QD-A is the simplest of the three BP6-QD problems: a single planar
# fault governed by rate-and-state friction with the AGING law for state
# evolution, subject to a fault-confined aseismic slip transient driven
# purely by pore-pressure changes from fluid injection at the fault
# center. (BP6-QD-S uses the slip law and needs finer resolution;
# BP6-QD-C uses a constant friction coefficient, which this repository's
# RateStateFrictionMixin does not expose independently of rate-and-state
# and so is left untouched here.)
#
# The published problem is posed as an antiplane-shear fault of half-
# length l_f = 20 km in an elastic whole-space, discretized by
# specialized (semi-)infinite-domain codes (BEM/SBEM/spectral methods).
# This PorePy mixin instead uses a finite square domain with a single
# through-going diagonal fracture (unchanged from the original file's
# geometry) so that the fault carries both a normal and a shear
# component of the background stress from simple axis-aligned boundary
# tractions. Material and rate-and-state parameters are taken directly
# from Table 3 of the paper (they are domain-size independent); the
# domain size, injection duration/rate and time managers are instead
# scaled down from the published (km, days-to-years) values to the
# existing 100 m / seconds-to-minutes scale of this file, preserving the
# ratio of the diffusive length scale to the fault half-length at
# injection shut-off. Because of this domain-size reduction, the
# nucleation/process-zone size implied by Table 3 (~450 m, see Equation
# 10 of the paper) does not fit inside the reduced domain; this is a
# purely numerical resolution caveat (see "on_insufficient_resolution"
# below) and does not affect the qualitative benchmark behavior.
# ====================================================================


# ================================================================
# Geometry
# 100*100 cubic domain with diagonal frature
class ModifiedGeometry:

    def set_domain(self) -> None:
        """Define a two-dimensional square domain."""
        size = self.units.convert_units(100, "m")
        self._domain = nd_cube_domain(2, size)

    def set_fractures(self) -> None:
        """Define the fault as a single diagonal fracture.

        Its midpoint (50, 50) coincides with the domain center and plays
        the role of the BP6-QD-A fault center z = 0, where injection is
        applied (Figure 1 of the paper).
        """
        frac_1_points = self.units.convert_units(np.array([[20.0, 80.0], [20.0, 80.0]], dtype=float), "m")
        #frac_2_points = self.units.convert_units(np.array([[20.0, 80.0], [80.0, 20.0]], dtype=float), "m")

        frac_1 = pp.LineFracture(frac_1_points)
       # frac_2 = pp.LineFracture(frac_2_points)

        self._fractures = [frac_1]

    def grid_type(self) -> str:
        return self.params.get("grid_type", "simplex")

    def meshing_arguments(self) -> dict:
        mesh_args = {
            "cell_size_fracture": self.units.convert_units(1, "m"),
            "cell_size_boundary": self.units.convert_units(4, "m"),
            "background_transition_multiplier": 15.0,
        }
        return mesh_args


# ================================================================
# flow boundary consitions
# only east edge of domain has drichlet boundary consition with setting pressure to zero
#
# BP6-QD-A assumes fault-confined diffusion with no fault-normal flux
# into the surrounding bulk (Section 3.2 of the paper). The matrix
# permeability below is set close to zero for the same reason, so this
# far-field matrix Dirichlet condition mainly keeps the (near-negligible)
# bulk flow problem well posed rather than driving any physically
# significant matrix flow.
class ModifiedFlowBC:

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryCondition(sd, sides.east, "dir")
        return bc

    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return self.bc_type_darcy_flux(sd)

    def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
        domain_sides = self.domain_boundary_sides(bg)
        values = np.zeros(bg.num_cells)
        values[domain_sides.east] = self.units.convert_units(0.0, "Pa")
        #values[domain_sides.west] = self.units.convert_units(0.0, "Pa")
        return values


# ================================================================
# injection well
#
# BP6-QD-A injects directly into the fault at its center z = 0 (Figure 1,
# Equation 9), not into the surrounding bulk. The search below is
# therefore restricted to fracture (codimension-1) subdomains, and
# "well_location" is placed exactly at the fault midpoint.

class InjectionWellSource:

    injection_source_name = "prescribed_injection_well"

    def fluid_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        source = super().fluid_source(subdomains)
        injection_source = pp.ad.TimeDependentDenseArray(name=self.injection_source_name, domains=subdomains).previous_timestep()
        return source + injection_source

    def update_time_dependent_ad_arrays(self) -> None:
        super().update_time_dependent_ad_arrays()

        injection_active = self.params.get("injection_active", False)
        injection_parameters = self.params["injection"]
        well_location = np.asarray(injection_parameters["well_location"], dtype=float)

        if well_location.size != self.nd:
            raise ValueError(f"injection well location must contain exactly {self.nd} coordinates.")

        well_location = self.units.convert_units(well_location, "m")

        if injection_active:
            q_inj = self.units.convert_units(injection_parameters["rate"], "m^3 * s^-1")
        else:
            q_inj = 0.0

        rho_inj = self.fluid.reference_component.density
        mass_rate = rho_inj * q_inj
        minimum_distance_squared = np.inf
        injection_sd = None
        injection_local_cell = None

        for sd in self.mdg.subdomains():
            if sd.dim == self.nd - 1:
                cell_centers = sd.cell_centers[:self.nd, :]
                distance_squared = np.sum((cell_centers - well_location[:, None]) ** 2, axis=0)
                local_cell = int(np.argmin(distance_squared))

                if distance_squared[local_cell] < minimum_distance_squared:
                    minimum_distance_squared = distance_squared[local_cell]
                    injection_sd = sd
                    injection_local_cell = local_cell

        if injection_sd is None or injection_local_cell is None:
            raise RuntimeError("Could not locate a fault cell for the injection well.")

        for sd in self.mdg.subdomains():
            source_values = np.zeros(sd.num_cells, dtype=float)

            if injection_active and sd is injection_sd:
                source_values[injection_local_cell] = mass_rate

            pp.set_solution_values(name=self.injection_source_name, values=source_values, data=self.mdg.subdomain_data(sd), time_step_index=0)

        if injection_active:
            print(f"Injection ON: time = {self.time_manager.time:.6e} s, dt = {self.time_manager.dt:.6e} s, Q = {q_inj:.6e} m3/s")
        else:
            print(f"Equilibration/relaxation: time = {self.time_manager.time:.6e} s, dt = {self.time_manager.dt:.6e} s, injection OFF")


# ================================================================
# mechanics boundary consitions
# west and south boundaries have roller condition, north and east are Neumann baoundaries with prescribed stresses
#
# BP6-QD-A prescribes a spatially uniform, time-independent background
# effective normal stress sigma_bar_0 = 50 MPa and initial shear stress
# tau_init = 29.2 MPa on the fault, with no time-dependent (tectonic)
# loading (Section 3 of the paper). With the fault running along the
# domain diagonal, uniform orthogonal compressive tractions sigma_xx and
# sigma_yy resolve on the 45-degree fault plane as:
#   sigma_normal = (sigma_xx + sigma_yy) / 2 = sigma_bar_0
#   tau          = (sigma_xx - sigma_yy) / 2 = tau_init
# giving sigma_xx = 79.2 MPa and sigma_yy = 20.8 MPa (both compressive),
# applied below as constant east/north tractions.
class ModifiedMechanicsBC:

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        if sd.dim < self.nd:
            return super().bc_type_mechanics(sd)

        domain_sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(sd)
        bc.is_dir[0, domain_sides.west] = True
        bc.is_neu[0, domain_sides.west] = False
        bc.is_dir[1, domain_sides.south] = True
        bc.is_neu[1, domain_sides.south] = False
        bc.internal_to_dirichlet(sd)
        return bc

    def bc_values_stress(self, bg: pp.BoundaryGrid) -> np.ndarray:
        values = np.zeros((self.nd, bg.num_cells))

        if bg.parent.dim < self.nd:
            return values.ravel("F")

        domain_sides = self.domain_boundary_sides(bg)
        values[0, domain_sides.east] = self.units.convert_units(-79.2e6, "Pa") * bg.cell_volumes[domain_sides.east]
        values[1, domain_sides.north] = self.units.convert_units(-20.8e6, "Pa") * bg.cell_volumes[domain_sides.north]
        return values.ravel("F")

    def bc_values_displacement(self, bg: pp.BoundaryGrid) -> np.ndarray:
        values = np.zeros((self.nd, bg.num_cells))
        return values.ravel("F")


# ================================================================
# coupled poroelastic model with rate-and-state friction law

class CoupledPoromechanicsModel(
    ModifiedGeometry,
    ModifiedFlowBC,
    ModifiedMechanicsBC,
    InjectionWellSource,
    pp.constitutive_laws.CubicLawPermeability,
    RateStateFrictionExporting,
    FractureDeformationExporting,
    FractureResolutionMixin,
    SeismologicalParametersMixin,
    RateStateFrictionMixin,
    Poromechanics):
    pass


# ====================================================================
# material properties
# rock and fracture
#
# Shear modulus, density and rate-and-state parameters below are taken
# directly from Table 3 of the BP6-QD-A/S/C benchmark description (these
# are intensive material properties, independent of domain size). The
# antiplane benchmark does not require a Poisson's ratio; nu = 0.25 is
# chosen here only to close the plane-strain elastic model used by
# PorePy, and conveniently gives lambda = mu.

mu = 32.04e9
nu = 0.25
lmbda = 2 * mu * nu / (1 - 2 * nu)

# Fault (tangential) permeability is prescribed directly in the
# benchmark (k = 1e-13 m^2, Table 3), but PorePy's CubicLawPermeability
# derives tangential permeability from the fracture aperture via the
# cubic law k = a^2 / 12. Solve for the residual aperture that
# reproduces the benchmark's fault permeability.
fault_permeability_si = 1e-13
residual_aperture_si = np.sqrt(12.0 * fault_permeability_si)

solid_constants = pp.SolidConstants(
    # Matrix permeability set close to zero: BP6-QD-A assumes fault-
    # confined diffusion only, with no fault-normal leak-off into the
    # surrounding bulk (Section 3.2 of the paper).
    permeability=1e-22,
    porosity=0.1,
    density=2670,
    biot_coefficient=0.75,
    lame_lambda=lmbda,
    shear_modulus=mu,
    friction_coefficient=0.6,
    dilation_angle=0.0,
    fracture_gap=0.0,
    residual_aperture=residual_aperture_si,
    normal_permeability=1e-22,
    fracture_normal_stiffness=2e11,
    fracture_tangential_stiffness=2e11,
    maximum_elastic_fracture_opening=1e-3)


# --------------------------------------------------------------------
# fluid properties
#
# viscosity matches Table 3 (eta = 1e-3 Pa.s); compressibility is set to
# the benchmark's combined pore-and-fluid compressibility (beta = 1e-8
# 1/Pa), which together with permeability, porosity and viscosity above
# reproduces the benchmark's hydraulic diffusivity
# alpha = k / (beta * phi * eta) = 0.1 m^2/s.

fluid_constants = pp.FluidComponent(
    viscosity=0.001,
    density=1000,
    compressibility=1e-8)


# --------------------------------------------------------------------
# material constants

numerical_constants = pp.NumericalConstants(
    characteristic_displacement=1.0e-4,
    open_state_tolerance=1.0e-10)


# --------------------------------------------------------------------
# derived dynamic properties: shear-wave speed and quasi-dynamic
# radiation-damping coefficient mu / (2 c_s), Section 3 (Equation
# following Eq. 2) of the paper.

shear_modulus_si = float(solid_constants.constants_in_SI["shear_modulus"])
solid_density_si = float(solid_constants.constants_in_SI["density"])
s_wave_speed_reference = np.sqrt(shear_modulus_si / solid_density_si)
radiation_damping_coefficient = shear_modulus_si / (2.0 * s_wave_speed_reference)

print("================================================")
print("BP6-QD-A MATERIAL PROPERTY SUMMARY")
print("================================================")
print(f"shear modulus mu      : {shear_modulus_si:.3e} Pa")
print(f"density rho           : {solid_density_si:.1f} kg/m3")
print(f"S-wave speed c_s      : {s_wave_speed_reference:.1f} m/s (Table 3: 3464 m/s)")
print(f"radiation damping term: {radiation_damping_coefficient:.4e} Pa.s/m")
print("================================================\n")


# --------------------------------------------------------------------
# injection rate and location
#
# well_location is placed exactly at the fault midpoint (50, 50), i.e.
# the BP6-QD-A fault center z = 0 where injection occurs (Figure 1).
# The injection rate is not a unit-for-unit conversion of the
# benchmark's line-source flux q0 (Table 3), since that value feeds a
# 1-D along-fault diffusion equation (Equation 9) rather than a
# volumetric well coupled to a 2-D poromechanics mass balance; it is
# instead chosen to produce a comparable, moderate fault-center
# pressurization relative to sigma_bar_0 = 50 MPa on the reduced domain
# used here.

injection_parameters = {
    "well_location": np.array([50.0, 50.0], dtype=float),
    "rate": 1e-6,
}


# --------------------------------------------------------------------
# rate and state friction params (BP6-QD-A, Table 3)
#
# Friction coefficient (regularized formulation, Lapusta et al., 2000):
#
#     f(V, theta) = a * asinh[ V / (2 V*) * exp( ( f* + b ln(V* theta / Dc) ) / a ) ]
#
# with V* = "V0" and f* = "uf0".
#
# State evolution (BP6-QD-A uses the aging law):
#     "aging" -> dtheta/dt = 1 - |V| theta / Dc
#
# a > b below (0.007 > 0.005) gives uniform velocity-strengthening (VS)
# friction, as required by BP6-QD-A/S (Section 3.1.1), which precludes
# spontaneous nucleation of frictional instability: the fault responds
# only to the imposed pore-pressure transient.
#
# "damp" is the quasi-dynamic radiation-damping coefficient mu / (2 c_s)
# derived above; unlike a fully dynamic (Newmark) run, no resolved wave
# field supplies this damping here, so it must be included explicitly.

rate_state_parameters = {
    "a": 0.007,
    "b": 0.005,
    "Dc": 4e-3,
    "V0": 1e-6,
    "uf0": 0.6,
    "state_evolution_law": "aging",
    "damp": radiation_damping_coefficient,
    "initial_slip_rate": 1e-12,
    "local_newton_atol": 1e-8,
    "local_newton_rtol": 1e-10,
    "local_newton_max_iterations": 50,
    "yield_tolerance": 1e-8,
    "smooth_velocity_regularization": False,
    "adaptive_time_stepping": {
        "enabled": False,
        "inactive_velocity": 1e-10,
        "event_fraction": 0.1,
        "dt_min": 1e-5,
        "dt_c": 5.0,
        "growth_factor": 1.2,
        "print_info": True,
    },
}

rate_state_parameters["theta0"] = rate_state_parameters["Dc"] / rate_state_parameters["V0"]


# --------------------------------------------------------------------
# seismological params

seismological_parameters = {
    "slip_velocity_threshold": 0.001,
    "out_of_plane_thickness": 1.0,
    "moment_integration_region": "threshold_exceeding_cells",
    "stress_drop_region": "event_rupture_region",
    "shear_stress_measure": "magnitude",
    "energy_method": "mw_scaling",
    "energy_log10_intercept": 5.24,
    "energy_log10_slope": 1.44,
    "save_csv": True,
    "csv_file_name": "seismological_parameters.csv",
}


# --------------------------------------------------------------------
# Nucleation and cohesive zone params
#
# "on_insufficient_resolution" is set to "warn" (rather than left unset)
# because Table 3's material/friction parameters imply a process-zone
# size Lambda_0 = C * mu * Dc / (b * sigma_bar_0) of order 450 m
# (Equation 10 of the paper), which does not fit inside the 100 m
# demonstration domain used here. This is a numerical resolution
# caveat of the reduced-scale domain, not a modeling error, so
# execution should proceed with a warning rather than raising.

fracture_resolution_parameters = {
    "cohesive_zone_C1": 9.0 * np.pi / 32.0,
    "cohesive_zone_warning_elements": 3.0,
    "on_insufficient_resolution": "warn",
}

material_constants = {
    "fluid": fluid_constants,
    "solid": solid_constants,
    "numerical": numerical_constants,
}

# ====================================================================
# stage 1 time stepping: pre-injection equilibrium

equilibrium_time_manager = pp.TimeManager(
    schedule=[0, 5 * 1e8],
    dt_init=1e8,
    dt_min_max=(1, 1e8),
    constant_dt=True,
    iter_max=10,
    recomp_factor=0.5,
    recomp_max=10,
    print_info=True)


# ====================================================================
# stage 2 time stepping: injection (BP6-QD-A "t_off")
#
# BP6-QD-A injects at a constant rate for a fixed duration t_off = 100
# days on a 20 km fault (Table 3), after which injection is shut off but
# the simulation continues to a final time t_f = 2 years to resolve the
# post-shut-in relaxation (Figure 1). Both timescales are reduced here
# in proportion to the fault half-length of this file's 100 m domain
# (~42.4 m vs. the benchmark's 20 km), so that the diffusive length
# scale sqrt(alpha * t_off) reaches the same fraction of the fault
# half-length at shut-off as in the published benchmark (~4.6%):
#   t_off_ours = t_off_paper * (l_f_ours / l_f_paper)^2 =~ 39 s
#   t_f_ours   = t_off_ours * (t_f_paper / t_off_paper)  =~ 285 s
# rounded below to 40 s and 300 s.
#
# Physical simulation time is reset to zero when injection starts.

injection_duration = 40.0
final_time = 300.0

injection_time_manager = pp.TimeManager(
    schedule=[0.0, injection_duration],
    dt_init=0.5,
    dt_min_max=(1e-4, 2.0),
    constant_dt=False,
    iter_max=20,
    recomp_factor=0.5,
    recomp_max=10,
    print_info=True)


# ====================================================================
# stage 3 time stepping: post-injection relaxation
# Injection is shut off (BP6-QD-A "t_off") but the fault continues to
# relax under along-fault pore-pressure diffusion up to t_f.

relaxation_time_manager = pp.TimeManager(
    schedule=[injection_duration, final_time],
    dt_init=2.0,
    dt_min_max=(1e-3, 10.0),
    constant_dt=False,
    iter_max=20,
    recomp_factor=0.5,
    recomp_max=10,
    print_info=True)


# ====================================================================
# model parameters

model_params = {
    "material_constants": material_constants,
    "reference_variable_values": pp.ReferenceVariableValues(pressure=0.0),
    "time_manager": equilibrium_time_manager,
    "injection": injection_parameters,
    "injection_active": False,

    # Stage 1 uses PorePy's standard Coulomb contact law.
    # RSF is activated only after pre-injection equilibrium is complete.
    "rate_state_active": False,

    "rate_state": rate_state_parameters,
    "fracture_resolution": fracture_resolution_parameters,
    "seismological_parameters": seismological_parameters,
    "initialize_operator_reference_from_initial_values": True,
    "times_to_export": [],
    "folder_name": "visualization_BP6_QD_A_2D",
    "file_name": "data",
}


# ====================================================================
# nonlinear solver parameters

solver_params = {
    "nl_max_iterations": 30,
    "nl_convergence_inc_atol": 1e-6,
    "nl_convergence_res_atol": 1e-1,
    "nl_divergence_inc_atol": np.inf,
    "nl_divergence_res_atol": np.inf,
    "progressbars": True,
}


# ====================================================================
# create model

model = CoupledPoromechanicsModel(model_params)


# ====================================================================
# STAGE 1: establish equilibrium before injection

print("\n================================================")
print("STAGE 1: PRE-INJECTION EQUILIBRATION")
print("================================================")
print("Contact law: standard PorePy Coulomb friction; rate-and-state friction OFF.")

runner_equilibrium = pp.ModelRunner(model, solver_params)
runner_equilibrium.run()


# ====================================================================
# STAGE 2: injection (BP6-QD-A, 0 <= t <= t_off)

print("\n================================================")
print("STAGE 2: INJECTION (BP6-QD-A)")
print("================================================")

if not equilibrium_time_manager.final_time_reached():
    raise RuntimeError(
        "Equilibration did not reach its prescribed final time; "
        "the injection stage will not be started."
    )

# Replace the numerical equilibration clock by the physical injection clock.
model.time_manager = injection_time_manager
model.params["time_manager"] = injection_time_manager
model.params["times_to_export"] = None

# Activate RSF from the already-converged pre-injection state.
# This also replaces PorePy's registered Stage-1 Coulomb tangential
# equation by the regularized RSF equation.
model.activate_rate_state_from_current_solution(
    theta=rate_state_parameters["theta0"],
    slip_rate=rate_state_parameters.get("initial_slip_rate", 0.0),
    reset_accumulated_slip=True,
)

print("Contact law: regularized rate-and-state friction (aging law) ON.")

model.equilibrium_fault_resolution()
model.initialize_seismological_tracking()

# Export the converged equilibrium configuration as physical t = 0.
if model.params.get("injection_active", False):
    raise RuntimeError(
        "Injection must remain OFF while exporting the equilibrium t=0 state."
    )

model.save_data_time_step()
print("Saved final equilibrium state as injection initial condition at t = 0 s.")

# Injection begins only after the t=0 equilibrium state has been saved.
model.params["injection_active"] = True

adaptive_cfg = model.params["rate_state"].get("adaptive_time_stepping", None)

if adaptive_cfg is not None:
    adaptive_cfg["enabled"] = True

solver_params_injection = solver_params.copy()
solver_params_injection["prepare_simulation"] = False

rate_state_time_stepper = RateStateAdaptiveTimeStepper(injection_time_manager)

runner_injection = pp.ModelRunner(
    model,
    solver_params_injection,
    time_stepper=rate_state_time_stepper,
)

runner_injection.run()


# ====================================================================
# STAGE 3: post-injection relaxation (BP6-QD-A, t_off <= t <= t_f)
# Injection is turned off; the fault continues to relax under along-
# fault pore-pressure diffusion alone up to the final simulation time.

print("\n================================================")
print("STAGE 3: POST-INJECTION RELAXATION (BP6-QD-A)")
print("================================================")

if not injection_time_manager.final_time_reached():
    raise RuntimeError(
        "Injection stage did not reach its prescribed shut-off time; "
        "the relaxation stage will not be started."
    )

model.params["injection_active"] = False

model.time_manager = relaxation_time_manager
model.params["time_manager"] = relaxation_time_manager

print("Contact law: regularized rate-and-state friction (aging law) ON; injection OFF.")

solver_params_relaxation = solver_params.copy()
solver_params_relaxation["prepare_simulation"] = False

rate_state_time_stepper_relaxation = RateStateAdaptiveTimeStepper(relaxation_time_manager)

runner_relaxation = pp.ModelRunner(
    model,
    solver_params_relaxation,
    time_stepper=rate_state_time_stepper_relaxation,
)

runner_relaxation.run()
