"""Pure read-only hip-roll diagnostic for both NumPy tests and MJX exams.

Sign definition:
  passive spring torque = -stiffness * (hip_angle - springref)
  opposing centring spring: motor_torque * spring_torque < 0
  assisted by centring spring: motor_torque * spring_torque > 0

This does NOT claim a motor is fighting the spring *because* it saturates.
It reports simultaneous torque directions on actual first-life exam frames.
No reward, actuator limits, curriculum, dynamics or policy changes.
"""
from __future__ import annotations


def hip_roll_load_terms(xp, motor_nm, angle_rad, stiffness_nm_rad,
                        spring_reference_rad, rated_limit_nm):
    """Return elementwise signed-load facts; arrays broadcast as NumPy/JAX.

    `motor_nm` is a signed actual actuator force, not the NN target.
    Angle/sign conventions are those of the compiled MuJoCo hinge.
    """
    spring_nm = -stiffness_nm_rad * (angle_rad - spring_reference_rad)
    opposition = motor_nm * spring_nm < -1e-6
    assistance = motor_nm * spring_nm > 1e-6
    rated_fraction = xp.abs(motor_nm) / xp.maximum(rated_limit_nm, 1e-6)
    near95 = rated_fraction >= 0.95
    return {
        "spring_nm": spring_nm,
        "rated_fraction": rated_fraction,
        "opposition": opposition,
        "assistance": assistance,
        "near95": near95,
        "near95_opposing": near95 & opposition,
    }
