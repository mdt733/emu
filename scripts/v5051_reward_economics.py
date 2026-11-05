"""v5051: small, deterministic counterfactual for the stepping-activity gate.

Run with: python -m scripts.v5051_reward_economics
No MuJoCo, GPU, or training framework required. This is REWARD arithmetic
only; it does not establish what a learned policy actually does.
"""
from __future__ import annotations

import numpy as np

from emu.contracts import CONTROL_DT
from emu.rewards import (
    STEPPING_ACTIVITY_TARGET,
    _gait_maturity,
    weighted_training_terms,
)


def gate(age_s: float, activity: float, *, balance: bool = False,
         vx: float = 0.0, vy: float = 0.0, yaw: float = 0.0) -> float:
    m = dict(
        age_s=np.asarray(age_s, dtype=float),
        stepping_activity_so_far=np.asarray(activity, dtype=float),
        cmd_vx=np.asarray(vx, dtype=float),
        cmd_vy=np.asarray(vy, dtype=float),
        cmd_yaw=np.asarray(yaw, dtype=float),
    )
    _, value = _gait_maturity(
        m, np, allow_zero_command_balance=balance)
    return float(value)


def posture_contact_payment(g: float, posture: float = 1.0,
                            contact: float = 1.0) -> float:
    _, terms = weighted_training_terms(
        alive=1.0, locomotion=0.0, posture=posture, contact=contact,
        anti_slip=0.0, clearance=0.0, smoothness=0.0,
        joint_safety=0.0, stepping_gate=g)
    return float(terms["posture"] + terms["contact"])


def main() -> None:
    assert abs(CONTROL_DT - 0.02) < 1e-9
    for activity in (0.0, 0.03, 0.075, 0.15, 0.20):
        g = gate(2.0, activity)
        loss = posture_contact_payment(1.0) - posture_contact_payment(g)
        print(f"activity={activity:.3f} gate={g:.3f} "
              f"best-case posture+contact penalty={loss:.4f}/frame "
              f"={loss / CONTROL_DT:.2f} reward-units/second")
    assert abs(gate(2.0, 0.0) - .30) < 1e-9
    assert abs(gate(2.0, STEPPING_ACTIVITY_TARGET) - 1.0) < 1e-9
    assert abs(gate(.5, 0.0) - 1.0) < 1e-9
    assert abs(gate(2., 0., balance=True) - 1.0) < 1e-9
    assert abs(gate(2., 0., balance=True, vx=.03) - .30) < 1e-9
    assert abs(gate(2., 0., balance=True, yaw=.06) - .30) < 1e-9
    assert abs(posture_contact_payment(1.) -
               posture_contact_payment(.30) - .826) < 1e-9
    print("PASS: arithmetic and near-zero-command exception boundaries")


if __name__ == "__main__":
    main()
