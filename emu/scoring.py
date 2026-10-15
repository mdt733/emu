"""One auditable ledger for the eight existing PPO reward payments.

These values are *training* reward contributions, not held-out promotion
scores or measured electrical energy. Keep order/associativity equivalent to
the original environment._reward expression when updating this function.
"""


def weighted_training_terms(*, alive, locomotion, posture, contact,
                            anti_slip, clearance, smoothness, joint_safety,
                            stepping_gate):
    """Return (unpenalised PPO reward, individually weighted payments).

    Array inputs may be NumPy or JAX arrays; this function is pure and does
    not know about training, physics, curriculum or evaluation.
    """
    terms = {
        'alive': 0.58 * alive,
        'locomotion': 1.55 * locomotion,
        'posture': 0.68 * posture * stepping_gate,
        'contact': 0.50 * contact * stepping_gate,
        'anti_slip': 0.18 * anti_slip,
        'clearance': 0.16 * clearance,
        'smoothness': 0.12 * smoothness,
        'joint_safety': 0.17 * joint_safety,
    }
    # Explicit left-to-right addition matches the pre-extraction expression.
    reward = (terms['alive'] + terms['locomotion'] + terms['posture'] +
              terms['contact'] + terms['anti_slip'] + terms['clearance'] +
              terms['smoothness'] + terms['joint_safety'])
    return reward, terms
