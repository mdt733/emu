"""Deterministic operator-requested motion schedules for held-out exams.

Only command requests, not robot-derived corrective trajectories.
"""
from __future__ import annotations

# Operator-command courses: unmodified requests, never robot-derived.
import math
import numpy as np

SCENARIOS = ('steady', 'turn_stop_left', 'turn_stop_right',
             'sidestep_left', 'sidestep_right', 'reverse', 'stop_resume',
             'rest_start', 'speed_step', 'brake_stop')


def make_schedule(scenario, horizon, *, speed=.10, lateral=.06,
                  yaw_rate=0., turn_rate=.25):
    if scenario not in SCENARIOS:
        raise ValueError('unknown scenario: '+str(scenario))
    if not 30 <= int(horizon) <= 700:
        raise ValueError('schedule horizon must be between 30 and 700 frames')
    if not all(math.isfinite(x) for x in (speed, lateral, yaw_rate, turn_rate)):
        raise ValueError('commands must be finite')
    if abs(speed) > .6 or abs(lateral) > .25 or abs(yaw_rate) > 1.5 or not 0 < turn_rate <= 1.5:
        raise ValueError('command outside safe test envelope')
    n = int(horizon)
    cmds = np.tile(np.array([speed, lateral, yaw_rate], dtype=np.float32), (n, 1))
    a = n//4
    b = n*2//3
    if scenario in ('turn_stop_left','turn_stop_right'):
        cmds[:,2] = 0.
        cmds[a:b,2] = turn_rate if scenario.endswith('left') else -turn_rate
    elif scenario in ('sidestep_left','sidestep_right'):
        if abs(lateral) < 1e-6:
            raise ValueError('sidestep exam requires a nonzero --lateral command')
        cmds[:,2] = 0.
        cmds[:,1] = 0.
        cmds[a:b,0] = 0.
        cmds[a:b,1] = abs(lateral) * (1 if scenario.endswith('left') else -1)
    elif scenario == 'reverse':
        if np.hypot(speed,lateral) < .02:
            raise ValueError('reverse exam requires a nonzero horizontal command')
        cmds[:,2] = 0.
        cmds[a:b,:2] *= -1.
    elif scenario == 'stop_resume':
        cmds[:,2] = 0.
        cmds[a:b,:2] = 0.
    elif scenario == 'rest_start':
        cmds[:,2] = 0.
        cmds[:a,:2] = 0.
    elif scenario == 'speed_step':
        cmds[:,2] = 0.
        cmds[:a,:2] *= .4
    elif scenario == 'brake_stop':
        cmds[:,2] = 0.
        cmds[b:,:2] = 0.
    return cmds
