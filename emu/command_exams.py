"""Shared deterministic v5016 operator-course schedules.

A schedule is a time-indexed series of *operator* (vx,vy,yaw_rate) requests.
It is never derived from the robot's observed or delivered velocity.
"""
from __future__ import annotations

import math
import numpy as np

SCENARIOS = ('steady', 'turn_stop_left', 'turn_stop_right',
             'sidestep_left', 'sidestep_right', 'reverse', 'stop_resume')


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
    cmds = np.tile(np.array([speed, 0., yaw_rate], dtype=np.float32), (n, 1))
    a = n//4
    b = n*2//3
    if scenario in ('turn_stop_left','turn_stop_right'):
        cmds[:,2] = 0.
        cmds[a:b,2] = turn_rate if scenario.endswith('left') else -turn_rate
    elif scenario in ('sidestep_left','sidestep_right'):
        cmds[:,2] = 0.
        cmds[a:b,0] = 0.
        cmds[a:b,1] = lateral if scenario.endswith('left') else -lateral
    elif scenario == 'reverse':
        cmds[:,2] = 0.
        cmds[a:b,0] = -abs(speed)
    elif scenario == 'stop_resume':
        cmds[:,2] = 0.
        cmds[a:b,0] = 0.
    return cmds
