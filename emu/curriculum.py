"""Current EMU curriculum controller.

This module owns the task distribution only: regime difficulty/allocation, core
retreat, and demonstrated physical frontiers. PPO/training health and rollback
decisions live in TrainingHealth below.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields as dataclass_fields
from typing import Any
import math

from .contracts import CONTROL_DT
from .training_config import FRONTIER_GROWTH_FRAC, FRONTIER_RETREAT_FRAC, FRONTIER_SENSOR_SAFETY

# ---------------------------------------------------------------------------
# Compact organic-birth curriculum/governor.
# ---------------------------------------------------------------------------

DEFAULT_DIFFICULTY_BASE_STEP = 0.0014
DEFAULT_FALL_TARGET = 0.30

@dataclass
class RegimeParams:
    """Curriculum state: four 0-1 difficulty scalars (one per regime, plus
    posture) and three regime-allocation fractions. environment.py's own
    _lerp expands each difficulty scalar into a real physical range -- see
    its module doctrine for why one shared linear scalar per regime is
    enough, with no separate per-component rate table.
    """
    steady_difficulty: float = 0.0
    transition_difficulty: float = 0.0
    resilience_difficulty: float = 0.0
    posture_difficulty: float = 0.0
    regime_frac_steady: float = 1.0
    regime_frac_transition: float = 0.0
    regime_frac_resilience: float = 0.0
    regime_command_frac: float = 0.5
    athletic_rehearsal_frac: float = 0.0
    # Demonstrated physical boundaries, not 0-1 completion scalars.
    frontier_vx_hi: float = 0.60
    frontier_vx_lo: float = -0.20
    frontier_vy: float = 0.22
    frontier_yaw: float = 1.50
    frontier_push_n: float = 0.70
    frontier_sensor_scale: float = 1.50
    frontier_combo_radius: float = 1.00
    frontier_transition_hold_s: float = 0.90
    frontier_ramp_steps: float = 4.0
    # Opt-in physical terrain, NOT a new regime. Both remain dormant until
    # real held-out crossing evidence is available for adaptive advancement.
    # Saved with ordinary governor rollback/resume state.
    terrain_exposure: float = 0.0
    terrain_height_m: float = 0.0
    # True inclined collision slabs; independent of discontinuous steps.
    slope_exposure: float = 0.0
    slope_max_deg: float = 0.0


class RegimeGovernor:
    """Curriculum controller for STEADY, TRANSITION, RESILIENCE and POSTURE.

    Each regime receives live data from the mixed vectorised rollout. Smoothed
    fall rate drives a banded proportional controller: expand below the target,
    hold near it, and retreat above it. Regime allocation favours skills that
    currently need more training while preserving a usable mix.

    Difficulty expansions share one combined per-update budget; retreats do
    not. Broad core neglect may retreat curriculum, while PPO/collapse health and
    rollback decisions remain separate in ``health.py``. Dynamic state is fully
    serialisable for exact resume/recovery.
    """

    REGIMES = ('steady', 'transition', 'resilience')
    # At 5% births, each PPO rollout contains too few finished terrain
    # episodes for reliable up/down decisions. Keep a short saved window.
    TERRAIN_MIN_ATTEMPTS_PER_DIRECTION = 20
    TERRAIN_MAX_WINDOW_UPDATES = 40
    TERRAIN_ASCENT_PRACTICE_MIN = 0.10
    TERRAIN_DESCENT_PRACTICE_MIN = 0.50
    TERRAIN_ASCENT_PRACTICE_MAX_EXPOSURE = 0.10
    COMMAND_SECTORS = ('vx_hi','vx_lo','vy_pos','vy_neg','yaw_pos','yaw_neg','mixed')
    COMMAND_MIN_EDGE_EPISODES = 4
    COMMAND_MIN_EASY_EPISODES = 4
    SLOPE_DIRECTIONS = ('uphill','downhill','cross_left','cross_right')
    SLOPE_MIN_EPISODES_PER_DIRECTION = 4

    # One shared step size for ALL FOUR difficulty scalars (steady/
    # transition/resilience/posture) -- see environment.py's module
    # doctrine for the calibration finding that justifies a single shared
    # rate instead of a separately-tuned step per axis.
    FALL_EMA_ALPHA = 0.25        # smoothing before the controller reacts, cuts noise-driven jitter
    PUSH_GAIN = 1.0              # push-side step scale at fall_ema == 0 (maximally easy)
    RETREAT_GAIN = 2.5           # retreat-side step scale at fall_ema == 1.0 -- bigger than PUSH_GAIN
                                    # on purpose, same reasoning SeedGovernor always used
    MAX_STEP_MULT = 3.0          # cap on either side's step, in units of difficulty_base_step
    MIN_ACTIONABLE_MULT = 0.05   # below this, treat as "at target" and don't move
    ALLOCATION_FLOOR = 0.10      # regime-allocation draw weight = FLOOR + fall_ema[regime] -- keeps a
                                    # currently-solid regime from dropping to near-zero envs forever
    BOOTSTRAP_UPDATES = 70       # steady-only updates before allocation opens up to all three regimes
                                    # (same magnitude as SeedGovernor's own TURN_LENGTH_UPDATES bootstrap)

    COMPETENCE_FLOOR = 0.35              # push also requires this -- don't expand what the robot can't do at all
    CORE_MIN_COMPETENCE = 0.50           # below this, the core (seed-band) is being neglected
    FALL_SEVERE_THRESHOLD = 0.65         # used by core-neglect and the stepping-collapse detector below
    COMPETENCE_SEVERE_THRESHOLD = 0.20
    CORE_RETREAT_STREAK_LIMIT = 20       # consecutive core-neglect updates before a hard retreat fires

    # Physical-frontier growth/retreat uses settled edge tracking with EMA
    # hysteresis; results.py owns rollout metrics and contracts.py owns the tracking tolerances.
    FRONTIER_EXPAND_PRECISION = 0.62
    FRONTIER_RETREAT_PRECISION = 0.28
    FRONTIER_PRECISION_EMA_ALPHA = 0.15
    FRONTIER_BAD_STREAK_LIMIT = 8
    PRECISION_FLOOR = 0.50

    # Retreat toward the proven athletic envelope; never erase known competence.
    FRONTIER_FLOOR_VX_HI = 0.60
    FRONTIER_FLOOR_VX_LO = -0.20
    FRONTIER_FLOOR_VY = 0.22
    FRONTIER_FLOOR_YAW = 1.50
    FRONTIER_FLOOR_COMBO = 1.00



    def __init__(self, fall_target: float | None = None,
                 difficulty_base_step: float | None = None,
                 regime_command_frac: float | None = None, athletic_rehearsal_frac: float | None = None,
                 open_frontier: bool = False, frontier_vx_hi_safety: float = 1.0,
                 frontier_vx_lo_safety: float = 0.45, frontier_vy_safety: float = 0.45,
                 frontier_yaw_safety: float = 3.0, frontier_combo_safety: float = 1.60,
                 frontier_push_safety_n: float = 3.0,
                 unified_capability_focus: bool = False):
        self.params = RegimeParams()
        self.unified_capability_focus = bool(unified_capability_focus)
        self.command_evidence = {k:self._empty_command_window()
                                 for k in self.COMMAND_SECTORS}
        self.slope_evidence = {k:self._empty_slope_window()
                               for k in self.SLOPE_DIRECTIONS}
        if self.unified_capability_focus:
            # These are initial *proven/being-probed* limits, not a
            # declaration that a cold robot can already do 0.6m/s.
            # Main's initial steady seed is +.02…+.06m/s, NOT mature .6.
            # Other axes get small nonzero probe magnitudes so they can
            # eventually earn competence and escape a zero-sized frontier.
            self.params.frontier_vx_hi = .06
            self.params.frontier_vx_lo = -.02  # Signed reverse speed, as in main.
            self.params.frontier_vy = .035
            self.params.frontier_yaw = .14
            self.params.frontier_combo_radius = 1.0
        if regime_command_frac is not None:
            self.params.regime_command_frac = float(regime_command_frac)
        if athletic_rehearsal_frac is not None:
            self.params.athletic_rehearsal_frac = float(athletic_rehearsal_frac)
        self.competence: dict[str, float] = {r: 0.0 for r in self.REGIMES}
        self.core_competence = 1.0  # optimistic prior: no evidence of core neglect yet
        self.best_len = 0.0
        self.core_bad_streak = 0
        # One fall_ema per regime plus one for posture -- each gets its own
        # honest, continuously-updated signal every update, since every
        # regime has a live, continuous slice of envs every update (no
        # turn-taking to cause one regime's signal to go stale between
        # turns).
        self.fall_ema = {r: 1.0 for r in self.REGIMES}
        self.fall_ema['posture'] = 1.0
        # Pooled count EMAs behind fall_ema (see _pooled_fall_ema). Start at
        # zero: fall_ema keeps its conservative 1.0 prior only until the
        # first real episode of that kind completes, then is pure data.
        self._falls_ema = {k: 0.0 for k in self.fall_ema}
        self._episodes_ema = {k: 0.0 for k in self.fall_ema}
        self._updates_done_total = 0
        self.terrain_evidence = self._empty_terrain_evidence()
        self.frontier_precision_ema = {k: -1.0 for k in ('vx_hi','vx_lo','vy','yaw','combo','accel')}
        self.frontier_bad_streak = {k: 0 for k in ('vx_hi','vx_lo','vy','yaw','combo','accel')}
        self.fall_target = DEFAULT_FALL_TARGET if fall_target is None else float(fall_target)
        self.difficulty_base_step = (
            DEFAULT_DIFFICULTY_BASE_STEP if difficulty_base_step is None else float(difficulty_base_step)
        )
        self.open_frontier = bool(open_frontier)
        # Safety ceilings, not target commands: edge competence must earn growth.
        self.frontier_vx_hi_safety = max(self.FRONTIER_FLOOR_VX_HI, float(frontier_vx_hi_safety))
        self.frontier_vx_lo_safety = max(abs(self.FRONTIER_FLOOR_VX_LO), float(frontier_vx_lo_safety))
        self.frontier_vy_safety = max(self.FRONTIER_FLOOR_VY, float(frontier_vy_safety))
        self.frontier_yaw_safety = max(self.FRONTIER_FLOOR_YAW, float(frontier_yaw_safety))
        self.frontier_combo_safety = max(self.FRONTIER_FLOOR_COMBO, float(frontier_combo_safety))
        self.frontier_push_safety_n = max(0.70, float(frontier_push_safety_n))

    @staticmethod
    def _empty_slope_window() -> dict[str, float]:
        return dict(near_attempts=0.,near_successes=0.,
                    easy_attempts=0.,easy_successes=0.)

    @staticmethod
    def _empty_command_window() -> dict[str, float]:
        return dict(attempts=0.,successes=0.,
                    easy_attempts=0.,easy_successes=0.)

    @staticmethod
    def _empty_terrain_evidence() -> dict[str, float]:
        # Retain the original pooled fields for legacy non-unified runs.
        # New closed cohorts are measured against REAL physical ledge height.
        return dict(up_attempts=0., up_successes=0., up_falls=0.,
                    down_attempts=0., down_successes=0., down_falls=0.,
                    updates=0.,
                    **{f'{direction}_{band}_{outcome}':0.
                       for direction in ('up','down')
                       for band in ('near','easy')
                       for outcome in ('attempts','successes')})

    def state_dict(self) -> dict[str, Any]:
        """Everything that can affect the governor's next decision.
        Recovery/resume is a temporal rewind only if this dynamic state is
        restored alongside the network and optimizer."""
        return {
            'params': asdict(self.params),
            'competence': dict(self.competence),
            'core_competence': self.core_competence, 'best_len': self.best_len,
            'core_bad_streak': self.core_bad_streak, 'fall_ema': dict(self.fall_ema),
            '_falls_ema': dict(self._falls_ema), '_episodes_ema': dict(self._episodes_ema),
            '_updates_done_total': self._updates_done_total,
            'terrain_evidence': dict(self.terrain_evidence),
            'frontier_precision_ema': dict(self.frontier_precision_ema),
            'frontier_bad_streak': dict(self.frontier_bad_streak),
            **({'command_evidence': {k:dict(v) for k,v in
                   self.command_evidence.items()},
                 'slope_evidence': {k:dict(v) for k,v in
                   self.slope_evidence.items()}} if self.unified_capability_focus else {}),
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        expected = {
            'params', 'competence', 'core_competence', 'best_len', 'core_bad_streak',
            'fall_ema', '_falls_ema', '_episodes_ema', '_updates_done_total',
            'frontier_precision_ema', 'frontier_bad_streak',
        }
        legacy = set(expected)
        expected.add('terrain_evidence')
        if self.unified_capability_focus:
            expected.update(('command_evidence','slope_evidence'))
        # Preserve exactly-known older flat governor state (empty new
        # window), without bypassing external checkpoint source provenance.
        if not isinstance(d, dict) or set(d) not in (legacy, expected):
            raise ValueError(f"curriculum state keys mismatch: expected={sorted(expected)} got={sorted(d) if isinstance(d, dict) else type(d).__name__}")
        param_names = {f.name for f in dataclass_fields(RegimeParams)}
        if not isinstance(d['params'], dict):
            raise ValueError('curriculum params must be a dictionary')
        old_names = param_names - {
            'terrain_exposure','terrain_height_m','slope_exposure','slope_max_deg'}
        previous_names = param_names - {'slope_exposure','slope_max_deg'}
        if set(d['params']) == old_names:
            saved_params = dict(d['params'],terrain_exposure=0.0,
                terrain_height_m=0.0,slope_exposure=0.0,slope_max_deg=0.0)
        elif set(d['params']) == previous_names:
            saved_params = dict(d['params'],slope_exposure=0.0,slope_max_deg=0.0)
        elif set(d['params']) == param_names:
            saved_params = d['params']
        else:
            raise ValueError(f"curriculum parameter keys mismatch: expected={sorted(param_names)} got={sorted(d['params'])}")
        expected_regimes = set(self.REGIMES)
        expected_falls = expected_regimes | {'posture'}
        expected_frontier = {'vx_hi', 'vx_lo', 'vy', 'yaw', 'combo', 'accel'}
        for name, keys in (('competence', expected_regimes), ('fall_ema', expected_falls),
                           ('_falls_ema', expected_falls), ('_episodes_ema', expected_falls),
                           ('frontier_precision_ema', expected_frontier), ('frontier_bad_streak', expected_frontier)):
            if not isinstance(d[name], dict) or set(d[name]) != keys:
                raise ValueError(f"curriculum {name} keys mismatch")
        evidence = d.get('terrain_evidence', self._empty_terrain_evidence())
        names = set(self._empty_terrain_evidence())
        legacy_names = {'up_attempts','up_successes','up_falls',
                        'down_attempts','down_successes','down_falls','updates'}
        if isinstance(evidence,dict) and set(evidence)==legacy_names:
            # 40M-era pooled terrain episodes cannot justify promotion under
            # the new severity-specific teacher; retain their diagnostic
            # counts but introduce empty proof cohorts.
            evidence={**self._empty_terrain_evidence(),**evidence}
        if (not isinstance(evidence, dict) or set(evidence) != names or
                any(not math.isfinite(float(v)) or float(v) < 0. for v in evidence.values())):
            raise ValueError('curriculum terrain evidence window mismatch')
        if (int(evidence['updates']) != evidence['updates'] or
                evidence['updates'] > self.TERRAIN_MAX_WINDOW_UPDATES):
            raise ValueError('curriculum terrain evidence age out of range')
        for direction in ('up','down'):
            if (evidence[f'{direction}_successes'] > evidence[f'{direction}_attempts'] or
                    evidence[f'{direction}_falls'] > evidence[f'{direction}_attempts']):
                raise ValueError('curriculum terrain evidence outcome exceeds attempts')
            for band in ('near','easy'):
                if (evidence[f'{direction}_{band}_successes'] >
                        evidence[f'{direction}_{band}_attempts']):
                    raise ValueError('terrain severity successes exceed attempts')
        self.terrain_evidence = {k: float(v) for k,v in evidence.items()}
        if self.unified_capability_focus:
            entries=d['command_evidence']
            if (not isinstance(entries,dict) or
                set(entries)!=set(self.COMMAND_SECTORS)):
                raise ValueError('unified command frontier evidence schema mismatch')
            for sector,values in entries.items():
                # Preserve exact 40M-era saved near-edge evidence while
                # introducing additional easier-task failure counters.
                valid=({'attempts','successes'},
                       set(self._empty_command_window()))
                if (not isinstance(values,dict) or set(values) not in valid or
                    not all(math.isfinite(float(x)) and float(x)>=0.
                            for x in values.values()) or
                    values['successes']>values['attempts'] or
                    values.get('easy_successes',0.)>
                    values.get('easy_attempts',0.)):
                    raise ValueError(f'bad command frontier evidence {sector}')
            self.command_evidence={
                k:{**self._empty_command_window(),**v}
                for k,v in entries.items()}
            entries=d['slope_evidence']
            if (not isinstance(entries,dict) or
                set(entries)!=set(self.SLOPE_DIRECTIONS)):
                raise ValueError('unified slope evidence schema mismatch')
            for direction,values in entries.items():
                # Old 40M-era cohorts stored pooled directions. Keep their
                # field schema loadable but never promote severity using
                # evidence whose actual incline was not recorded.
                old={'attempts','successes'}
                new=set(self._empty_slope_window())
                if (not isinstance(values,dict) or set(values) not in (old,new)
                    or not all(math.isfinite(float(x)) and float(x)>=0.
                               for x in values.values()) or
                    (values['successes']>values['attempts'] if set(values)==old
                     else values['near_successes']>values['near_attempts'] or
                          values['easy_successes']>values['easy_attempts'])):
                    raise ValueError(f'bad slope evidence {direction}')
            self.slope_evidence={
                k:(self._empty_slope_window() if set(v)==old else dict(v))
                for k,v in entries.items()}
        self.params = RegimeParams(**saved_params)
        for name in ('competence', 'fall_ema', '_falls_ema', '_episodes_ema',
                     'frontier_precision_ema', 'frontier_bad_streak'):
            setattr(self, name, dict(d[name]))
        for name in ('core_competence', 'best_len', 'core_bad_streak', '_updates_done_total'):
            setattr(self, name, d[name])

    def params_dict(self) -> dict[str, float]:
        return asdict(self.params)

    def _difficulty_field(self, name: str) -> str:
        return 'posture_difficulty' if name == 'posture' else f'{name}_difficulty'

    def _decide_difficulty_move(self, current_fall_ema: float, competence: float,
                                 precision_ok: bool, kl_ok_to_expand: bool) -> tuple[str, float]:
        """The decision half of the shared proportional controller: what
        does this scalar want to do, and how strongly -- without applying
        it yet. Split from application because push moves need to be
        capped in aggregate across however many scalars want to push on
        the same update (see _apply_shared_push_budget); retreat doesn't,
        and applies immediately via _apply_retreat instead."""
        error = current_fall_ema - self.fall_target  # >0 too hard (retreat), <0 too easy (push)
        if error <= 0.0:
            mult = min(1.0, -error / self.fall_target) * self.PUSH_GAIN
            can_push = competence > self.COMPETENCE_FLOOR and precision_ok and kl_ok_to_expand
            if can_push and mult > self.MIN_ACTIONABLE_MULT:
                return 'push', mult
            elif mult > self.MIN_ACTIONABLE_MULT:
                return 'hold_push_blocked', 0.0  # would push, but competence/precision/KL vetoed it
            return 'hold_at_target', 0.0
        else:
            mult = min(1.0, error / (1.0 - self.fall_target)) * self.RETREAT_GAIN
            return ('retreat', mult) if mult > self.MIN_ACTIONABLE_MULT else ('hold_at_target', 0.0)

    def _apply_retreat(self, name: str, mult: float) -> None:
        """Applied immediately and in full -- retreat is never shared
        across scalars, unlike push. Every scalar that wants to retreat
        does so the same update, at full strength, regardless of how many
        others are also retreating. This is consistent with an asymmetry
        already built into the gains (RETREAT_GAIN > PUSH_GAIN): safety
        is never rationed, only ambition is."""
        field = self._difficulty_field(name)
        cur = getattr(self.params, field)
        setattr(self.params, field, max(0.0, cur - self.difficulty_base_step * min(mult, self.MAX_STEP_MULT)))

    def _apply_shared_push_budget(self, candidates: list[tuple[str, float]]) -> None:
        """Push candidates share one combined budget instead of each
        moving independently. Originally, nothing stopped several of the
        four scalars (three regimes plus posture) from pushing on the same
        update, each individually calibrated to what a single mover was
        allowed -- but with several able to fire at once, the combined
        change to the task distribution could be several times what any
        one update was designed to survive. A real run showed exactly this
        shape: steady, transition, and resilience difficulty climbing
        together for about 15M steps before a collapse.

        The fix: split whatever ceiling a single push could reach evenly
        across however many scalars want to push this update. Three
        regimes all maximally eager to push now move a third as far each;
        the total equals what one would have reached alone, not three
        times as much. Relative eagerness between candidates is preserved
        -- a scalar with a smaller raw mult still gets proportionally less
        of the shared budget, it's just that every share is scaled down
        by the same factor.

        Allocation (which regimes get envs next update) is untouched by
        this -- every regime still gets a live, continuous slice of data
        every update. Only how fast its difficulty can move is shared."""
        if not candidates:
            return
        n = len(candidates)
        for name, mult in candidates:
            field = self._difficulty_field(name)
            cur = getattr(self.params, field)
            shared_mult = min(mult, self.MAX_STEP_MULT) / n
            setattr(self.params, field, min(1.0, cur + self.difficulty_base_step * shared_mult))

    def _pooled_fall_ema(self, key: str, falls: float, episodes: float) -> float:
        """Smoothed fall rate as a ratio of two EMAs (falls, episodes)
        rather than an EMA of per-rollout ratios. Rollouts complete wildly
        different numbers of episodes -- a real run saw 1,344 in one and
        5 in another -- and averaging their ratios lets a 5-episode
        rollout count as much as a 1,344-episode one. Pooling counts
        weights every completed episode equally, which is what "fall rate"
        is supposed to mean. Both EMAs start at zero, so there's no
        initial bias once any data arrives."""
        a = self.FALL_EMA_ALPHA
        self._falls_ema[key] += a * (falls - self._falls_ema[key])
        self._episodes_ema[key] += a * (episodes - self._episodes_ema[key])
        if self._episodes_ema[key] > 1e-9:
            self.fall_ema[key] = self._falls_ema[key] / self._episodes_ema[key]
        return self.fall_ema[key]

    def _advance_physical_frontiers(
        self, row: dict[str, float], *,
        global_fall: float, global_retreat: bool,
        specialist_growth_enabled: bool = True,
    ) -> dict[str, str]:
        """Use REAL settled tracking evidence to revise physical command limits.

        Maintain the original axis EMAs, hysteresis and safety bounds. No
        fixed-policy exam or PPO reward feeds these training decisions.
        """
        # Growth is judged on settled tracking near the current boundary, not
        # on all rollout steps. This prevents
        # command transitions, low-amplitude centre samples and estimator noise
        # from voting a proven axis downward. A slow EMA + bad-streak hysteresis
        # keeps a single ugly rollout from eating the envelope.
        frontier_events = {}
        if (self.open_frontier and not global_retreat and
                (not self.unified_capability_focus or specialist_growth_enabled)):
            axis_specs = (() if self.unified_capability_focus else (
                ('vx_hi', 'frontier_precision_vx_forward', +1.0, self.FRONTIER_FLOOR_VX_HI, self.frontier_vx_hi_safety),
                ('vx_lo', 'frontier_precision_vx_reverse', -1.0, abs(self.FRONTIER_FLOOR_VX_LO), self.frontier_vx_lo_safety),
                ('vy', 'frontier_precision_vy', +1.0, self.FRONTIER_FLOOR_VY, self.frontier_vy_safety),
                ('yaw', 'frontier_precision_yaw', +1.0, self.FRONTIER_FLOOR_YAW, self.frontier_yaw_safety),
            ))
            easy_enough = self.fall_ema['steady'] <= 0.25 and self.competence['steady'] >= 0.60
            for axis, pkey, sign, floor_mag, safety_mag in axis_specs:
                field = f'frontier_{axis}'
                cur = float(getattr(self.params, field))
                precision = float(row.get(pkey, -1.0))
                if precision < 0.0:
                    frontier_events[axis] = 'no_edge_data'
                    continue
                prev = float(self.frontier_precision_ema.get(axis, -1.0))
                ema = precision if prev < 0.0 else ((1.0-self.FRONTIER_PRECISION_EMA_ALPHA)*prev + self.FRONTIER_PRECISION_EMA_ALPHA*precision)
                self.frontier_precision_ema[axis] = ema
                if ema < self.FRONTIER_RETREAT_PRECISION and precision < 0.35:
                    self.frontier_bad_streak[axis] = int(self.frontier_bad_streak.get(axis, 0)) + 1
                else:
                    self.frontier_bad_streak[axis] = max(0, int(self.frontier_bad_streak.get(axis, 0)) - 1)
                good = ema >= self.FRONTIER_EXPAND_PRECISION and precision >= 0.55
                sustained_bad = self.frontier_bad_streak[axis] >= self.FRONTIER_BAD_STREAK_LIMIT
                if easy_enough and good:
                    mag = min(safety_mag, max(abs(cur), floor_mag) * (1.0 + FRONTIER_GROWTH_FRAC))
                    setattr(self.params, field, sign * mag)
                    frontier_events[axis] = 'expand_edge_proven'
                elif sustained_bad:
                    mag = max(floor_mag, abs(cur) * (1.0 - FRONTIER_RETREAT_FRAC))
                    setattr(self.params, field, sign * mag)
                    frontier_events[axis] = 'retreat_sustained_tracking'
                else:
                    frontier_events[axis] = 'hold'

            cur = float(self.params.frontier_push_n)
            if self.fall_ema['resilience'] <= 0.25 and self.competence['resilience'] >= 0.48:
                self.params.frontier_push_n = min(self.frontier_push_safety_n, cur * (1.0 + (.08 if self.unified_capability_focus else FRONTIER_GROWTH_FRAC)))
                frontier_events['push_n'] = 'expand'
            elif self.fall_ema['resilience'] >= 0.42:
                self.params.frontier_push_n = max(0.70, cur * (1.0 - (.12 if self.unified_capability_focus else FRONTIER_RETREAT_FRAC)))
                frontier_events['push_n'] = 'retreat'
            else:
                frontier_events['push_n'] = 'hold'

            cur = float(self.params.frontier_sensor_scale)
            settled = min(float(row.get('precision_vx', 0.0)),
                          float(row.get('precision_vy', 0.0)),
                          float(row.get('precision_yaw', 0.0)))
            if global_fall <= 0.25 and settled >= 0.55:
                self.params.frontier_sensor_scale = min(FRONTIER_SENSOR_SAFETY, cur * (1.0 + (.015 if self.unified_capability_focus else .5*FRONTIER_GROWTH_FRAC)))
                frontier_events['sensor'] = 'expand'
            elif global_fall >= 0.50:
                self.params.frontier_sensor_scale = max(1.50, cur * (1.0 - 0.5*FRONTIER_RETREAT_FRAC))
                frontier_events['sensor'] = 'retreat'
            else:
                frontier_events['sensor'] = 'hold'

            cp = (-1.0 if self.unified_capability_focus
                  else float(row.get('frontier_precision_combo', -1.0)))
            cur = float(self.params.frontier_combo_radius)
            if cp >= 0.0:
                prev = float(self.frontier_precision_ema.get('combo', -1.0))
                ema = cp if prev < 0.0 else ((1.0-self.FRONTIER_PRECISION_EMA_ALPHA)*prev + self.FRONTIER_PRECISION_EMA_ALPHA*cp)
                self.frontier_precision_ema['combo'] = ema
                if ema < self.FRONTIER_RETREAT_PRECISION and cp < 0.35:
                    self.frontier_bad_streak['combo'] = int(self.frontier_bad_streak.get('combo', 0)) + 1
                else:
                    self.frontier_bad_streak['combo'] = max(0, int(self.frontier_bad_streak.get('combo', 0)) - 1)
                if easy_enough and ema >= 0.60 and cp >= 0.55:
                    self.params.frontier_combo_radius = min(self.frontier_combo_safety, cur * (1.0 + (.06 if self.unified_capability_focus else FRONTIER_GROWTH_FRAC)))
                    frontier_events['combo'] = 'expand_edge_proven'
                elif self.frontier_bad_streak['combo'] >= self.FRONTIER_BAD_STREAK_LIMIT:
                    self.params.frontier_combo_radius = max(self.FRONTIER_FLOOR_COMBO, cur * (1.0 - FRONTIER_RETREAT_FRAC))
                    frontier_events['combo'] = 'retreat_sustained_tracking'
                else:
                    frontier_events['combo'] = 'hold'
            else:
                frontier_events['combo'] = 'no_edge_data'

            tp = float(row.get('transition_precision', -1.0))
            if tp >= 0.0:
                prev = float(self.frontier_precision_ema.get('accel', -1.0))
                ema = tp if prev < 0.0 else ((1.0-self.FRONTIER_PRECISION_EMA_ALPHA)*prev + self.FRONTIER_PRECISION_EMA_ALPHA*tp)
                self.frontier_precision_ema['accel'] = ema
                if ema < 0.25 and tp < 0.32:
                    self.frontier_bad_streak['accel'] = int(self.frontier_bad_streak.get('accel', 0)) + 1
                else:
                    self.frontier_bad_streak['accel'] = max(0, int(self.frontier_bad_streak.get('accel', 0)) - 1)
                if self.fall_ema['transition'] <= 0.25 and self.competence['transition'] >= 0.60 and ema >= 0.55:
                    self.params.frontier_transition_hold_s = max(CONTROL_DT, self.params.frontier_transition_hold_s * (1.0 - (.025 if self.unified_capability_focus else FRONTIER_GROWTH_FRAC)))
                    self.params.frontier_ramp_steps = max(1.0, self.params.frontier_ramp_steps * (1.0 - (.05 if self.unified_capability_focus else 2.0*FRONTIER_GROWTH_FRAC)))
                    frontier_events['accel'] = 'expand_transition_proven'
                elif self.fall_ema['transition'] >= 0.50 or self.frontier_bad_streak['accel'] >= self.FRONTIER_BAD_STREAK_LIMIT:
                    self.params.frontier_transition_hold_s = min(0.90, self.params.frontier_transition_hold_s * (1.0 + FRONTIER_RETREAT_FRAC))
                    self.params.frontier_ramp_steps = min(4.0, self.params.frontier_ramp_steps * (1.0 + 2.0*FRONTIER_RETREAT_FRAC))
                    frontier_events['accel'] = 'retreat'
                else:
                    frontier_events['accel'] = 'hold'
            else:
                frontier_events['accel'] = 'no_transition_data'

        return frontier_events

    def advance_slope(self, row: dict[str,float], *, kl_ok: bool = True):
        """One shared *physical-practice* incline target, four directions.

        The near/easy governor now consumes supported, actually MOVING
        stable-episode evidence, not perfect command tracking. Full velocity
        and heading tracking remain separately reported and frozen examined.
        Missing near data never promotes; real instability on easy terrain
        still retreats. Every direction must earn its own evidence.
        """
        if not self.unified_capability_focus:
            return 'off'
        p=self.params
        if p.slope_exposure<=0.:
            p.slope_exposure=.15
            p.slope_max_deg=.70
            self.slope_evidence={k:self._empty_slope_window()
                                 for k in self.SLOPE_DIRECTIONS}
            return 'seed_real_incline_0p7deg'

        for direction in self.SLOPE_DIRECTIONS:
            window=self.slope_evidence[direction]
            for cohort,outcome in (('near','attempts'),('near','successes'),
                                   ('easy','attempts'),('easy','successes')):
                # Stability earns physical incline practice. The old
                # success key remains FULL command tracking for diagnostics
                # and independent fixed examination.
                key=(f'slope_{cohort}_stability_successes_{direction}'
                     if outcome=='successes'
                     else f'slope_{cohort}_attempts_{direction}')
                value=float(row.get(key,0.))
                if not math.isfinite(value) or value<0.:
                    raise ValueError(f'bad slope evidence {key}')
                window[f'{cohort}_{outcome}']+=value
            if (window['near_successes']>window['near_attempts'] or
                    window['easy_successes']>window['easy_attempts']):
                raise ValueError(f'bad slope evidence counts {direction}')

        windows=self.slope_evidence
        # Bad evidence on a supposedly easier challenge is the most
        # informative failure, regardless of which direction suffered.
        easy_bad=any(
            v['easy_attempts']>=self.SLOPE_MIN_EPISODES_PER_DIRECTION and
            v['easy_successes']/v['easy_attempts']<.45
            for v in windows.values())
        if easy_bad:
            p.slope_max_deg=max(.35,p.slope_max_deg*.88)
            self.slope_evidence={k:self._empty_slope_window()
                                 for k in self.SLOPE_DIRECTIONS}
            return 'retreat_easy_incline_failure'

        # A successful easier cohort is informative about retention but
        # does not promote the frontier; refresh it without clearing sparse
        # near-grade evidence, and vice versa.
        for window in windows.values():
            if window['easy_attempts']>=self.SLOPE_MIN_EPISODES_PER_DIRECTION:
                window['easy_attempts']=window['easy_successes']=0.
        ready=all(v['near_attempts']>=self.SLOPE_MIN_EPISODES_PER_DIRECTION
                  for v in windows.values())
        if not ready:
            return 'collect_real_slope'
        good=all(v['near_successes']/v['near_attempts']>=.65
                 for v in windows.values())
        for window in windows.values():
            window['near_attempts']=window['near_successes']=0.
        if not kl_ok:
            return 'hold_ppo_health'
        if good:
            p.slope_max_deg=min(15.,p.slope_max_deg+max(.25,.16*p.slope_max_deg))
            # No evidence from an older easier/severity band should be
            # recycled after the difficulty itself changes.
            self.slope_evidence={k:self._empty_slope_window()
                                 for k in self.SLOPE_DIRECTIONS}
            return 'expand_real_slope_angle'
        return 'hold_probe_learning'

    def advance_terrain(self, row: dict[str,float], *, step: int,
                        max_height_m: float, foundation_steps: int,
                        unified: bool = False, kl_ok: bool = True) -> str:
        """Physically verified up/down growth, using a checkpointed cohort.

        A single PPO rollout at 5% terrain exposure sees only a handful of
        completed episodes. Aggregate direction-conditioned outcomes across
        updates, then judge a CLOSED evidence batch; never pool down-only
        successes or spend old/easier evidence after difficulty changes.
        """
        if max_height_m <= 0. or (step < foundation_steps and not unified):
            return 'foundation_flat'
        # Specialist worlds independently probe actual supported steps from
        # their first birth. Main's non-unified bootstrap remains unchanged.
        p = self.params
        speed_lesson = unified
        exposure_min = .10 if speed_lesson else .05
        practice_max = .20 if speed_lesson else self.TERRAIN_ASCENT_PRACTICE_MAX_EXPOSURE
        expansion_max = .20 if speed_lesson else .25
        height_increment = .0005 if speed_lesson else .00025
        # Legacy terrain retains its historical core-gait prerequisites.
        # Only the unified six-cohort teacher removes cross-specialist
        # dependencies. This must not change old 220/232 training recipes.
        core_ok = self.core_competence >= .55 and self.fall_ema['posture'] <= .32
        if p.terrain_exposure <= 0.:
            if not unified and not core_ok:
                return 'await_core_walk'
            self.terrain_evidence = self._empty_terrain_evidence()
            p.terrain_exposure = .15 if speed_lesson else .05
            p.terrain_height_m = min(.004 if unified else .010,max_height_m)
            return 'terrain_unified_seed_4mm' if unified else 'terrain_seed_1cm_5pct'
        if not unified and not core_ok:
            self.terrain_evidence = self._empty_terrain_evidence()
            p.terrain_exposure = max(exposure_min,p.terrain_exposure*.90)
            p.terrain_height_m = max(min(.010,max_height_m),p.terrain_height_m*.98)
            return 'terrain_retreat_core'

        if unified:
            # One slowly wandering step-height target. Practice across 30-100%
            # of the target stays broad; only actual near-target supported
            # crossings can nudge it higher. A HARD failure is not a retreat
            # signal; repeated difficulty on easier steps is. Up and down
            # each earn their own evidence; no down-only graduation.
            evidence=self.terrain_evidence
            for direction in ('up','down'):
                for band in ('near','easy'):
                    for outcome in ('attempts','successes'):
                        key=f'terrain_{band}_{outcome}_{direction}'
                        value=float(row.get(key,0.))
                        if not math.isfinite(value) or value<0.:
                            raise ValueError(f'invalid step severity evidence {key}')
                        evidence[f'{direction}_{band}_{outcome}']+=value
                    if (evidence[f'{direction}_{band}_successes'] >
                        evidence[f'{direction}_{band}_attempts']):
                        raise ValueError(f'step successes exceed attempts {direction}/{band}')
            evidence['updates']+=1.
            min_attempts=4
            easy_bad=any(
                evidence[f'{d}_easy_attempts']>=min_attempts and
                evidence[f'{d}_easy_successes']/evidence[f'{d}_easy_attempts']<.30
                for d in ('up','down'))
            if easy_bad:
                p.terrain_height_m=max(min(.0015,max_height_m),p.terrain_height_m*.82)
                self.terrain_evidence=self._empty_terrain_evidence()
                return 'terrain_retreat_easy_failure'
            for direction in ('up','down'):
                key=f'{direction}_easy_attempts'
                if evidence[key]>=min_attempts:
                    evidence[key]=evidence[f'{direction}_easy_successes']=0.
            ready=all(evidence[f'{d}_near_attempts']>=min_attempts
                      for d in ('up','down'))
            if ready:
                good=all(
                    evidence[f'{d}_near_successes']/
                    evidence[f'{d}_near_attempts']>=.55
                    for d in ('up','down'))
                for direction in ('up','down'):
                    evidence[f'{direction}_near_attempts']=0.
                    evidence[f'{direction}_near_successes']=0.
                evidence['updates']=0.
                if not kl_ok:
                    return 'terrain_hold_ppo_health'
                if good:
                    p.terrain_height_m=min(max_height_m,p.terrain_height_m+max(.0010,.18*p.terrain_height_m))
                    self.terrain_evidence=self._empty_terrain_evidence()
                    return 'terrain_height_expand_real_cross'
                return 'terrain_hold_probe_learning'
            if evidence['updates']>=self.TERRAIN_MAX_WINDOW_UPDATES:
                self.terrain_evidence=self._empty_terrain_evidence()
                return 'terrain_window_insufficient_attempts'
            return 'terrain_collect_real_step'

        required = tuple(f'terrain_{outcome}_{direction}'
                         for direction in ('up','down')
                         for outcome in ('attempts','successes','falls'))
        if not all(key in row for key in required):
            return 'await_directional_terrain_evidence'
        window = self.terrain_evidence
        for direction in ('up','down'):
            attempts = float(row[f'terrain_attempts_{direction}'])
            successes = float(row[f'terrain_successes_{direction}'])
            falls = float(row[f'terrain_falls_{direction}'])
            if (not all(math.isfinite(v) for v in (attempts,successes,falls)) or
                    min(attempts,successes,falls)<0. or
                    successes>attempts or falls>attempts):
                raise ValueError('invalid direction-conditioned terrain outcomes')
            window[f'{direction}_attempts'] += attempts
            window[f'{direction}_successes'] += successes
            window[f'{direction}_falls'] += falls
        window['updates'] += 1.

        up_n = window['up_attempts']
        down_n = window['down_attempts']
        if min(up_n,down_n) < self.TERRAIN_MIN_ATTEMPTS_PER_DIRECTION:
            if window['updates'] >= self.TERRAIN_MAX_WINDOW_UPDATES:
                self.terrain_evidence = self._empty_terrain_evidence()
                return 'terrain_window_insufficient_attempts'
            return 'terrain_collect_directional_evidence'

        up_rate = window['up_successes'] / up_n
        down_rate = window['down_successes'] / down_n
        # No evidence can be recycled after a decision, including holds.
        self.terrain_evidence = self._empty_terrain_evidence()
        if unified and not kl_ok:
            return 'terrain_hold_ppo_health'
        if min(up_rate,down_rate) >= .50:
            if p.terrain_exposure < expansion_max:
                p.terrain_exposure = min(expansion_max,p.terrain_exposure+.005)
                return 'terrain_exposure_expand_real_cross'
            p.terrain_height_m = min(max_height_m,p.terrain_height_m+height_increment)
            return 'terrain_height_expand_real_cross'

        # Practice dose != earned obstacle height. A small genuine ascent
        # foothold and strong descent may justify at most 10% terrain births
        # while preserving >=90% normal flat lessons.
        if (up_rate >= self.TERRAIN_ASCENT_PRACTICE_MIN and
                down_rate >= self.TERRAIN_DESCENT_PRACTICE_MIN and
                p.terrain_exposure < practice_max):
            p.terrain_exposure = min(practice_max,
                                     p.terrain_exposure+.005)
            return 'terrain_exposure_practice_ascent'

        if min(up_rate,down_rate) < .12:
            p.terrain_height_m = max(min(.004 if unified else .010,max_height_m),
                                     p.terrain_height_m*.985)
            p.terrain_exposure = max(exposure_min,p.terrain_exposure*.975)
            return ('terrain_retreat_failed_ascent' if up_rate < .12
                    else 'terrain_retreat_failed_descent')
        return 'terrain_hold_directional_learning'

    def _advance_specialist_commands(self, row, *, global_retreat: bool,
                                      kl_ok: bool) -> dict[str,str]:
        """Slow, local signed-command targets with easier-task failure repair.

        Frontier success invites small growth. A hard probe failure is normal
        learning; sustained failure on easier commands invites faster retreat.
        Both sources are COMPLETED episodes in the relevant signed sector.
        """
        events={}
        for sector in self.COMMAND_SECTORS:
            evidence=self.command_evidence[sector]
            for prefix,field in (
                ('command_probe_attempts_','attempts'),
                ('command_probe_successes_','successes'),
                ('command_easy_attempts_','easy_attempts'),
                ('command_easy_successes_','easy_successes')):
                value=float(row.get(prefix+sector,0.))
                if not math.isfinite(value) or value<0.:
                    raise ValueError(f'bad command evidence {sector}/{field}')
                evidence[field]+=value
            if (evidence['successes']>evidence['attempts'] or
                evidence['easy_successes']>evidence['easy_attempts']):
                raise ValueError(f'command successes exceed attempts: {sector}')

        def near_ready(k):
            return self.command_evidence[k]['attempts']>=self.COMMAND_MIN_EDGE_EPISODES

        def easy_bad(k):
            e=self.command_evidence[k]
            return (e['easy_attempts']>=self.COMMAND_MIN_EASY_EPISODES
                    and e['easy_successes']/e['easy_attempts']<.35)

        def near_good(k):
            e=self.command_evidence[k]
            return e['successes']/e['attempts']>=.60

        def close(*names, kind='both'):
            for k in names:
                window=self.command_evidence[k]
                if kind in ('near','both'):
                    window['attempts']=window['successes']=0.
                if kind in ('easy','both'):
                    window['easy_attempts']=window['easy_successes']=0.

        safe=not global_retreat and kl_ok
        for sector,field,limit,sign,floor in (
            ('vx_hi','frontier_vx_hi',self.frontier_vx_hi_safety,+1.,.06),
            ('vx_lo','frontier_vx_lo',self.frontier_vx_lo_safety,-1.,.02),
            ('mixed','frontier_combo_radius',self.frontier_combo_safety,+1.,1.)):
            if easy_bad(sector):
                old=abs(getattr(self.params,field))
                setattr(self.params,field,sign*max(floor,old*.88))
                events[sector]='retreat_easy_failure'
                close(sector)
            elif near_ready(sector):
                if safe and near_good(sector):
                    old=abs(getattr(self.params,field))
                    factor=1.10 if sector=='mixed' else 1.15
                    setattr(self.params,field,sign*min(limit,old*factor))
                    events[sector]='expand'
                else:
                    events[sector]='hold_probe_learning'
                close(sector,kind=('both' if events[sector]=='expand' else 'near'))
            else:
                events[sector]='collect'
            # An easier cohort has its OWN window, not a side-effect of
            # the usually more frequent near-target proof window.
            easy=self.command_evidence[sector]
            if easy['easy_attempts']>=self.COMMAND_MIN_EASY_EPISODES:
                close(sector,kind='easy')

        # Lateral and yaw have symmetric targets, so both signed directions
        # should succeed for expansion. Either sign may trigger local retreat.
        for axis,floor in (('vy',.035),('yaw',.14)):
            pair=(axis+'_pos',axis+'_neg')
            field='frontier_'+axis
            if any(easy_bad(k) for k in pair):
                setattr(self.params,field,
                        max(floor,getattr(self.params,field)*.88))
                action='retreat_easy_failure'
                close(*pair)
            elif all(near_ready(k) for k in pair):
                if safe and all(near_good(k) for k in pair):
                    limit=getattr(self,field+'_safety')
                    setattr(self.params,field,min(
                        limit,getattr(self.params,field)*1.15))
                    action='expand'
                else:
                    action='hold_probe_learning'
                close(*pair,kind=('both' if action=='expand' else 'near'))
            else:
                action='collect'
            for k in pair:
                evidence=self.command_evidence[k]
                if evidence['easy_attempts']>=self.COMMAND_MIN_EASY_EPISODES:
                    close(k,kind='easy')
                events[k]=action
        return events

    def advance_curriculum(self, row: dict[str, float], *, step: int,
                           max_height_m: float, foundation_steps: int,
                           kl_ok_to_expand: bool = True) -> dict[str, float | str]:
        """One path for the ordinary and final rollout's curriculum decisions.

        Reuse the original regime logic and existing per-skill closed evidence.
        Each specialist already nudges its own target by a bounded small step,
        so there is no redundant second post-decision growth budget.
        Legacy training stays on its historical regime controller.
        """
        # Only lesson share (not competence) fades in after 20M. The
        # controller cannot seed specialist limits during cold foundation.
        growth_enabled = step >= foundation_steps
        report = self.update(
            row, kl_ok_to_expand=kl_ok_to_expand,
            specialist_growth_enabled=growth_enabled)
        report['terrain_event'] = (
            self.advance_terrain(
                row, step=step, max_height_m=max_height_m,
                foundation_steps=foundation_steps,
                unified=self.unified_capability_focus, kl_ok=kl_ok_to_expand)
            if growth_enabled or not self.unified_capability_focus
            else 'foundation_flat')
        if self.unified_capability_focus:
            report['slope_event'] = (
                self.advance_slope(row, kl_ok=kl_ok_to_expand)
                if growth_enabled else 'foundation_flat')
            # Publish the actual next-birth settings, not stale
            # pre-decision values.
            for name in (
                'steady_difficulty', 'transition_difficulty',
                'resilience_difficulty', 'posture_difficulty',
                'frontier_vx_hi', 'frontier_vx_lo', 'frontier_vy',
                'frontier_yaw', 'frontier_combo_radius', 'frontier_push_n',
                'frontier_sensor_scale', 'frontier_transition_hold_s',
                'frontier_ramp_steps'):
                report[name] = getattr(self.params, name)
            report.update(
                slope_exposure=self.params.slope_exposure,
                slope_max_deg=self.params.slope_max_deg,
                physical_slope_deg_max=row.get('physical_slope_deg_max', 0.),
                physical_slope_deg_mean=row.get('physical_slope_deg_mean', 0.),
                physical_slope_world_fraction=row.get('physical_slope_world_fraction', 0.),
                **{f'slope_{outcome}_{direction}':
                   row.get(f'slope_{outcome}_{direction}', 0.)
                   for outcome in ('attempts', 'successes')
                   for direction in self.SLOPE_DIRECTIONS})
        report.update(
            terrain_exposure=self.params.terrain_exposure,
            terrain_height_m=self.params.terrain_height_m,
            terrain_attempts=row.get('terrain_attempts', 0.),
            terrain_successes=row.get('terrain_successes', 0.),
            **{name:row.get(name, 0.) for name in (
                'terrain_attempts_up', 'terrain_attempts_down',
                'terrain_successes_up', 'terrain_successes_down',
                'terrain_falls_up', 'terrain_falls_down')})
        report.update({f'terrain_window_{key}': value
                       for key, value in self.terrain_evidence.items()})
        report.update({f'terrain_{outcome}_up_{band}':
                       row.get(f'terrain_{outcome}_up_{band}', 0.)
                       for outcome in ('attempts', 'successes', 'falls')
                       for band in ('slow', 'medium', 'fast')})
        return report

    def update(self, row: dict[str, float], *, kl_ok_to_expand: bool = True,
               specialist_growth_enabled: bool = True) -> dict[str, float | str]:
        """`row` carries PER-REGIME aggregates (fall_rate_steady,
        ep_len_mean_steady, etc, from summarize_rollout's masked-by-regime
        breakdown) alongside GLOBAL ones -- global is used for posture (a
        dial every env samples regardless of regime) and for cross-regime
        concerns below (core-neglect, stepping-collapse, KL-stress) that
        aren't specific to any one regime."""
        ep_len = float(row.get('ep_len_mean', 0.0))
        fall = float(row.get('fall_rate', 1.0))
        # Pooled GLOBAL fall rate (every completed episode, any regime) drives
        # posture and stepping-collapse detection.
        n_ep_all = float(row.get('episodes', 1.0))
        if n_ep_all > 0:
            self._pooled_fall_ema('posture', fall * n_ep_all, n_ep_all)
        global_fall = self.fall_ema['posture']
        self.best_len = max(self.best_len, ep_len)

        # Core-neglect stays global too, measured in a masked way
        # (episode_done + command-in-seed-band, regime-independent by
        # construction) -- a sustained violation retreats all FOUR scalars
        # together, not just one, since a basic-walk failure means the
        # whole policy needs to back off, not just whichever regime
        # happened to be pushing hardest.
        core_fall = row.get('core_fall_rate', -1.0)
        core_len = row.get('core_ep_len_mean', -1.0)
        if core_fall >= 0.0 and core_len >= 0.0:
            core_survival = min(1.0, core_len / 350.0) * (1.0 - min(1.0, core_fall))
            self.core_competence = float(max(0.0, min(1.0, core_survival)))
        best_competence = max(self.competence.values()) if self.competence else 0.0
        core_neglected = (self.core_competence < self.CORE_MIN_COMPETENCE
                          and best_competence >= self.COMPETENCE_SEVERE_THRESHOLD
                          and fall <= self.FALL_SEVERE_THRESHOLD)
        if core_neglected:
            self.core_bad_streak += 1
        else:
            self.core_bad_streak = 0
        global_retreat = self.core_bad_streak >= self.CORE_RETREAT_STREAK_LIMIT
        if global_retreat:
            for name in (*self.REGIMES, 'posture'):
                field = self._difficulty_field(name)
                cur = getattr(self.params, field)
                setattr(self.params, field, max(0.0, cur - self.difficulty_base_step * self.MAX_STEP_MULT))
            self.core_bad_streak = 0

        prec_vx = float(row.get('precision_vx', -1.0))
        precision_ok = prec_vx < 0.0 or prec_vx >= self.PRECISION_FLOOR
        # THREE regimes, each judged off its OWN live data every update
        # -- no round-robin, no turn, no "which one is active" selection.
        # Every regime gets real attention every update because every
        # regime got a real, continuous slice of this update's own envs
        # (environment.py's per-env regime assignment). Decision and
        # application are SPLIT (see _decide_difficulty_move/
        # _apply_retreat/_apply_shared_push_budget): retreat applies
        # immediately, every time, for every scalar that wants it, but
        # push candidates are collected first and SHARE one aggregate
        # budget, so several regimes wanting to get harder on the same
        # update can't combine into a larger change than any single mover
        # was ever meant to produce.
        events: dict[str, str] = {}
        push_candidates: list[tuple[str, float]] = []
        for name in self.REGIMES:
            # If this regime had NO real episodes complete this rollout
            # (e.g. zero envs allocated to it during the steady-only
            # bootstrap, or just unlucky timing), skip its push/retreat
            # decision and its fall_ema update entirely -- leave everything
            # exactly as it was, rather than falling back to the global
            # fall rate, which would let a different regime's performance
            # silently drive this one's difficulty.
            regime_fall = row.get(f'fall_rate_{name}', -1.0)
            has_data = regime_fall >= 0.0
            if not has_data:
                events[name] = 'no_data'
                continue
            regime_fall = float(regime_fall)
            n_ep = float(row.get(f'episodes_{name}', 1.0))
            n_fall = float(row.get(f'falls_{name}', regime_fall * n_ep))
            self._pooled_fall_ema(name, n_fall, n_ep)
            ep_len_r = float(row.get(f'ep_len_mean_{name}', ep_len))
            cone_r = float(row.get(f'foot_mid_cone_{name}', 0.0))
            vx_r = row.get(f'vx_mean_{name}', -1.0)
            cmd_vx_r = row.get(f'cmd_vx_mean_{name}', -1.0)
            track_err_r = max(0.0, 1.0 - abs(float(vx_r) - float(cmd_vx_r)) / 0.18) if vx_r >= -0.5 and cmd_vx_r >= -0.5 else 0.0
            survival_r = min(1.0, ep_len_r / 350.0) * (1.0 - min(1.0, regime_fall))
            self.competence[name] = float(max(0.0, min(1.0, 0.65 * survival_r + 0.20 * cone_r * survival_r + 0.15 * track_err_r * survival_r)))
            if global_retreat:
                events[name] = 'retreat_core_neglect_all'
                continue
            action, mult = self._decide_difficulty_move(self.fall_ema[name], self.competence[name], precision_ok, kl_ok_to_expand)
            if action == 'retreat':
                self._apply_retreat(name, mult)
                events[name] = f'retreat_{name}'
            elif action == 'push':
                push_candidates.append((name, mult))
                events[name] = f'push_{name}'  # may be downgraded below if it ends up sharing a very small budget slice
            else:
                events[name] = action

        # Posture stays a global dial, judged on the OVERALL fall rate --
        # every env samples posture variation regardless of which regime
        # it's in, so there's no regime-specific subset to isolate its own
        # performance from. Still part of the same shared push budget as
        # the three regimes -- posture pushing alongside two regimes also
        # pushing is the same aggregate-change risk either way.
        if global_retreat:
            events['posture'] = 'retreat_core_neglect_all'
        else:
            action, mult = self._decide_difficulty_move(self.fall_ema['posture'], best_competence, precision_ok, kl_ok_to_expand)
            if action == 'retreat':
                self._apply_retreat('posture', mult)
                events['posture'] = 'retreat_posture'
            elif action == 'push':
                push_candidates.append(('posture', mult))
                events['posture'] = 'push_posture'
            else:
                events['posture'] = action

        # Keep main's first 20M budget; after foundation push the four
        # difficulty scalars 2.5x faster but still share ONE growth budget.
        if self.unified_capability_focus and specialist_growth_enabled:
            base_rate=self.difficulty_base_step
            self.difficulty_base_step=base_rate*2.5
            try:
                self._apply_shared_push_budget(push_candidates)
            finally:
                self.difficulty_base_step=base_rate
        else:
            self._apply_shared_push_budget(push_candidates)
        if len(push_candidates) > 1:
            shared_note = f'push_shared_x{len(push_candidates)}'
            for name, _ in push_candidates:
                events[name] = shared_note

        # Only demonstrated training-rollout tracking may move the physical
        # limits; the fixed-policy examiner cannot teach this controller.
        frontier_events = self._advance_physical_frontiers(
            row, global_fall=global_fall, global_retreat=global_retreat,
            specialist_growth_enabled=specialist_growth_enabled)
        if self.unified_capability_focus and specialist_growth_enabled:
            frontier_events.update(self._advance_specialist_commands(
                row, global_retreat=global_retreat, kl_ok=kl_ok_to_expand))

        # Regime ALLOCATION -- what fraction of envs each regime gets
        # NEXT update. Draw weight = FLOOR + fall_ema[x], so a regime
        # currently struggling gets more env-share than one that's solid.
        # Steady-only bootstrap until a foundation gait exists.
        self._updates_done_total += 1
        if self._updates_done_total <= self.BOOTSTRAP_UPDATES:
            alloc = {'steady': 1.0, 'transition': 0.0, 'resilience': 0.0}
        else:
            weights = {r: self.ALLOCATION_FLOOR + self.fall_ema[r] for r in self.REGIMES}
            total = sum(weights.values())
            alloc = {r: w / total for r, w in weights.items()}
        self.params.regime_frac_steady = alloc['steady']
        self.params.regime_frac_transition = alloc['transition']
        self.params.regime_frac_resilience = alloc['resilience']

        report: dict[str, float | str] = {
            'competence_steady': self.competence['steady'],
            'competence_transition': self.competence['transition'],
            'competence_resilience': self.competence['resilience'],
            'core_competence': self.core_competence,
            'fall_target': float(self.fall_target),
            'best_len': self.best_len,
            'steady_difficulty': self.params.steady_difficulty,
            'transition_difficulty': self.params.transition_difficulty,
            'resilience_difficulty': self.params.resilience_difficulty,
            'posture_difficulty': self.params.posture_difficulty,
            'regime_frac_steady': alloc['steady'],
            'regime_frac_transition': alloc['transition'],
            'regime_frac_resilience': alloc['resilience'],
            'fall_ema_steady': self.fall_ema['steady'],
            'fall_ema_transition': self.fall_ema['transition'],
            'fall_ema_resilience': self.fall_ema['resilience'],
            'fall_ema_posture': self.fall_ema['posture'],
            'event_steady': events['steady'], 'event_transition': events['transition'],
            'event_resilience': events['resilience'], 'event_posture': events['posture'],
            'core_bad_streak': float(self.core_bad_streak),
            'precision_vx': prec_vx,
            'precision_vy': float(row.get('precision_vy', -1.0)),
            'precision_yaw': float(row.get('precision_yaw', -1.0)),
            'frontier_precision_vx_forward': float(row.get('frontier_precision_vx_forward', -1.0)),
            'frontier_precision_vx_reverse': float(row.get('frontier_precision_vx_reverse', -1.0)),
            'frontier_precision_vy': float(row.get('frontier_precision_vy', -1.0)),
            'frontier_precision_yaw': float(row.get('frontier_precision_yaw', -1.0)),
            'frontier_precision_ema_vx_hi': float(self.frontier_precision_ema.get('vx_hi', -1.0)),
            'frontier_precision_ema_vx_lo': float(self.frontier_precision_ema.get('vx_lo', -1.0)),
            'frontier_precision_ema_vy': float(self.frontier_precision_ema.get('vy', -1.0)),
            'frontier_precision_ema_yaw': float(self.frontier_precision_ema.get('yaw', -1.0)),
            'frontier_vx_hi': self.params.frontier_vx_hi,
            'frontier_vx_lo': self.params.frontier_vx_lo,
            'frontier_vy': self.params.frontier_vy,
            'frontier_yaw': self.params.frontier_yaw,
            'frontier_push_n': self.params.frontier_push_n,
            'frontier_sensor_scale': self.params.frontier_sensor_scale,
            'frontier_sensor_settled_precision': (min(float(row.get('precision_vx', 0.0)),
                                                       float(row.get('precision_vy', 0.0)),
                                                       float(row.get('precision_yaw', 0.0)))
                                                  if self.open_frontier else -1.0),
            'frontier_event_sensor': frontier_events.get('sensor','off'),
            'frontier_combo_radius': self.params.frontier_combo_radius,
            'precision_combo': float(row.get('precision_combo', -1.0)),
            'frontier_event_combo': frontier_events.get('combo','off'),
            'frontier_transition_hold_s': self.params.frontier_transition_hold_s,
            'frontier_ramp_steps': self.params.frontier_ramp_steps,
            'frontier_event_accel': frontier_events.get('accel','off'),
            'frontier_event_vx_hi': frontier_events.get('vx_hi','off'),
            'frontier_event_vx_lo': frontier_events.get('vx_lo','off'),
            # The specialist governor emits SIGNED pair labels; querying
            # 'vy'/'yaw' always reported off despite active evidence.
            'frontier_event_vy': frontier_events.get('vy_pos','off'),
            'frontier_event_yaw': frontier_events.get('yaw_pos','off'),
            'global_retreat': global_retreat,
        }
        if self.unified_capability_focus:
            report.update({f'command_boundary_{k}_{m}':v[m]
                           for k,v in self.command_evidence.items()
                           for m in ('attempts','successes')})
            report.update({f'focus_{k}_episodes':row.get(f'focus_episodes_{k}',0.)
                           for k in ('command','resilience','slope','step','transition','control')})
            report['control_shaft_j_per_m']=row.get('control_shaft_j_per_m',-1.)
            for direction in self.SLOPE_DIRECTIONS:
                report[f'slope_tracking_near_{direction}']=row.get(
                    f'slope_near_successes_{direction}',0.)
                report[f'slope_stability_near_{direction}']=row.get(
                    f'slope_near_stability_successes_{direction}',0.)
            report['control_episode_quality']=row.get('control_episode_quality',-1.)
        return report


# PPO health and recovery vetoes: control curriculum expansion, not PPO math.
from typing import Any
import math


class TrainingHealth:
    KL_STRESS_EXPAND_FLOOR = 0.10
    KL_STRESS_CHRONIC_WINDOW = 50
    KL_STRESS_CHRONIC_FRACTION = 0.90

    COLLAPSE_FALL_THRESHOLD = 0.65
    COLLAPSE_STREAK_LIMIT = 100

    RECOVERY_GOOD_FLOOR = 0.60
    RECOVERY_EFFORT_FLOOR = 0.20
    RECOVERY_SEEN_FLOOR = 0.10
    RECOVERY_WINDOW = 8

    def __init__(self) -> None:
        self.kl_stress_history: list[float] = []
        self.kl_stress_chronic = False
        self.stepping_collapse_streak = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            'kl_stress_history': list(self.kl_stress_history),
            'kl_stress_chronic': bool(self.kl_stress_chronic),
            'stepping_collapse_streak': int(self.stepping_collapse_streak),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = {'kl_stress_history', 'kl_stress_chronic', 'stepping_collapse_streak'}
        if not isinstance(state, dict) or set(state) != expected:
            raise ValueError(f"training-health state keys mismatch: expected={sorted(expected)} got={sorted(state) if isinstance(state, dict) else type(state).__name__}")
        hist = state['kl_stress_history']
        if not isinstance(hist, list):
            raise ValueError('training-health kl_stress_history must be a list')
        self.kl_stress_history = [float(v) for v in hist][-self.KL_STRESS_CHRONIC_WINDOW:]
        self.kl_stress_chronic = bool(state['kl_stress_chronic'])
        self.stepping_collapse_streak = int(state['stepping_collapse_streak'])

    def observe_ppo(self, row: dict[str, float]) -> dict[str, float | bool]:
        """Update the KL-stress gate for this rollout before curriculum moves."""
        kl_stopped_frac = float(row.get('ppo_kl_stopped_frac', 0.0))
        was_chronic = self.kl_stress_chronic
        self.kl_stress_history.append(kl_stopped_frac)
        self.kl_stress_history = self.kl_stress_history[-self.KL_STRESS_CHRONIC_WINDOW:]
        engaged = sum(v > self.KL_STRESS_EXPAND_FLOOR for v in self.kl_stress_history)
        self.kl_stress_chronic = (
            len(self.kl_stress_history) >= self.KL_STRESS_CHRONIC_WINDOW
            and engaged >= self.KL_STRESS_CHRONIC_FRACTION * self.KL_STRESS_CHRONIC_WINDOW
        )
        stress_now = kl_stopped_frac > self.KL_STRESS_EXPAND_FLOOR
        return {
            'kl_stopped_frac_seen': kl_stopped_frac,
            'kl_stress_chronic': float(self.kl_stress_chronic),
            'kl_ok_to_expand': (not stress_now) or self.kl_stress_chronic,
            'kl_became_chronic': self.kl_stress_chronic and not was_chronic,
        }

    def observe_collapse(self, smoothed_global_fall_rate: float) -> dict[str, float | bool]:
        """Update global collapse evidence after the curriculum's pooled fall EMA."""
        collapsed = float(smoothed_global_fall_rate) > self.COLLAPSE_FALL_THRESHOLD
        self.stepping_collapse_streak = self.stepping_collapse_streak + 1 if collapsed else 0
        return {
            'stepping_collapse_streak': float(self.stepping_collapse_streak),
            'needs_recovery': self.stepping_collapse_streak >= self.COLLAPSE_STREAK_LIMIT,
        }


    @classmethod
    def recovery_candidate(cls, competence: float, core_competence: float) -> tuple[tuple[int, float], str]:
        """Rank one sustained-competence window for the canonical rollback snapshot."""
        if competence >= cls.RECOVERY_GOOD_FLOOR and core_competence >= cls.RECOVERY_GOOD_FLOOR:
            return (2, competence), 'confirmed-good'
        if competence >= cls.RECOVERY_EFFORT_FLOOR and core_competence >= cls.RECOVERY_EFFORT_FLOOR:
            return (1, competence), 'best-effort'
        if competence >= cls.RECOVERY_SEEN_FLOOR:
            return (0, competence), 'high-water-mark'
        return (-1, -1.0), 'none'

    @staticmethod
    def adjust_learning_rate(
        scale: float, *, approx_kl: float, target_kl: float,
        factor: float, min_scale: float, max_scale: float,
    ) -> float:
        """Soft PPO KL controller, distinct from the hard minibatch KL stop.

        Use only the KL of the collected TRAINING rollout, never held-out
        exam scores. Non-finite KL preserves the prior scale; an in-band
        update also leaves it alone. The clamp applies only to finite KL,
        matching the established controller's timing and arithmetic.
        """
        if math.isfinite(approx_kl):
            if approx_kl > 2.0 * target_kl:
                scale /= factor
            elif approx_kl < 0.5 * target_kl:
                scale *= factor
            scale = float(min(max_scale, max(min_scale, scale)))
        return scale

    def note_recovery_performed(self) -> None:
        self.stepping_collapse_streak = 0

# Public compatibility imports: existing training, exams and archived
# checkpoint loader code may still import these names from emu.curriculum.
# Canonical implementations live in two cohesive modules below.
from .capability_field import (
    FIELD_VERSION,
    FIELD_BOUNDS,
    RECOVERY_BINS,
    DEFAULT_HALF_LIFE_STEPS,
    RECOVERY_MIN_AGE_S,
    SLOW_ALPHA,
    MIN_EFFECTIVE_EVIDENCE,
    STRONG_EFFECTIVE_EVIDENCE,
    NODE_MASTER_LOWER,
    RECOVERY_MASTER_LOWER,
    EDGE_MASTER_LOWER,
    STALE_HALF_LIVES,
    NODE_ANCHORS,
    EDGE_ANCHORS,
    EDGE_DIRS,
    EDGE_SCALES,
    CapabilityField,
    _posterior_mean,
    _wilson_lower,
)
from .capability_cloud import (
    FAMILIES,
    CMD_SCALE,
    halton,
    pnorm,
    _direction,
    command_from_normalized,
    command_sector,
    make_probe,
    make_panel,
    make_calibration_panel,
    make_common_recovery_panel,
    disturbance_overrides,
    physical_descriptor,
    base_episode,
    _not_before_for_spec,
    _row_from_times,
    evaluate_specs_batch,
    annotate_matched_controls,
    _bin,
    summarize_rows,
    record_capability_cloud_exam,
)
