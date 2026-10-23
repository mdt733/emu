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
                 frontier_push_safety_n: float = 3.0):
        self.params = RegimeParams()
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
    def _empty_terrain_evidence() -> dict[str, float]:
        return dict(up_attempts=0., up_successes=0., up_falls=0.,
                    down_attempts=0., down_successes=0., down_falls=0.,
                    updates=0.)

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
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        expected = {
            'params', 'competence', 'core_competence', 'best_len', 'core_bad_streak',
            'fall_ema', '_falls_ema', '_episodes_ema', '_updates_done_total',
            'frontier_precision_ema', 'frontier_bad_streak',
        }
        legacy = set(expected)
        expected.add('terrain_evidence')
        # Preserve exactly-known older flat governor state (empty new
        # window), without bypassing external checkpoint source provenance.
        if not isinstance(d, dict) or set(d) not in (legacy, expected):
            raise ValueError(f"curriculum state keys mismatch: expected={sorted(expected)} got={sorted(d) if isinstance(d, dict) else type(d).__name__}")
        param_names = {f.name for f in dataclass_fields(RegimeParams)}
        if not isinstance(d['params'], dict):
            raise ValueError('curriculum params must be a dictionary')
        legacy_names = param_names - {'terrain_exposure', 'terrain_height_m'}
        if set(d['params']) == legacy_names:
            # Strictly recognise the OLD complete schema, not an arbitrary
            # missing-field packet. Historical flat checkpoints stay resumable.
            saved_params = dict(d['params'], terrain_exposure=0.0, terrain_height_m=0.0)
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
        self.terrain_evidence = {k: float(v) for k,v in evidence.items()}
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
        if self.open_frontier and not global_retreat:
            axis_specs = (
                ('vx_hi', 'frontier_precision_vx_forward', +1.0, self.FRONTIER_FLOOR_VX_HI, self.frontier_vx_hi_safety),
                ('vx_lo', 'frontier_precision_vx_reverse', -1.0, abs(self.FRONTIER_FLOOR_VX_LO), self.frontier_vx_lo_safety),
                ('vy', 'frontier_precision_vy', +1.0, self.FRONTIER_FLOOR_VY, self.frontier_vy_safety),
                ('yaw', 'frontier_precision_yaw', +1.0, self.FRONTIER_FLOOR_YAW, self.frontier_yaw_safety),
            )
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
            if self.fall_ema['resilience'] <= 0.22 and self.competence['resilience'] >= 0.60:
                self.params.frontier_push_n = min(self.frontier_push_safety_n, cur * (1.0 + FRONTIER_GROWTH_FRAC))
                frontier_events['push_n'] = 'expand'
            elif self.fall_ema['resilience'] >= 0.45:
                self.params.frontier_push_n = max(0.70, cur * (1.0 - FRONTIER_RETREAT_FRAC))
                frontier_events['push_n'] = 'retreat'
            else:
                frontier_events['push_n'] = 'hold'

            cur = float(self.params.frontier_sensor_scale)
            settled = min(float(row.get('precision_vx', 0.0)),
                          float(row.get('precision_vy', 0.0)),
                          float(row.get('precision_yaw', 0.0)))
            if global_fall <= 0.25 and settled >= 0.55:
                self.params.frontier_sensor_scale = min(FRONTIER_SENSOR_SAFETY, cur * (1.0 + 0.5*FRONTIER_GROWTH_FRAC))
                frontier_events['sensor'] = 'expand'
            elif global_fall >= 0.50:
                self.params.frontier_sensor_scale = max(1.50, cur * (1.0 - 0.5*FRONTIER_RETREAT_FRAC))
                frontier_events['sensor'] = 'retreat'
            else:
                frontier_events['sensor'] = 'hold'

            cp = float(row.get('frontier_precision_combo', -1.0))
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
                    self.params.frontier_combo_radius = min(self.frontier_combo_safety, cur * (1.0 + FRONTIER_GROWTH_FRAC))
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
                    self.params.frontier_transition_hold_s = max(CONTROL_DT, self.params.frontier_transition_hold_s * (1.0 - FRONTIER_GROWTH_FRAC))
                    self.params.frontier_ramp_steps = max(1.0, self.params.frontier_ramp_steps * (1.0 - 2.0*FRONTIER_GROWTH_FRAC))
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

    def advance_terrain(self, row: dict[str,float], *, step: int,
                        max_height_m: float, foundation_steps: int) -> str:
        """Physically verified up/down growth, using a checkpointed cohort.

        A single PPO rollout at 5% terrain exposure sees only a handful of
        completed episodes. Aggregate direction-conditioned outcomes across
        updates, then judge a CLOSED evidence batch; never pool down-only
        successes or spend old/easier evidence after difficulty changes.
        """
        if max_height_m <= 0. or step < foundation_steps:
            return 'foundation_flat'
        p = self.params
        core_ok = self.core_competence >= .55 and self.fall_ema['posture'] <= .32
        if p.terrain_exposure <= 0.:
            if not core_ok:
                return 'await_core_walk'
            self.terrain_evidence = self._empty_terrain_evidence()
            p.terrain_exposure = .05
            p.terrain_height_m = min(.010,max_height_m)
            return 'terrain_seed_1cm_5pct'
        if not core_ok:
            self.terrain_evidence = self._empty_terrain_evidence()
            p.terrain_exposure = max(.05,p.terrain_exposure*.90)
            p.terrain_height_m = max(min(.010,max_height_m),p.terrain_height_m*.98)
            return 'terrain_retreat_core'

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
        if min(up_rate,down_rate) >= .50:
            if p.terrain_exposure < .25:
                p.terrain_exposure = min(.25,p.terrain_exposure+.005)
                return 'terrain_exposure_expand_real_cross'
            p.terrain_height_m = min(max_height_m,p.terrain_height_m+.00025)
            return 'terrain_height_expand_real_cross'

        # Practice dose != earned obstacle height. A small genuine ascent
        # foothold and strong descent may justify at most 10% terrain births
        # while preserving >=90% normal flat lessons.
        if (up_rate >= self.TERRAIN_ASCENT_PRACTICE_MIN and
                down_rate >= self.TERRAIN_DESCENT_PRACTICE_MIN and
                p.terrain_exposure < self.TERRAIN_ASCENT_PRACTICE_MAX_EXPOSURE):
            p.terrain_exposure = min(self.TERRAIN_ASCENT_PRACTICE_MAX_EXPOSURE,
                                     p.terrain_exposure+.005)
            return 'terrain_exposure_practice_ascent'

        if min(up_rate,down_rate) < .12:
            p.terrain_height_m = max(min(.010,max_height_m),
                                     p.terrain_height_m*.985)
            p.terrain_exposure = max(.05,p.terrain_exposure*.975)
            return ('terrain_retreat_failed_ascent' if up_rate < .12
                    else 'terrain_retreat_failed_descent')
        return 'terrain_hold_directional_learning'

    def update(self, row: dict[str, float], *, kl_ok_to_expand: bool = True) -> dict[str, float | str]:
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

        self._apply_shared_push_budget(push_candidates)
        if len(push_candidates) > 1:
            shared_note = f'push_shared_x{len(push_candidates)}'
            for name, _ in push_candidates:
                events[name] = shared_note

        # Only demonstrated training-rollout tracking may move the physical
        # limits; the fixed-policy examiner cannot teach this controller.
        frontier_events = self._advance_physical_frontiers(
            row, global_fall=global_fall, global_retreat=global_retreat)

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
            'frontier_event_vy': frontier_events.get('vy','off'),
            'frontier_event_yaw': frontier_events.get('yaw','off'),
            'global_retreat': global_retreat,
        }
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

# Capability map: live learning evidence and time-window confidence.
#!/usr/bin/env python3
"""Continuous, event-based capability field for EMU.

The field keeps compact command geometry and semantic event-based evidence:

* NODE competence is a decayed probability of succeeding at a meaningful,
  sustained command window.  Continuous tracking/upright/support quality is
  recorded separately and never masquerades as a Bernoulli success.
* RECOVERY competence is one verdict per integrated resilience birth: did the
  policy return to competent locomotion and sustain it, or fail first?
* EDGE competence is one verdict per complete command maneuver: did the robot
  survive the requested transition and settle onto the new command?

The field is deliberately host-side.  PPO already transfers its rollout metric
pytree to the host once per update; this module reuses that transfer and never
adds per-death callbacks or an extra simulator pass.  Held-out evaluator
results remain quarantined and NEVER update this object.

Field schema 6 requires gait-window mean tracking error, treats one-frame
maneuvers as first-class events, and persists only evidence modern teaching or
visualization actually consumes. Incompatible maps are rejected on resume.
"""
import json
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from .contracts import CONTROL_DT, FIELD_SHAPE, FIELD_NODE_ANCHORS, FIELD_EDGE_ANCHORS, FIELD_EDGE_SCALES
from .contracts import (VX_ABS_TOL,VX_REL_TOL,VY_ABS_TOL,VY_REL_TOL,YAW_ABS_TOL,YAW_REL_TOL,UPRIGHT_GRAVITY_Z_MIN,SUPPORT_MIN,NODE_WINDOW_S,NODE_SETTLE_DELAY_S,NODE_STRUCTURAL_FRACTION_MIN,RECOVERY_STABLE_S,EDGE_STABLE_S)

FIELD_VERSION = 6

FIELD_BOUNDS = ((-0.65, 1.20), (-0.65, 0.65), (-5.0, 5.0))
RECOVERY_BINS = 6
DEFAULT_HALF_LIFE_STEPS = 20_000_000.0
RECOVERY_MIN_AGE_S = 0.16
SLOW_ALPHA = 0.015
MIN_EFFECTIVE_EVIDENCE = 3.0
STRONG_EFFECTIVE_EVIDENCE = 8.0
NODE_MASTER_LOWER = 0.80
RECOVERY_MASTER_LOWER = 0.70
EDGE_MASTER_LOWER = 0.70
STALE_HALF_LIVES = 1.0
NODE_ANCHORS = FIELD_NODE_ANCHORS
EDGE_ANCHORS = FIELD_EDGE_ANCHORS

# Twenty-six local directions on the 3-D command lattice.  These are direction
# sectors, not literal requirements that a maneuver end in an adjacent voxel.
EDGE_DIRS = np.asarray(
    [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)
     if not (x == 0 and y == 0 and z == 0)], dtype=np.float32)
EDGE_DIRS /= np.maximum(np.linalg.norm(EDGE_DIRS, axis=1, keepdims=True), 1e-6)

# Command normalization used only for maneuver direction/rate geometry.
EDGE_SCALES = np.asarray(FIELD_EDGE_SCALES, np.float32)



def _posterior_mean(success: np.ndarray, failure: np.ndarray) -> np.ndarray:
    """Beta(1,1) posterior mean from decayed event evidence."""
    return (1.0 + success) / np.maximum(2.0 + success + failure, 1e-6)


def _wilson_lower(success: np.ndarray, failure: np.ndarray, z: float = 1.2815515655) -> np.ndarray:
    """One-sided ~90% Wilson lower bound for possibly fractional counts.

    Exponential forgetting makes the effective counts fractional.  Wilson's
    algebra remains useful as a conservative confidence score even though the
    counts are effective evidence rather than IID integer observations.
    """
    s = np.asarray(success, np.float32)
    f = np.asarray(failure, np.float32)
    n = s + f
    p = np.divide(s, np.maximum(n, 1e-6))
    z2 = float(z * z)
    den = 1.0 + z2 / np.maximum(n, 1e-6)
    centre = p + z2 / (2.0 * np.maximum(n, 1e-6))
    rad = z * np.sqrt(np.maximum(0.0, p * (1.0 - p) / np.maximum(n, 1e-6)
                                  + z2 / (4.0 * np.maximum(n, 1e-6) ** 2)))
    out = (centre - rad) / den
    return np.where(n > 0.0, np.clip(out, 0.0, 1.0), 0.0).astype(np.float32)


class CapabilityField:
    """Dense-storage, low-resolution self-map with event-based evidence."""

    def __init__(self, half_life_steps: float = DEFAULT_HALF_LIFE_STEPS):
        self.shape = FIELD_SHAPE
        self.bounds = FIELD_BOUNDS
        self.half_life_steps = float(half_life_steps)
        self.centers = tuple(np.linspace(lo, hi, n, dtype=np.float32)
                             for (lo, hi), n in zip(self.bounds, self.shape))
        sh = self.shape

        # Clean command nodes: success/failure probability + separate quality.
        self.node_success = np.zeros(sh, np.float32)
        self.node_failure = np.zeros(sh, np.float32)
        self.node_quality_sum = np.zeros(sh, np.float32)
        self.node_slow = np.full(sh, 0.5, np.float32)
        self.node_last_step = np.full(sh, -1, np.int64)

        # Integrated recovery layer, indexed by command node + severity band.
        rsh = sh + (RECOVERY_BINS,)
        self.rec_success = np.zeros(rsh, np.float32)
        self.rec_failure = np.zeros(rsh, np.float32)

        # Maneuver graph edges: source node + one of 26 command directions.
        esh = sh + (len(EDGE_DIRS),)
        self.edge_success = np.zeros(esh, np.float32)
        self.edge_failure = np.zeros(esh, np.float32)
        self.edge_rate_sum = np.zeros(esh, np.float32)
        self.edge_slow = np.full(esh, 0.5, np.float32)

        self.total_updates = 0
        self.total_node_events = 0
        self.total_recovery_events = 0
        self.total_edge_events = 0
        self.last_step = 0
        self.last_decay = 1.0

        # Runtime event trackers are intentionally NOT serialized.  Environment
        # physics/episode buffers also restart fresh on process resume, so a
        # half-finished one-second window cannot be continued honestly anyway.
        self._reset_runtime(0)

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    def _axis_index(self, x: np.ndarray, axis: int) -> np.ndarray:
        lo, hi = self.bounds[axis]
        n = self.shape[axis]
        u = (np.asarray(x, np.float32) - lo) / max(hi - lo, 1e-6)
        return np.clip(np.rint(u * (n - 1)), 0, n - 1).astype(np.int32)

    def command_indices(self, vx, vy, yaw):
        return self._axis_index(vx, 0), self._axis_index(vy, 1), self._axis_index(yaw, 2)

    def command_for_index(self, ix: int, iy: int, iz: int) -> np.ndarray:
        return np.asarray([self.centers[0][ix], self.centers[1][iy], self.centers[2][iz]], np.float32)

    def _flat_index(self, ix, iy, iz):
        return np.ravel_multi_index((ix, iy, iz), self.shape)

    # ------------------------------------------------------------------
    # Runtime event trackers
    # ------------------------------------------------------------------
    def _reset_runtime(self, n_envs: int) -> None:
        """One birth/reset layout for all in-progress semantic event windows.

        Runtime windows follow MJX world's live env count, never checkpoint
        state. Restart or changed batch size intentionally discards unfinished
        node/recovery/maneuver episodes, not the accumulated capability map.
        """
        n_envs = int(n_envs)
        self._runtime_n = n_envs
        self._node_active = np.zeros(n_envs, bool)
        self._node_flat = np.zeros(n_envs, np.int64)
        self._node_frames = np.zeros(n_envs, np.int32)
        self._node_structural_good_frames = np.zeros(n_envs, np.int32)
        self._node_err_sum_rt = np.zeros((n_envs, 3), np.float32)
        self._node_tol_sum_rt = np.zeros((n_envs, 3), np.float32)
        self._node_quality_sum_rt = np.zeros(n_envs, np.float32)

        self._rec_active = np.zeros(n_envs, bool)
        self._rec_flat = np.zeros(n_envs, np.int64)
        self._rec_bin = np.zeros(n_envs, np.int32)
        self._rec_streak = np.zeros(n_envs, np.int32)

        self._edge_phase = np.zeros(n_envs, np.int8)
        self._edge_source = np.zeros((n_envs, 3), np.float32)
        self._edge_target = np.zeros((n_envs, 3), np.float32)
        self._edge_source_flat = np.zeros(n_envs, np.int64)
        self._edge_rate_cmd = np.zeros(n_envs, np.float32)
        self._edge_success_streak = np.zeros(n_envs, np.int32)

        self._prev_cmd = np.zeros((n_envs, 3), np.float32)
        self._have_prev = np.zeros(n_envs, bool)

    def _ensure_runtime(self, n_envs: int) -> None:
        if self._runtime_n != int(n_envs):
            self._reset_runtime(int(n_envs))

    # ------------------------------------------------------------------
    # Exponential forgetting
    # ------------------------------------------------------------------
    def decay_to(self, step: int) -> float:
        step = int(step)
        delta = max(0, step - int(self.last_step))
        if delta <= 0 or self.half_life_steps <= 0:
            self.last_step = step
            self.last_decay = 1.0
            return 1.0
        d = float(2.0 ** (-delta / float(self.half_life_steps)))
        for arr in (
            self.node_success, self.node_failure, self.node_quality_sum,
            self.rec_success, self.rec_failure,
            self.edge_success, self.edge_failure, self.edge_rate_sum,
        ):
            arr *= d
        self.last_step = step
        self.last_decay = d
        return d

    # ------------------------------------------------------------------
    # Basic estimates
    # ------------------------------------------------------------------
    def node_competence(self) -> np.ndarray:
        return _posterior_mean(self.node_success, self.node_failure)

    def node_lower(self) -> np.ndarray:
        return _wilson_lower(self.node_success, self.node_failure)

    def node_evidence(self) -> np.ndarray:
        return self.node_success + self.node_failure

    def node_quality(self) -> np.ndarray:
        return self.node_quality_sum / np.maximum(self.node_evidence(), 1e-6)

    def node_confidence(self) -> np.ndarray:
        return (1.0 - np.exp(-self.node_evidence() / max(STRONG_EFFECTIVE_EVIDENCE, 1e-6))).astype(np.float32)

    def recovery_competence(self) -> np.ndarray:
        return _posterior_mean(self.rec_success, self.rec_failure)

    def recovery_lower(self) -> np.ndarray:
        return _wilson_lower(self.rec_success, self.rec_failure)

    def recovery_evidence(self) -> np.ndarray:
        return self.rec_success + self.rec_failure

    def edge_competence(self) -> np.ndarray:
        return _posterior_mean(self.edge_success, self.edge_failure)

    def edge_lower(self) -> np.ndarray:
        return _wilson_lower(self.edge_success, self.edge_failure)

    def edge_evidence(self) -> np.ndarray:
        return self.edge_success + self.edge_failure

    def edge_rate(self) -> np.ndarray:
        return self.edge_rate_sum / np.maximum(self.edge_success, 1e-6)

    # ------------------------------------------------------------------
    # Event accumulation helpers
    # ------------------------------------------------------------------
    def _add_node_events(self, flats: list[np.ndarray], successes: list[np.ndarray], qualities: list[np.ndarray], step: int) -> int:
        if not flats:
            return 0
        flat = np.concatenate(flats).astype(np.int64, copy=False)
        suc = np.concatenate(successes).astype(np.float32, copy=False)
        qua = np.concatenate(qualities).astype(np.float32, copy=False)
        if flat.size == 0:
            return 0
        size = int(np.prod(self.shape))
        self.node_success += np.bincount(flat, weights=suc, minlength=size).reshape(self.shape).astype(np.float32)
        self.node_failure += np.bincount(flat, weights=1.0-suc, minlength=size).reshape(self.shape).astype(np.float32)
        self.node_quality_sum += np.bincount(flat, weights=qua, minlength=size).reshape(self.shape).astype(np.float32)
        touched = np.unique(flat)
        self.node_last_step.ravel()[touched] = int(step)
        self.total_node_events += int(flat.size)
        return int(flat.size)

    def _add_recovery_events(self, flats: list[np.ndarray], bins: list[np.ndarray], successes: list[np.ndarray]) -> int:
        if not flats:
            return 0
        flat = np.concatenate(flats).astype(np.int64, copy=False)
        rb = np.concatenate(bins).astype(np.int64, copy=False)
        suc = np.concatenate(successes).astype(np.float32, copy=False)
        if flat.size == 0:
            return 0
        rflat = flat * RECOVERY_BINS + rb
        rsize = int(np.prod(self.shape)) * RECOVERY_BINS
        shape = self.rec_success.shape
        self.rec_success += np.bincount(rflat, weights=suc, minlength=rsize).reshape(shape).astype(np.float32)
        self.rec_failure += np.bincount(rflat, weights=1.0-suc, minlength=rsize).reshape(shape).astype(np.float32)
        self.total_recovery_events += int(flat.size)
        return int(flat.size)

    def _edge_sector_and_rate(self, source: np.ndarray, target: np.ndarray, rate_cmd: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        delta_n = (target - source) / EDGE_SCALES[None, :]
        mag = np.linalg.norm(delta_n, axis=1)
        unit = delta_n / np.maximum(mag[:, None], 1e-6)
        sector = np.argmax(unit @ EDGE_DIRS.T, axis=1).astype(np.int32)
        rate = np.where(rate_cmd > 1e-6, rate_cmd, mag).astype(np.float32)
        return sector, rate, mag

    def _add_edge_events(self, source_flats: list[np.ndarray], sources: list[np.ndarray], targets: list[np.ndarray],
                         rate_cmds: list[np.ndarray], successes: list[np.ndarray]) -> int:
        if not source_flats:
            return 0
        sf = np.concatenate(source_flats).astype(np.int64, copy=False)
        src = np.concatenate(sources).astype(np.float32, copy=False)
        tgt = np.concatenate(targets).astype(np.float32, copy=False)
        rc = np.concatenate(rate_cmds).astype(np.float32, copy=False)
        suc = np.concatenate(successes).astype(np.float32, copy=False)
        if sf.size == 0:
            return 0
        sec, rate, mag = self._edge_sector_and_rate(src, tgt, rc)
        keep = mag >= 0.04  # ignore tiny command jitter / same-cell resamples
        if not np.any(keep):
            return 0
        sf, sec, rate, suc = (x[keep] for x in (sf, sec, rate, suc))
        eflat = sf * len(EDGE_DIRS) + sec.astype(np.int64)
        esize = int(np.prod(self.shape)) * len(EDGE_DIRS)
        shape = self.edge_success.shape
        self.edge_success += np.bincount(eflat, weights=suc, minlength=esize).reshape(shape).astype(np.float32)
        self.edge_failure += np.bincount(eflat, weights=1.0-suc, minlength=esize).reshape(shape).astype(np.float32)
        # Maneuver-rate frontier is meaningful only for successful edges.
        self.edge_rate_sum += np.bincount(eflat, weights=rate*suc, minlength=esize).reshape(shape).astype(np.float32)
        self.total_edge_events += int(eflat.size)
        return int(eflat.size)

    # ------------------------------------------------------------------
    # Main rollout ingestion
    # ------------------------------------------------------------------
    def _cancel_event_windows(self, ids: np.ndarray) -> None:
        """Discard unfinished node/recovery/edge evidence at a boundary.

        Called when a new episode starts or a field-taught frame begins.
        Neither event can complete an unbiased window from the previous
        frames. This never removes accumulated capability evidence.
        """
        if not len(ids):
            return
        self._node_active[ids] = False
        self._node_frames[ids] = 0
        self._node_structural_good_frames[ids] = 0
        self._node_err_sum_rt[ids] = 0.0
        self._node_tol_sum_rt[ids] = 0.0
        self._node_quality_sum_rt[ids] = 0.0
        self._rec_active[ids] = False
        self._rec_streak[ids] = 0
        self._edge_phase[ids] = 0
        self._edge_success_streak[ids] = 0
        self._have_prev[ids] = False

    def update_from_rollout(self, metrics: dict[str, Any], *, step: int, control_dt: float = CONTROL_DT,
                            exclude_field_taught: bool = False) -> dict[str, float]:
        """Convert one PPO rollout into semantic node/recovery/maneuver events.

        ``exclude_field_taught`` is used only by the clean-evidence tournament arm:
        experience deliberately selected by the field still trains PPO, but cannot
        reinforce the field that selected it.  Untaught governor experience remains
        normal evidence.  Missing metrics are a broken trainer contract, not
        permission to continue with an inert self-map.
        """
        self.decay_to(step)
        needed = ('cmd_vx','cmd_vy','cmd_yaw','vx','vy','yaw_rate','gravity_z','support',
                  'bad_contact','episode_fell','episode_done','age_s','transition_frac','cmd_resampled',
                  'regime_resilience','birth_resilience_d','target_cmd_vx','target_cmd_vy','target_cmd_yaw',
                  'transition_ramp_speed')
        missing = [k for k in needed if k not in metrics]
        if missing:
            raise KeyError('capability field missing required rollout metrics: ' + ', '.join(missing))

        def arr(name): return np.asarray(metrics[name], np.float32)
        cvx, cvy, cyaw = arr('cmd_vx'), arr('cmd_vy'), arr('cmd_yaw')
        if cvx.ndim == 1: cvx, cvy, cyaw = cvx[None,:], cvy[None,:], cyaw[None,:]
        T,N=cvx.shape; self._ensure_runtime(N)
        vx,vy,yaw=arr('vx'),arr('vy'),arr('yaw_rate')
        gz,support,bad=arr('gravity_z'),arr('support'),arr('bad_contact')
        fell,done,age=arr('episode_fell'),arr('episode_done'),arr('age_s')
        ramp=arr('transition_frac')>.5; cmd_resampled=arr('cmd_resampled')>.5
        resilience=arr('regime_resilience')>.5; sev=np.clip(arr('birth_resilience_d'),0.0,.999999)
        tgt_vx,tgt_vy,tgt_yaw=arr('target_cmd_vx'),arr('target_cmd_vy'),arr('target_cmd_yaw')
        ramp_speed=arr('transition_ramp_speed')
        if exclude_field_taught:
            if 'field_taught' not in metrics:
                raise KeyError('capability field clean-evidence mode requires field_taught rollout metric')
            taught=arr('field_taught')>.5
        else:
            taught=np.zeros_like(cvx,dtype=bool)

        vx_tol=np.maximum(VX_ABS_TOL,VX_REL_TOL*np.abs(cvx)); vy_tol=np.maximum(VY_ABS_TOL,VY_REL_TOL*np.abs(cvy)); yaw_tol=np.maximum(YAW_ABS_TOL,YAW_REL_TOL*np.abs(cyaw))
        within=(np.abs(vx-cvx)<=vx_tol)&(np.abs(vy-cvy)<=vy_tol)&(np.abs(yaw-cyaw)<=yaw_tol)
        attitude_ok=(arr('structural_attitude_ok')>.5 if 'structural_attitude_ok' in metrics
                     else gz>UPRIGHT_GRAVITY_Z_MIN)
        structural_ok=attitude_ok&(support>SUPPORT_MIN)&(bad<.5)&(fell<.5)
        core_ok=within&structural_ok
        e1=np.abs(vx-cvx)/np.maximum(vx_tol,1e-6); e2=np.abs(vy-cvy)/np.maximum(vy_tol,1e-6); e3=np.abs(yaw-cyaw)/np.maximum(yaw_tol,1e-6)
        tracking_q=np.exp(-.5*(e1*e1+e2*e2+e3*e3)/3.0)
        upright_q=(np.clip(arr('attitude_quality'),0,1) if 'attitude_quality' in metrics
                   else np.clip((gz-.72)/(1-.72),0,1)); support_q=np.clip(support/.35,0,1)
        quality=np.clip(tracking_q*(.65+.20*upright_q+.15*support_q),0,1).astype(np.float32)
        ix,iy,iz=self.command_indices(cvx,cvy,cyaw); flat_all=self._flat_index(ix.ravel(),iy.ravel(),iz.ravel()).reshape((T,N))

        node_flats=[]; node_succ=[]; node_qual=[]; rec_flats=[]; rec_bins=[]; rec_succ=[]
        edge_sf=[]; edge_src=[]; edge_tgt=[]; edge_rate=[]; edge_succ=[]
        node_target=max(1,int(round(float(NODE_WINDOW_S)/max(control_dt,1e-6))))
        rec_target=max(1,int(round(float(RECOVERY_STABLE_S)/max(control_dt,1e-6))))
        edge_target=max(1,int(round(float(EDGE_STABLE_S)/max(control_dt,1e-6))))

        def emit_edge(mask,success):
            ids=np.flatnonzero(mask)
            if not ids.size: return
            edge_sf.append(self._edge_source_flat[ids].copy()); edge_src.append(self._edge_source[ids].copy()); edge_tgt.append(self._edge_target[ids].copy())
            edge_rate.append(self._edge_rate_cmd[ids].copy()); edge_succ.append(np.full(ids.size,success,np.float32))
            self._edge_phase[ids]=0; self._edge_success_streak[ids]=0

        for t in range(T):
            cmd_t=np.stack([cvx[t],cvy[t],cyaw[t]],axis=1).astype(np.float32); target_t=np.stack([tgt_vx[t],tgt_vy[t],tgt_yaw[t]],axis=1).astype(np.float32)
            blocked=taught[t]
            if np.any(blocked):
                # Break every semantic event machine at the boundary of
                # field-selected practice.  Otherwise an event could begin on
                # unbiased experience and finish inside the self-selected region.
                self._cancel_event_windows(np.flatnonzero(blocked))
            new_ep=age[t] <= (1.5*float(control_dt)+1e-6)
            node_fall=(fell[t]>.5)&(~resilience[t])&(~ramp[t])&(~blocked)&(age[t]>=float(NODE_SETTLE_DELAY_S))
            ids=np.flatnonzero(node_fall)
            if ids.size:
                ff=np.where(self._node_active[ids],self._node_flat[ids],flat_all[t,ids]); node_flats.append(ff.copy()); node_succ.append(np.zeros(ids.size,np.float32))
                q=np.where(self._node_active[ids],np.divide(self._node_quality_sum_rt[ids],np.maximum(self._node_frames[ids],1)),quality[t,ids]); node_qual.append(np.asarray(q,np.float32))
            if np.any(new_ep):
                ids=np.flatnonzero(new_ep)
                self._cancel_event_windows(ids)
                rids=ids[resilience[t,ids]&(~blocked[ids])]
                if rids.size:
                    self._rec_active[rids]=True; self._rec_flat[rids]=flat_all[t,rids]; self._rec_bin[rids]=np.clip(np.floor(sev[t,rids]*RECOVERY_BINS),0,RECOVERY_BINS-1).astype(np.int32)

            eligible=self._rec_active&(~blocked)&(age[t]>=float(RECOVERY_MIN_AGE_S)); self._rec_streak[eligible&core_ok[t]]+=1; self._rec_streak[eligible&(~core_ok[t])]=0
            ids=np.flatnonzero(self._rec_active&(self._rec_streak>=rec_target))
            if ids.size:
                rec_flats.append(self._rec_flat[ids].copy()); rec_bins.append(self._rec_bin[ids].copy()); rec_succ.append(np.ones(ids.size,np.float32)); self._rec_active[ids]=False; self._rec_streak[ids]=0
            ids=np.flatnonzero(self._rec_active&(done[t]>.5))
            if ids.size:
                rec_flats.append(self._rec_flat[ids].copy()); rec_bins.append(self._rec_bin[ids].copy()); rec_succ.append(np.zeros(ids.size,np.float32)); self._rec_active[ids]=False; self._rec_streak[ids]=0

            emit_edge((self._edge_phase>0)&(done[t]>.5),0.0)
            edge_signal=(ramp[t]|cmd_resampled[t])&(~blocked); emit_edge((self._edge_phase==2)&edge_signal&(done[t]<.5),0.0)
            ids=np.flatnonzero((self._edge_phase==0)&edge_signal&(done[t]<.5))
            if ids.size:
                src=np.where(self._have_prev[ids,None],self._prev_cmd[ids],cmd_t[ids]); self._edge_source[ids]=src; self._edge_target[ids]=target_t[ids]
                six,siy,siz=self.command_indices(src[:,0],src[:,1],src[:,2]); self._edge_source_flat[ids]=self._flat_index(six,siy,siz)
                delta=np.linalg.norm((target_t[ids]-src)/EDGE_SCALES[None,:],axis=1); duration=np.where(ramp_speed[t,ids]>1e-6,float(control_dt)/np.maximum(ramp_speed[t,ids],1e-6),float(control_dt)); self._edge_rate_cmd[ids]=delta/np.maximum(duration,float(control_dt))
                self._edge_phase[ids]=1; self._edge_success_streak[ids]=0
            finish=(self._edge_phase==1)&(~edge_signal)&(done[t]<.5); ids=np.flatnonzero(finish)
            if ids.size: self._edge_phase[ids]=2; self._edge_success_streak[ids]=core_ok[t,ids].astype(np.int32)
            settling=self._edge_phase==2; cont=settling&(~finish); self._edge_success_streak[cont&core_ok[t]]+=1; self._edge_success_streak[cont&(~core_ok[t])]=0; emit_edge((self._edge_phase==2)&(self._edge_success_streak>=edge_target),1.0)

            clean=(age[t]>=float(NODE_SETTLE_DELAY_S))&(~resilience[t])&(~ramp[t])&(~blocked)&(done[t]<.5); current_flat=flat_all[t]
            ids=np.flatnonzero(self._node_active&((self._node_flat!=current_flat)|(~clean)))
            if ids.size:
                self._node_active[ids]=False; self._node_frames[ids]=0; self._node_structural_good_frames[ids]=0
                self._node_err_sum_rt[ids]=0.0; self._node_tol_sum_rt[ids]=0.0; self._node_quality_sum_rt[ids]=0.0
            ids=np.flatnonzero(clean&(~self._node_active))
            if ids.size: self._node_active[ids]=True; self._node_flat[ids]=current_flat[ids]
            active=self._node_active&clean; self._node_frames[active]+=1; self._node_structural_good_frames[active]+=structural_ok[t,active].astype(np.int32)
            self._node_err_sum_rt[active,0]+=np.abs(vx[t,active]-cvx[t,active]); self._node_err_sum_rt[active,1]+=np.abs(vy[t,active]-cvy[t,active]); self._node_err_sum_rt[active,2]+=np.abs(yaw[t,active]-cyaw[t,active])
            self._node_tol_sum_rt[active,0]+=vx_tol[t,active]; self._node_tol_sum_rt[active,1]+=vy_tol[t,active]; self._node_tol_sum_rt[active,2]+=yaw_tol[t,active]; self._node_quality_sum_rt[active]+=quality[t,active]
            ids=np.flatnonzero(active&(self._node_frames>=node_target))
            if ids.size:
                frames=np.maximum(self._node_frames[ids,None],1); ok_track=np.all((self._node_err_sum_rt[ids]/frames)<=(self._node_tol_sum_rt[ids]/frames),axis=1); ok_structure=(self._node_structural_good_frames[ids]/np.maximum(self._node_frames[ids],1))>=float(NODE_STRUCTURAL_FRACTION_MIN)
                node_flats.append(self._node_flat[ids].copy()); node_succ.append((ok_track&ok_structure).astype(np.float32)); node_qual.append((self._node_quality_sum_rt[ids]/np.maximum(self._node_frames[ids],1)).astype(np.float32)); self._node_frames[ids]=0; self._node_structural_good_frames[ids]=0; self._node_err_sum_rt[ids]=0.; self._node_tol_sum_rt[ids]=0.; self._node_quality_sum_rt[ids]=0.
            unbiased=~blocked
            self._prev_cmd[unbiased]=cmd_t[unbiased]; self._have_prev[unbiased]=True
            self._have_prev[blocked]=False

        n_node=self._add_node_events(node_flats,node_succ,node_qual,step); n_rec=self._add_recovery_events(rec_flats,rec_bins,rec_succ); n_edge=self._add_edge_events(edge_sf,edge_src,edge_tgt,edge_rate,edge_succ)
        a=float(SLOW_ALPHA); self.node_slow+=a*(self.node_competence()-self.node_slow); self.edge_slow+=a*(self.edge_competence()-self.edge_slow); self.total_updates+=1
        out=self.summary(); out.update({'field_node_events_update':float(n_node),'field_recovery_events_update':float(n_rec),'field_edge_events_update':float(n_edge)}); return out

    # ------------------------------------------------------------------
    # Curriculum packet generation
    # ------------------------------------------------------------------
    def _diverse_top(self, score: np.ndarray, count: int, *, valid: np.ndarray | None = None) -> list[tuple[int,int,int]]:
        s = np.asarray(score, np.float64).copy()
        if valid is not None:
            s = np.where(valid, s, -np.inf)
        chosen: list[tuple[int,int,int]] = []
        for _ in range(max(0, int(count))):
            flat = int(np.argmax(s))
            if not np.isfinite(s.ravel()[flat]):
                break
            idx = tuple(int(x) for x in np.unravel_index(flat, self.shape))
            chosen.append(idx)
            x,y,z = idx
            s[max(0,x-1):min(self.shape[0],x+2),
              max(0,y-1):min(self.shape[1],y+2),
              max(0,z-1):min(self.shape[2],z+2)] *= 0.25
            s[idx] = -np.inf
        return chosen

    def _recovery_teaching_severity(self, q: tuple[int,int,int]) -> float:
        p = self.recovery_competence()[q]
        lb = self.recovery_lower()[q]
        n = self.recovery_evidence()[q]
        bins = RECOVERY_BINS
        known = n >= 1.5
        if not np.any(known):
            return float(0.5 / max(1, bins))
        # Consolidate near the learnable ~60% frontier.
        score = np.where(known, np.abs(p - 0.62), np.inf)
        k = int(np.argmin(score))
        mastered = np.flatnonzero((n >= STRONG_EFFECTIVE_EVIDENCE) &
                                  (lb >= RECOVERY_MASTER_LOWER))
        if mastered.size:
            nxt = min(bins-1, int(mastered.max())+1)
            if n[nxt] < STRONG_EFFECTIVE_EVIDENCE:
                k = nxt
        return float((k + 0.5) / max(1, bins))

    def node_anchor_packet(self, count: int = NODE_ANCHORS) -> np.ndarray:
        p = self.node_competence(); lb = self.node_lower(); n = self.node_evidence()
        trend = p - self.node_slow
        known = n >= MIN_EFFECTIVE_EVIDENCE
        strong = n >= STRONG_EFFECTIVE_EVIDENCE
        mastered = strong & (lb >= NODE_MASTER_LOWER)

        # Productive frontier: learnable, not already mastered, with bonus for
        # positive learning velocity and enough evidence to be meaningful.
        frontier = np.exp(-0.5*((p-0.62)/0.19)**2)
        frontier *= (0.65 + 0.35*np.tanh(4.0*np.maximum(trend,0.0)))
        frontier *= np.clip(n / max(STRONG_EFFECTIVE_EVIDENCE,1e-6),0.0,1.0)
        frontier *= (~mastered)

        # Regression/staleness: a formerly stronger cell whose current
        # competence/evidence is fading deserves rehearsal.
        repair = np.maximum(self.node_slow-p,0.0)*3.0
        repair += np.where((p>0.75)&(n<STRONG_EFFECTIVE_EVIDENCE),0.35,0.0)

        # Unknown cells immediately adjacent to mastered territory are the
        # cleanest expansion targets. This mechanism was effectively disabled
        # because pseudo-probability alone never reaches reliable mastery.
        adjacent = np.zeros_like(mastered)
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                for dz in (-1,0,1):
                    if dx==dy==dz==0:
                        continue
                    src = mastered[max(0,-dx):self.shape[0]-max(0,dx),
                                   max(0,-dy):self.shape[1]-max(0,dy),
                                   max(0,-dz):self.shape[2]-max(0,dz)]
                    dst = adjacent[max(0,dx):self.shape[0]-max(0,-dx),
                                   max(0,dy):self.shape[1]-max(0,-dy),
                                   max(0,dz):self.shape[2]-max(0,-dz)]
                    dst |= src
        explore = adjacent & (~known)

        # Under-covered but already encountered cells receive a modest bonus;
        # this reduces seed-history bias without forcing artificial symmetry.
        undercovered = known & (n < STRONG_EFFECTIVE_EVIDENCE) & (p > 0.45)
        coverage_score = np.where(undercovered, 1.0 - n/max(STRONG_EFFECTIVE_EVIDENCE,1e-6), 0.0)

        n_front=max(1,int(round(count*0.40)))
        n_repair=max(1,int(round(count*0.20)))
        n_explore=max(1,int(round(count*0.25)))
        n_cover=max(1,int(round(count*0.10)))
        n_master=max(0,count-n_front-n_repair-n_explore-n_cover)
        picks=[]
        picks += self._diverse_top(frontier,n_front,valid=known)
        picks += self._diverse_top(repair,n_repair,valid=(n>0.5))
        picks += self._diverse_top(np.ones(self.shape),n_explore,valid=explore)
        picks += self._diverse_top(coverage_score,n_cover,valid=undercovered)
        picks += self._diverse_top(p,n_master,valid=mastered)

        uniq=[]; seen=set()
        for q in picks:
            if q not in seen:
                uniq.append(q); seen.add(q)
        if len(uniq)<count:
            for q in self._diverse_top(n,count-len(uniq),valid=n>0):
                if q not in seen:
                    uniq.append(q); seen.add(q)

        out=np.zeros((count,5),np.float32)
        for i,q in enumerate(uniq[:count]):
            out[i,:3]=self.command_for_index(*q)
            w=0.20+float(frontier[q])+float(repair[q])+0.45*float(explore[q])+0.35*float(coverage_score[q])
            out[i,3]=max(0.05,w)
            out[i,4]=self._recovery_teaching_severity(q)
        return out

    def edge_anchor_packet(self, count: int = EDGE_ANCHORS) -> np.ndarray:
        ep=self.edge_competence(); elb=self.edge_lower(); en=self.edge_evidence(); trend=ep-self.edge_slow
        np_=self.node_competence(); nlb=self.node_lower(); nn=self.node_evidence()
        known=en>=2.0
        src_ok=(nn>=STRONG_EFFECTIVE_EVIDENCE)&(nlb>=0.65)
        mastered=(en>=STRONG_EFFECTIVE_EVIDENCE)&(elb>=EDGE_MASTER_LOWER)
        score=np.exp(-0.5*((ep-0.62)/0.20)**2)*(0.7+0.3*np.tanh(4*np.maximum(trend,0)))
        score += 1.5*np.maximum(self.edge_slow-ep,0)
        # Under-covered local directions get a small exploration bonus.
        score += np.where((en>0)&(en<STRONG_EFFECTIVE_EVIDENCE),0.25,0.0)
        score *= src_ok[...,None]
        score *= (~mastered)
        score=np.where(known,score,-np.inf)

        out=np.zeros((count,7),np.float32)
        work=score.copy()
        grid_step=np.asarray([(hi-lo)/(n-1) for (lo,hi),n in zip(self.bounds,self.shape)],np.float32)
        for i in range(count):
            flat=int(np.argmax(work))
            if not np.isfinite(work.ravel()[flat]):
                break
            ix,iy,iz,sec=np.unravel_index(flat,work.shape)
            src=self.command_for_index(ix,iy,iz)
            delta=EDGE_DIRS[sec]*grid_step*2.0
            tgt=src+delta
            for a,(lo,hi) in enumerate(self.bounds):
                tgt[a]=np.clip(tgt[a],lo,hi)
            out[i,:3]=src; out[i,3:6]=tgt; out[i,6]=max(0.05,float(work[ix,iy,iz,sec]))
            work[max(0,ix-1):min(self.shape[0],ix+2),
                 max(0,iy-1):min(self.shape[1],iy+2),
                 max(0,iz-1):min(self.shape[2],iz+2),sec] *= 0.25
            work[ix,iy,iz,sec]=-np.inf
        return out

    def influence(self, step: int, *, start: int, full: int, maximum: float, mode: str,
                  fade_start: int = 0, zero: int = 0) -> tuple[float,float]:
        """Teaching dose with optional ramp-up and fade-out.

        The default is the historical one-way ramp.  ``fade_start``/``zero``
        create the tutor-then-release schedule without adding another runtime
        subsystem: influence is simply multiplied by a linear fade to zero.
        """
        if mode in ('off','shadow') or step <= start:
            return 0.0,0.0
        x=1.0 if full<=start else float(np.clip((step-start)/(full-start),0.0,1.0))
        if fade_start > 0 and zero > fade_start and step > fade_start:
            x *= float(np.clip((zero-step)/(zero-fade_start),0.0,1.0))
        node=float(maximum)*x
        edge=node if mode=='graph' else 0.0
        return node,edge

    def teaching_packet(self, *, step: int, mode: str, start: int, full: int, maximum: float,
                        fade_start: int = 0, zero: int = 0,
                        teach_nodes: bool = True, teach_edges: bool = True,
                        node_count: int = NODE_ANCHORS, edge_count: int = EDGE_ANCHORS) -> dict[str, Any]:
        nf,ef=self.influence(step,start=start,full=full,maximum=maximum,mode=mode,
                             fade_start=fade_start,zero=zero)
        if not teach_nodes:
            nf=0.0
        if not teach_edges:
            ef=0.0
        return {
            'field_node_frac':nf,
            'field_edge_frac':ef,
            'field_node_anchors':self.node_anchor_packet(node_count),
            'field_edge_anchors':self.edge_anchor_packet(edge_count),
        }

    # ------------------------------------------------------------------
    # Held-out calibration selection (selection only; no feedback)
    # ------------------------------------------------------------------
    def predict_command(self, vx: float, vy: float, yaw: float) -> tuple[float, float]:
        ix,iy,iz=self.command_indices(np.asarray([vx]),np.asarray([vy]),np.asarray([yaw]))
        q=(int(ix[0]),int(iy[0]),int(iz[0]))
        return float(self.node_competence()[q]), float(self.node_evidence()[q])

    def calibration_commands(self, count: int) -> list[tuple[float,float,float,float,float]]:
        count=max(1,int(count))
        p=self.node_competence(); n=self.node_evidence()
        canonical=np.asarray([0.92,0.78,0.62,0.45,0.28],dtype=np.float32)
        targets=[float(canonical[j%len(canonical)]) for j in range(count)]
        candidates=[]; used=set(); known=n>=MIN_EFFECTIVE_EVIDENCE
        for t in targets:
            score=np.abs(p-t)+0.16/np.sqrt(np.maximum(n,0.05))
            score=np.where(known,score,np.inf)
            for q in used:
                x,y,z=q
                score[max(0,x-1):min(self.shape[0],x+2),
                      max(0,y-1):min(self.shape[1],y+2),
                      max(0,z-1):min(self.shape[2],z+2)]=np.inf
            flat=int(np.argmin(score))
            if not np.isfinite(score.ravel()[flat]):
                break
            q=tuple(int(x) for x in np.unravel_index(flat,self.shape)); used.add(q)
            cmd=self.command_for_index(*q)
            candidates.append((float(cmd[0]),float(cmd[1]),float(cmd[2]),float(p[q]),float(n[q])))
        if len(candidates)<count:
            work=n.copy()
            for _ in range(count-len(candidates)):
                flat=int(np.argmax(work)); q=tuple(int(x) for x in np.unravel_index(flat,self.shape))
                if work[q]<=0: break
                cmd=self.command_for_index(*q)
                candidates.append((float(cmd[0]),float(cmd[1]),float(cmd[2]),float(p[q]),float(n[q])))
                work[q]=-1
        return candidates

    # ------------------------------------------------------------------
    # Reporting / persistence
    # ------------------------------------------------------------------
    def summary(self) -> dict[str,float]:
        p=self.node_competence(); lb=self.node_lower(); n=self.node_evidence(); q=self.node_quality()
        known=n>=MIN_EFFECTIVE_EVIDENCE
        strong=n>=STRONG_EFFECTIVE_EVIDENCE
        mastered=strong&(lb>=NODE_MASTER_LOWER)
        frontier=known&(~mastered)&(p>=0.35)
        age=np.where(self.node_last_step>=0,np.maximum(0,int(self.last_step)-self.node_last_step),0)
        stale=mastered&(age>=float(self.half_life_steps)*float(STALE_HALF_LIVES))

        rp=self.recovery_competence(); rlb=self.recovery_lower(); rn=self.recovery_evidence()
        rknown=rn>=2.0; rmaster=(rn>=STRONG_EFFECTIVE_EVIDENCE)&(rlb>=RECOVERY_MASTER_LOWER)
        ep=self.edge_competence(); elb=self.edge_lower(); en=self.edge_evidence()
        eknown=en>=2.0; emaster=(en>=STRONG_EFFECTIVE_EVIDENCE)&(elb>=EDGE_MASTER_LOWER)
        return {
            'field_schema':float(FIELD_VERSION),
            'field_known_nodes':float(np.sum(known)),
            'field_strong_nodes':float(np.sum(strong)),
            'field_competent_nodes':float(np.sum(mastered)),
            'field_frontier_nodes':float(np.sum(frontier)),
            'field_stale_nodes':float(np.sum(stale)),
            'field_mean_competence_known':float(np.mean(p[known])) if np.any(known) else 0.5,
            'field_mean_lower_known':float(np.mean(lb[known])) if np.any(known) else 0.0,
            'field_mean_quality_known':float(np.mean(q[known])) if np.any(known) else 0.0,
            'field_learning_nodes':float(np.sum(known&((p-self.node_slow)>0.05))),
            'field_forgetting_nodes':float(np.sum(known&((self.node_slow-p)>0.05))),
            'field_known_recovery_cells':float(np.sum(rknown)),
            'field_competent_recovery_cells':float(np.sum(rmaster)),
            'field_known_edges':float(np.sum(eknown)),
            'field_competent_edges':float(np.sum(emaster)),
            'field_mean_edge_competence':float(np.mean(ep[eknown])) if np.any(eknown) else 0.5,
            'field_mean_edge_rate':float(np.mean(self.edge_rate()[eknown])) if np.any(eknown) else 0.0,
            'field_updates':float(self.total_updates),
            'field_node_events_total':float(self.total_node_events),
            'field_recovery_events_total':float(self.total_recovery_events),
            'field_edge_events_total':float(self.total_edge_events),
            'field_effective_samples':float(np.sum(n)),
            'field_last_decay':float(self.last_decay),
        }

    def state_dict(self) -> dict[str,Any]:
        return {
            'field_version':FIELD_VERSION,
            'half_life_steps':float(self.half_life_steps),
            'last_step':int(self.last_step),'total_updates':int(self.total_updates),
            'total_node_events':int(self.total_node_events),'total_recovery_events':int(self.total_recovery_events),
            'total_edge_events':int(self.total_edge_events),
            'node_success':self.node_success,'node_failure':self.node_failure,
            'node_quality_sum':self.node_quality_sum,
            'node_slow':self.node_slow,'node_last_step':self.node_last_step,
            'rec_success':self.rec_success,'rec_failure':self.rec_failure,
            'edge_success':self.edge_success,'edge_failure':self.edge_failure,
            'edge_rate_sum':self.edge_rate_sum,'edge_slow':self.edge_slow,
        }

    @classmethod
    def from_state_dict(cls, d: dict[str,Any]) -> 'CapabilityField':
        arrays = (
            'node_success','node_failure','node_quality_sum','node_slow','node_last_step',
            'rec_success','rec_failure',
            'edge_success','edge_failure','edge_rate_sum','edge_slow')
        expected = {
            'field_version','half_life_steps','last_step','total_updates','total_node_events',
            'total_recovery_events','total_edge_events', *arrays}
        if not isinstance(d, dict) or set(d) != expected:
            got = sorted(d) if isinstance(d, dict) else type(d).__name__
            raise ValueError(f'capability field state keys mismatch: expected={sorted(expected)} got={got}')
        version=int(d['field_version'])
        if version != FIELD_VERSION:
            raise ValueError(f'capability field schema {version} cannot resume as schema {FIELD_VERSION}; start a fresh current-schema map')
        obj=cls(float(d['half_life_steps']))
        for name in arrays:
            dtype=np.int64 if name.endswith('_last_step') else np.float32
            arr=np.asarray(d[name],dtype)
            expected_shape=getattr(obj,name).shape
            if arr.shape != expected_shape:
                raise ValueError(f'capability field array shape mismatch for {name}: checkpoint={arr.shape} current={expected_shape}')
            setattr(obj,name,arr.copy())
        obj.last_step=int(d['last_step']); obj.total_updates=int(d['total_updates'])
        obj.total_node_events=int(d['total_node_events']); obj.total_recovery_events=int(d['total_recovery_events'])
        obj.total_edge_events=int(d['total_edge_events'])
        return obj

    def fingerprint(self) -> str:
        """Stable digest of all serialized field state.

        Held-out evaluation uses this as a quarantine invariant: asking the
        field which commands to examine is allowed; any mutation caused by the
        exam itself is a hard error.
        """
        h=hashlib.sha256()
        h.update(str(float(self.half_life_steps)).encode('ascii'))
        h.update(str((self.last_step,self.total_updates,self.total_node_events,self.total_recovery_events,self.total_edge_events)).encode('ascii'))
        for name in (
            'node_success','node_failure','node_quality_sum','node_slow','node_last_step',
            'rec_success','rec_failure',
            'edge_success','edge_failure','edge_rate_sum','edge_slow'):
            a=np.ascontiguousarray(getattr(self,name))
            h.update(name.encode('ascii')); h.update(str(a.shape).encode('ascii')); h.update(str(a.dtype).encode('ascii')); h.update(a.tobytes())
        return h.hexdigest()

    def save_npz(self, path: str | Path) -> None:
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,
            field_schema=np.asarray([FIELD_VERSION],np.int32),
            step=np.asarray([self.last_step],np.int64),half_life_steps=np.asarray([self.half_life_steps],np.float64),
            vx=self.centers[0],vy=self.centers[1],yaw=self.centers[2],
            node_competence=self.node_competence(),node_lower=self.node_lower(),node_evidence=self.node_evidence(),
            node_quality=self.node_quality(),node_confidence=self.node_confidence(),node_slow=self.node_slow,
            node_last_step=self.node_last_step,
            recovery_competence=self.recovery_competence(),recovery_lower=self.recovery_lower(),
            recovery_evidence=self.recovery_evidence(),
            edge_competence=self.edge_competence(),edge_lower=self.edge_lower(),edge_evidence=self.edge_evidence(),
            edge_rate=self.edge_rate(),edge_dirs=EDGE_DIRS)

    def write_summary(self, path: str | Path, *, step: int, mode: str, node_frac: float, edge_frac: float) -> None:
        payload={'step':int(step),'mode':str(mode),'node_influence':float(node_frac),'edge_influence':float(edge_frac),**self.summary()}
        Path(path).write_text(json.dumps(payload,indent=2,sort_keys=True),encoding='utf-8')


# Held-out capability cloud: test teaching hypotheses without feedback leakage.
import math
from typing import Iterable

import numpy as np
from .contracts import (CONTROL_DT, COMBO_P, RECOVERY_TILT_MAX_DEG, RECOVERY_LINVEL_MAX_MPS,
                        RECOVERY_ANGVEL_MAX_RAD_S, RECOVERY_JOINT_MAX_DEG, task_param_defaults)
from .contracts import recovery_outcome_numpy


FAMILIES = ('none', 'pose', 'momentum', 'impact', 'chaos')
# Deliberately a little beyond the current 120M reference envelope.  These are
# sampling scales, NOT claims of hardware capability and NOT training limits.
CMD_SCALE = {
    'vx_pos': 1.05,
    'vx_neg': 0.55,
    'vy': 0.55,
    'yaw': 4.8,
}


def halton(index: int, base: int) -> float:
    """Small deterministic low-discrepancy scalar in [0,1)."""
    i = int(index) + 1
    f = 1.0
    r = 0.0
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


def pnorm(v, p: float = COMBO_P) -> float:
    a = np.abs(np.asarray(v, dtype=float))
    return float(np.power(np.sum(np.power(a, p)), 1.0 / p))


def _direction(index: int) -> np.ndarray:
    # Three irrational-looking low-discrepancy channels.  If a rare point lands
    # too close to the origin, rotate to another deterministic point rather than
    # creating an ill-defined direction.
    v = np.array([2*halton(index, 2)-1, 2*halton(index, 3)-1, 2*halton(index, 5)-1], dtype=float)
    n = pnorm(v)
    if n < 0.12:
        v = np.array([2*halton(index+17, 2)-1, 2*halton(index+17, 3)-1, 2*halton(index+17, 5)-1], dtype=float)
        n = max(pnorm(v), 1e-9)
    return v / n


def command_from_normalized(c: Iterable[float]) -> tuple[float,float,float]:
    cx, cy, cz = [float(x) for x in c]
    vx = cx * (CMD_SCALE['vx_pos'] if cx >= 0 else CMD_SCALE['vx_neg'])
    vy = cy * CMD_SCALE['vy']
    yaw = cz * CMD_SCALE['yaw']
    return float(vx), float(vy), float(yaw)


def command_sector(c: Iterable[float], active_frac: float = 0.34) -> str:
    """Human-readable coarse direction cell for a genuine 3-D command mixture.

    Raw cx/cy/cyaw remain authoritative.  The sector is only a compact way to
    keep direction visible in summaries so opposite/mixed corners are never
    averaged together merely because they share the same scalar command load.
    Components smaller than ``active_frac`` of the largest component are treated
    as nuisance mixture for the label, not discarded from the actual probe.
    """
    a=np.asarray(list(c),dtype=float)
    mx=max(float(np.max(np.abs(a))),1e-9)
    names=(('F','R'),('L','Rt'),('CCW','CW'))
    bits=[]
    for x,(pos,neg) in zip(a,names):
        if abs(float(x)) >= active_frac*mx:
            bits.append(pos if x>=0 else neg)
    return '+'.join(bits) if bits else 'mixed'



def make_probe(index: int, *, family: str | None = None, load: float | None = None,
               severity: float | None = None, source: str = 'explore',
               anchor_index: int | None = None, bundle_id: int | None = None,
               panel_id: int = 0) -> dict:
    """Generate one deterministic held-out point with matched-command bundling.

    Consecutive groups of ``len(FAMILIES)`` share the *exact same* vx/vy/yaw
    command and differ only in disturbance family.  This removes the stage-4
    family/yaw alias and gives every disturbed probe an undisturbed matched
    control without adding another map dimension.
    """
    fam_count = len(FAMILIES)
    if family is None:
        family = FAMILIES[int(index) % fam_count]
    if family not in FAMILIES:
        raise ValueError(f'unknown disturbance family {family!r}')
    if anchor_index is None:
        anchor_index = int(index) // fam_count
    anchor_index = int(anchor_index)
    if bundle_id is None:
        bundle_id = int(panel_id) * 100000 + anchor_index

    # Command generation is keyed only by the anchor, never by family.
    d = _direction(anchor_index)
    if load is None:
        load = 0.18 + 1.06 * halton(anchor_index, 7)
    c = d * float(load)
    vx, vy, yaw = command_from_normalized(c)

    if severity is None:
        if family == 'none':
            severity = 0.0
        else:
            fi = FAMILIES.index(family)
            # Family-specific severity sequence is offset independently from
            # command direction.  Because every family shares each command
            # anchor, any residual sequence structure cannot masquerade as
            # a direction-family effect.
            severity = 0.10 + 1.20 * halton(anchor_index + 97 * (fi + 1), 11)
    severity = max(0.0, float(severity))
    return {
        'probe_index': int(index), 'source': source, 'panel_id': int(panel_id),
        'bundle_id': int(bundle_id), 'anchor_index': anchor_index,
        'cmd_cx': float(c[0]), 'cmd_cy': float(c[1]), 'cmd_cyaw': float(c[2]),
        'cmd_load': float(pnorm(c)), 'cmd_sector': command_sector(c),
        'cmd_vx': vx, 'cmd_vy': vy, 'cmd_yaw': yaw,
        'family': family, 'severity': severity, 'cloud_role': 'heldout_probe',
    }


def make_panel(point_budget: int, *, panel_id: int = 0, source: str = 'panel',
               core_bundles: int = 2) -> list[dict]:
    """Return complete matched-command bundles with fixed core + rotating scouts.

    A routine 20-point panel contains four command anchors × five contexts.
    The first two command anchors are permanent longitudinal controls.  The
    remaining anchors rotate deterministically with ``panel_id`` so the held-out
    point cloud expands over policy age without sacrificing direct comparison.
    A larger final panel with panel_id=0 begins at anchor zero and therefore
    densely rechecks the region accumulated by the early routine panels.
    """
    fam_count = len(FAMILIES)
    bundles = max(1, int(math.ceil(max(1, int(point_budget)) / fam_count)))
    core = min(max(0, int(core_bundles)), bundles)
    rotating = max(0, bundles - core)
    out = []
    for b in range(bundles):
        if b < core:
            anchor = b
        else:
            anchor = core + int(panel_id) * max(1, rotating) + (b - core)
        bundle_id = int(panel_id) * 100000 + anchor
        for fi, fam in enumerate(FAMILIES):
            idx = anchor * fam_count + fi
            out.append(make_probe(idx, family=fam, anchor_index=anchor, bundle_id=bundle_id,
                                  panel_id=panel_id, source=source))
    return out


def make_calibration_panel(commands, *, panel_id: int = 0, source: str = 'field_calibration') -> list[dict]:
    """Build matched held-out bundles from commands chosen by the live field.

    ``commands`` items are (vx, vy, yaw, predicted_competence, evidence).  Only
    the coordinates/prediction are copied; evaluator outcomes stay quarantined
    and are never returned to the training-derived field.
    """
    out=[]
    for b,item in enumerate(commands):
        vx,vy,yaw=[float(x) for x in item[:3]]
        pred=float(item[3]) if len(item)>3 else float('nan')
        evidence=float(item[4]) if len(item)>4 else float('nan')
        cx=vx/(CMD_SCALE['vx_pos'] if vx>=0 else CMD_SCALE['vx_neg'])
        cy=vy/CMD_SCALE['vy']; cz=yaw/CMD_SCALE['yaw']
        c=np.asarray([cx,cy,cz],dtype=float)
        bundle_id=int(panel_id)*100000+b
        for fi,fam in enumerate(FAMILIES):
            sev=0.0 if fam=='none' else 0.10+1.20*halton(b+97*(fi+1)+panel_id*13,11)
            out.append({
                'probe_index':bundle_id*len(FAMILIES)+fi,'source':source,'panel_id':int(panel_id),
                'bundle_id':bundle_id,'anchor_index':b,
                'cmd_cx':float(c[0]),'cmd_cy':float(c[1]),'cmd_cyaw':float(c[2]),
                'cmd_load':float(pnorm(c)),'cmd_sector':command_sector(c),
                'cmd_vx':vx,'cmd_vy':vy,'cmd_yaw':yaw,'family':fam,'severity':float(sev),
                'cloud_role':'heldout_calibration','field_predicted_competence':pred,
                'field_effective_evidence':evidence,
            })
    return out


def make_common_recovery_panel(point_budget: int = 50, *, source: str = 'final_common_recovery') -> list[dict]:
    """Deterministic mature-policy recovery exam with useful matched controls.

    Sample a conservative but
    genuinely 3-D operating region; physical limits still belong to the dense
    atlas.  Every command is tested as none/pose/momentum/impact/chaos.
    """
    anchors = [
        (0.10,  0.00,  0.00),
        (0.30,  0.00,  0.00),
        (0.50,  0.00,  0.00),
        (-0.12, 0.00,  0.00),
        (-0.28, 0.00,  0.00),
        (0.22,  0.10,  0.00),
        (0.22, -0.10,  0.00),
        (0.22,  0.00,  0.75),
        (0.22,  0.00, -0.75),
        (0.30,  0.09,  0.70),
        (0.30, -0.09, -0.70),
        (-0.18, 0.07,  0.55),
    ]
    fam_count = len(FAMILIES)
    bundles = max(1, int(math.ceil(max(1, int(point_budget)) / fam_count)))
    out=[]
    for b in range(bundles):
        vx,vy,yaw=anchors[b % len(anchors)]
        # If callers ask for more bundles than the canonical set, repeat at a
        # slightly reduced amplitude rather than inventing unbounded extremes.
        cycle=b//len(anchors)
        scale=max(0.70,1.0-0.08*cycle)
        vx,vy,yaw=vx*scale,vy*scale,yaw*scale
        cx=vx/(CMD_SCALE['vx_pos'] if vx>=0 else CMD_SCALE['vx_neg'])
        cy=vy/CMD_SCALE['vy']; cz=yaw/CMD_SCALE['yaw']
        c=np.asarray([cx,cy,cz],dtype=float)
        bundle_id=900000+b
        for fi,fam in enumerate(FAMILIES):
            # Moderate fixed severities make the exam comparable and avoid
            # conflating capability with a random severity draw.
            sev=0.0 if fam=='none' else (0.55 + 0.05*((b+fi)%3))
            out.append({
                'probe_index':bundle_id*fam_count+fi,'source':source,'panel_id':-1,
                'bundle_id':bundle_id,'anchor_index':b,
                'cmd_cx':float(c[0]),'cmd_cy':float(c[1]),'cmd_cyaw':float(c[2]),
                'cmd_load':float(pnorm(c)),'cmd_sector':command_sector(c),
                'cmd_vx':float(vx),'cmd_vy':float(vy),'cmd_yaw':float(yaw),
                'family':fam,'severity':float(sev),'cloud_role':'heldout_common_recovery',
            })
    return out

def disturbance_overrides(spec: dict) -> dict:
    """Map one family+severity into explicit physical probe quantities.

    Directions/signs remain randomized per environment inside environment.py,
    which deliberately integrates nuisance dimensions rather than mapping them.
    """
    fam = spec['family']; s = float(spec['severity'])
    out = {
        'probe_tilt_rad': -1.0,
        'probe_linvel_mps': -1.0,
        'probe_angvel_rad_s': -1.0,
        'probe_joint_rad': -1.0,
        'probe_push_n': -1.0,
        'probe_push_steps': 5.0,
    }
    if fam == 'pose':
        out['probe_tilt_rad'] = math.radians(38.0) * s
        out['probe_joint_rad'] = math.radians(20.0) * s
    elif fam == 'momentum':
        out['probe_linvel_mps'] = 0.95 * s
        out['probe_angvel_rad_s'] = 5.0 * s
    elif fam == 'impact':
        # Five control frames = 0.10 s.  At severity 1 this is 0.28 N*s.
        out['probe_push_steps'] = 5.0
        out['probe_push_n'] = 2.8 * s
    elif fam == 'chaos':
        out['probe_tilt_rad'] = math.radians(RECOVERY_TILT_MAX_DEG) * s
        out['probe_joint_rad'] = math.radians(RECOVERY_JOINT_MAX_DEG) * s
        out['probe_linvel_mps'] = RECOVERY_LINVEL_MAX_MPS * s
        out['probe_angvel_rad_s'] = RECOVERY_ANGVEL_MAX_RAD_S * s
        out['probe_push_steps'] = 5.0
        out['probe_push_n'] = 2.0 * s
    return out


def physical_descriptor(spec: dict) -> dict:
    ov = disturbance_overrides(spec)
    return {
        'tilt_deg_max': max(0.0, math.degrees(ov['probe_tilt_rad'])) if ov['probe_tilt_rad'] >= 0 else 0.0,
        'joint_offset_deg_max': max(0.0, math.degrees(ov['probe_joint_rad'])) if ov['probe_joint_rad'] >= 0 else 0.0,
        'linvel_kick_mps_max': max(0.0, ov['probe_linvel_mps']),
        'angvel_kick_rad_s_max': max(0.0, ov['probe_angvel_rad_s']),
        'push_force_n': max(0.0, ov['probe_push_n']),
        'push_duration_s': max(0.0, ov['probe_push_steps']) * CONTROL_DT if ov['probe_push_n'] >= 0 else 0.0,
        'push_impulse_ns': max(0.0, ov['probe_push_n']) * max(0.0, ov['probe_push_steps']) * CONTROL_DT if ov['probe_push_n'] >= 0 else 0.0,
    }


def base_episode(spec: dict) -> dict:
    ep = task_param_defaults()
    ep.update(regime_command_frac=1.0,
              probe_cmd_vx=float(spec['cmd_vx']), probe_cmd_vy=float(spec['cmd_vy']),
              probe_cmd_yaw=float(spec['cmd_yaw']))
    ep.update(disturbance_overrides(spec))
    return ep



def _not_before_for_spec(spec: dict) -> int:
    if spec.get('family') in ('impact', 'chaos'):
        return 6 + int(round(float(disturbance_overrides(spec)['probe_push_steps'])))
    return 0


def _row_from_times(spec: dict, ff, fr, not_before: int) -> dict:
    row = dict(spec)
    row.update(physical_descriptor(spec))
    row.update(recovery_outcome_numpy(ff, fr, not_before_steps=not_before, direction_bins=8))
    return row


def evaluate_specs_batch(trainer, env, params, obs_mean, obs_std, keys, horizon: int,
                         specs: list[dict], stable_steps: int = 12, chunk_size: int = 4) -> list[dict]:
    """Evaluate several cloud points concurrently with an outer vmap.

    Each point still contains ``env.num_envs`` independently stratified robots.
    Batching four 256-env points gives MJX roughly 1024 environments of work per
    dispatch instead of repeatedly under-filling the GPU with one small cloud
    point at a time. The final partial chunk is padded and then discarded so JAX
    sees one stable batch shape and compiles it once per horizon/chunk size.
    """
    if not specs:
        return []
    import jax
    import jax.numpy as jnp
    chunk_size=max(1,int(chunk_size))
    horizon=int(horizon); stable_steps=int(stable_steps)
    cache=trainer.setdefault('_cloud_batch_eval_cache', {})
    cache_key=(horizon, stable_steps, chunk_size)
    batched=cache.get(cache_key)
    if batched is None:
        def fn(policy_params, om, osd, cps, ks, nbs):
            return jax.vmap(
                lambda cp, k, nb: trainer['evaluate_recovery'](
                    policy_params, cp, om, osd, k, horizon, stable_steps, nb),
                in_axes=(0,0,0))(cps, ks, nbs)
        batched=jax.jit(fn)
        cache[cache_key]=batched

    rows=[]
    keys=list(keys)
    if len(keys) != len(specs):
        raise ValueError('keys/specs length mismatch')
    for start in range(0,len(specs),chunk_size):
        real_specs=list(specs[start:start+chunk_size])
        real_keys=list(keys[start:start+chunk_size])
        nreal=len(real_specs)
        pad_spec=real_specs[-1]
        pad_key=real_keys[-1]
        padded_specs=real_specs + [pad_spec]*(chunk_size-nreal)
        padded_keys=real_keys + [pad_key]*(chunk_size-nreal)
        cps=jnp.stack([env.params_array(base_episode(x)) for x in padded_specs],axis=0)
        ks=jnp.stack(padded_keys,axis=0)
        nbs=jnp.asarray([_not_before_for_spec(x) for x in padded_specs],dtype=jnp.int32)
        ff,fr=batched(params,obs_mean,obs_std,cps,ks,nbs)
        ff,fr=jax.device_get((ff,fr))
        for i in range(nreal):
            rows.append(_row_from_times(real_specs[i],ff[i],fr[i],int(nbs[i])))

    # Node competence is calibrated against a
    # held-out NODE event, not against the recovery event above. Only controls
    # need this extra pass; disturbed rows remain purely recovery evidence.
    control_idx=[i for i,spec in enumerate(specs) if spec.get('family') == 'none']
    if control_idx and 'evaluate_node_window' in trainer:
        ncache=trainer.setdefault('_cloud_node_eval_cache',{})
        nkey=(horizon,chunk_size)
        nbatched=ncache.get(nkey)
        if nbatched is None:
            def nfn(policy_params,om,osd,cps,ks):
                return jax.vmap(lambda cp,k: trainer['evaluate_node_window'](
                    policy_params,cp,om,osd,k,horizon,50,25),in_axes=(0,0))(cps,ks)
            nbatched=jax.jit(nfn); ncache[nkey]=nbatched
        for start in range(0,len(control_idx),chunk_size):
            ids=control_idx[start:start+chunk_size]; nreal=len(ids)
            pad=ids[-1]; pids=ids+[pad]*(chunk_size-nreal)
            cps=jnp.stack([env.params_array(base_episode(specs[i])) for i in pids],axis=0)
            ks=jnp.stack([keys[i] for i in pids],axis=0)
            verdict=np.asarray(jax.device_get(nbatched(params,obs_mean,obs_std,cps,ks)))
            for j,ri in enumerate(ids):
                rows[ri]['node_window_rate']=float(verdict[j].mean())
    return rows


def annotate_matched_controls(rows: list[dict]) -> list[dict]:
    """Attach matched undisturbed competence to every disturbed row.

    A disturbed result is treated as valid resilience evidence only when the
    exact same command is demonstrably competent without disturbance.
    """
    groups={}
    for r in rows:
        step=int(float(r.get('policy_step', -1) or -1))
        bid=int(float(r.get('bundle_id', -1) or -1))
        groups.setdefault((step,bid),[]).append(r)

    for vals in groups.values():
        controls=[r for r in vals if r.get('family') == 'none']
        if not controls:
            for r in vals:
                r['baseline_competent']=0
                r['resilience_valid']=0
                r['recovery_drop_from_control']=float('nan')
            continue
        # Panels intentionally have one control per bundle. If a final panel
        # duplicates a routine point at the same step, average the duplicates.
        base_rec=float(np.mean([float(r.get('recovery_rate',0.0)) for r in controls]))
        base_surv=float(np.mean([float(r.get('survival',0.0)) for r in controls]))
        competent = int(base_rec >= 0.80 and base_surv >= 0.95)
        for r in vals:
            r['baseline_competent']=competent
            r['resilience_valid']=int(competent and r.get('family') != 'none')
            r['recovery_drop_from_control']=(base_rec-float(r.get('recovery_rate',0.0))
                                             if r.get('family') != 'none' else 0.0)
    return rows


def _bin(x, edges):
    x=float(x)
    for i in range(len(edges)-1):
        if edges[i] <= x < edges[i+1]: return i
    return len(edges)-2



def summarize_rows(rows: list[dict]) -> dict:
    rows=annotate_matched_controls(rows)
    load_edges=[0.0,0.35,0.60,0.85,1.10,1.40]
    sev_edges=[0.0,0.25,0.50,0.75,1.00,1.35,9.0]
    out={'load_edges':load_edges,'severity_edges':sev_edges,'families':{}}

    for fam in FAMILIES:
        rr=[r for r in rows if r.get('family')==fam]
        grid=[]
        for li in range(len(load_edges)-1):
            line=[]
            for si in range(len(sev_edges)-1):
                vals=[r for r in rr if _bin(float(r['cmd_load']),load_edges)==li and _bin(float(r['severity']),sev_edges)==si]
                p90=[float(v['recovery_time_p90_s']) for v in vals if float(v.get('recovery_time_p90_s',-1))>=0]
                line.append({
                    'n':len(vals),
                    'recovery':float(np.mean([float(v['recovery_rate']) for v in vals])) if vals else None,
                    'survival':float(np.mean([float(v['survival']) for v in vals])) if vals else None,
                    'worst_direction':float(np.mean([float(v.get('recovery_worst_direction',0.0)) for v in vals])) if vals else None,
                    'p90_s':float(np.mean(p90)) if p90 else None,
                })
            grid.append(line)
        out['families'][fam]={'n':len(rr),'grid':grid}

    controls=[r for r in rows if r.get('family')=='none']
    disturbed=[r for r in rows if r.get('family')!='none']
    valid=[r for r in disturbed if int(float(r.get('resilience_valid',0) or 0))==1]
    out['paired']={
        'control_points':len(controls),
        'baseline_competent_fraction':float(np.mean([float(r.get('baseline_competent',0)) for r in controls])) if controls else None,
        'baseline_recovery_mean':float(np.mean([float(r.get('recovery_rate',0)) for r in controls])) if controls else None,
        'disturbed_points':len(disturbed),
        'valid_disturbance_points':len(valid),
        'valid_recovery_mean':float(np.mean([float(r.get('recovery_rate',0)) for r in valid])) if valid else None,
        'valid_worst_direction_mean':float(np.mean([float(r.get('recovery_worst_direction',0)) for r in valid])) if valid else None,
        'valid_survival_mean':float(np.mean([float(r.get('survival',0)) for r in valid])) if valid else None,
        'valid_recovery_drop_mean':float(np.mean([float(r.get('recovery_drop_from_control',0)) for r in valid])) if valid else None,
    }

    # Direction-aware sector summaries remain descriptive; raw vectors remain authoritative.
    sector_rows=[]
    keys=sorted(set((str(r.get('family')),str(r.get('cmd_sector'))) for r in rows))
    for fam,sector in keys:
        vals=[r for r in rows if str(r.get('family'))==fam and str(r.get('cmd_sector'))==sector]
        sector_rows.append({
            'family':fam,'sector':sector,'n':len(vals),
            'mean_load':float(np.mean([float(v['cmd_load']) for v in vals])),
            'mean_recovery':float(np.mean([float(v['recovery_rate']) for v in vals])),
            'mean_survival':float(np.mean([float(v['survival']) for v in vals])),
            'mean_worst_direction':float(np.mean([float(v.get('recovery_worst_direction',0)) for v in vals])),
        })
    out['direction_sectors']=sector_rows
    out['worst_points']=sorted(rows,key=lambda r:(float(r.get('recovery_rate',0)),float(r.get('survival',0))))[:12]
    return out


def record_capability_cloud_exam(
    snapshot, *, field, cloud_eval_env, cloud_eval_trainer,
    cloud_rows: list[dict], latest_eval: dict, run_dir, perf,
    log_prefix: str, force: bool, count: int, every: int,
) -> None:
    """Held-out competence/calibration exam; never a teaching observation.

    The field suggests *questions* and predicts their results. A fingerprint
    check forbids held-out answers from modifying the teaching field.
    """
    import jax
    import time
    from .training_config import EVAL_SEED, CLOUD_HORIZON, CLOUD_BATCH_POINTS, CLOUD_ENVS
    from .results import write_rows_csv, write_json
    step = snapshot.step
    t0_cloud = time.perf_counter()
    field_before_exam = field.fingerprint()

    # Screen no-disturbance controls first; only spend expensive disturbed
    # rollouts on command anchors that are already demonstrably competent.
    panel_id = 0 if force else max(0, int(step // max(1, every)) - 1)
    # Let the training-derived field choose
    # *questions* spanning its predicted competence bands, but never let
    # held-out answers update the field. If the map is still too young,
    # fall back to the deterministic paired panel.
    bundles=max(1,int(np.ceil(count/5)))
    if force:
        # Final comparison is a COMMON deterministic panel across every
        # trainer branch.  Field-selected calibration commands are useful
        # longitudinally, but ranking policies on different questions would
        # be scientifically invalid.  We still annotate the common controls
        # with the live field's prediction, so the same final exam also
        # measures self-map calibration.
        requested = make_common_recovery_panel(count, source='final_common_recovery')
        for spec in requested:
            pred,ev = field.predict_command(float(spec['cmd_vx']),float(spec['cmd_vy']),float(spec['cmd_yaw']))
            spec['field_predicted_competence']=pred
            spec['field_effective_evidence']=ev
    else:
        calibration_cmds = field.calibration_commands(bundles)
        if calibration_cmds:
            requested = make_calibration_panel(calibration_cmds, panel_id=panel_id,
                                                      source='field_calibration')
        else:
            requested = make_panel(count, panel_id=panel_id, source='routine_panel')
    controls = [x for x in requested if x.get('family') == 'none']
    base_key = jax.random.PRNGKey(int(EVAL_SEED) + 44000)
    control_keys = [jax.random.fold_in(base_key, int(x.get('probe_index', i)))
                    for i, x in enumerate(controls)]
    control_rows = evaluate_specs_batch(
        cloud_eval_trainer, cloud_eval_env, snapshot.params,
        snapshot.obs_mean, snapshot.obs_var ** 0.5,
        control_keys, int(CLOUD_HORIZON), controls, chunk_size=int(CLOUD_BATCH_POINTS))
    for r in control_rows:
        r['policy_step'] = int(step); r['forced_final'] = int(bool(force))

    competent_bundles = {int(r.get('bundle_id', -1)) for r in control_rows
                         if float(r.get('recovery_rate', 0.0)) >= 0.80
                         and float(r.get('survival', 0.0)) >= 0.95}
    disturbed = [x for x in requested if x.get('family') != 'none'
                 and int(x.get('bundle_id', -1)) in competent_bundles]
    disturb_keys = [jax.random.fold_in(base_key, int(x.get('probe_index', i)) + 1000003)
                    for i, x in enumerate(disturbed)]
    disturbed_rows = evaluate_specs_batch(
        cloud_eval_trainer, cloud_eval_env, snapshot.params,
        snapshot.obs_mean, snapshot.obs_var ** 0.5,
        disturb_keys, int(CLOUD_HORIZON), disturbed, chunk_size=int(CLOUD_BATCH_POINTS)) if disturbed else []
    for r in disturbed_rows:
        r['policy_step'] = int(step); r['forced_final'] = int(bool(force))

    new_rows = control_rows + disturbed_rows
    cloud_rows.extend(new_rows)
    summary = summarize_rows(cloud_rows)  # also annotates matched controls in-place
    write_rows_csv(run_dir / 'heldout_capability_cloud.csv', cloud_rows)
    cal_pairs=[(float(r.get('field_predicted_competence',np.nan)),float(r.get('node_window_rate',np.nan)))
               for r in control_rows if np.isfinite(float(r.get('field_predicted_competence',np.nan)))
               and np.isfinite(float(r.get('node_window_rate',np.nan)))]
    cal_mae=float(np.mean([abs(a-b) for a,b in cal_pairs])) if cal_pairs else -1.0
    cal_brier=float(np.mean([(a-b)**2 for a,b in cal_pairs])) if cal_pairs else -1.0
    summary.update({'policy_step': int(step), 'points': len(cloud_rows),
                    'requested_panel_points': len(requested), 'executed_panel_points': len(new_rows),
                    'envs_per_point': int(CLOUD_ENVS), 'horizon_steps': int(CLOUD_HORIZON),
                    'field_calibration_points':len(cal_pairs),'field_calibration_mae':cal_mae,
                    'field_calibration_brier':cal_brier,'field_calibration_target':'node_window_rate'})
    latest_eval['field_calibration_mae']=cal_mae
    latest_eval['field_calibration_brier']=cal_brier
    latest_eval['field_calibration_points']=float(len(cal_pairs))
    write_json(run_dir / 'heldout_capability_cloud.json', summary)
    elapsed = time.perf_counter() - t0_cloud
    perf.add('capability_cloud', elapsed)
    field_after_exam = field.fingerprint()
    if field_after_exam != field_before_exam:
        raise RuntimeError('held-out capability exam mutated the live capability field; quarantine invariant violated')

    valid=[r for r in new_rows if int(float(r.get('resilience_valid',0) or 0)) == 1]
    baseline_ok=float(len(competent_bundles) / max(1, len(control_rows)))
    mean_rec=float(np.mean([float(r['recovery_rate']) for r in valid])) if valid else 0.0
    worst=float(np.mean([float(r.get('recovery_worst_direction',0)) for r in valid])) if valid else 0.0
    print(f"{log_prefix} capability-cloud step={step:,}: controls={len(control_rows)} "
          f"disturbed={len(disturbed_rows)}/{len(requested)-len(controls)} baseline_ok={baseline_ok:.3f} "
          f"valid_recovery={mean_rec:.3f} worst_dir={worst:.3f} calibration_mae={cal_mae:.3f} wall={elapsed:.1f}s", flush=True)
