"""Current EMU curriculum controller.

This module owns the task distribution only: regime difficulty/allocation, core
retreat, and demonstrated physical frontiers. PPO/training health and rollback
decisions live in health.py.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields as dataclass_fields
from typing import Any

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
    # hysteresis; rollout_metrics.py owns the actual tracking tolerances.
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
            'frontier_precision_ema': dict(self.frontier_precision_ema),
            'frontier_bad_streak': dict(self.frontier_bad_streak),
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        expected = {
            'params', 'competence', 'core_competence', 'best_len', 'core_bad_streak',
            'fall_ema', '_falls_ema', '_episodes_ema', '_updates_done_total',
            'frontier_precision_ema', 'frontier_bad_streak',
        }
        if not isinstance(d, dict) or set(d) != expected:
            raise ValueError(f"curriculum state keys mismatch: expected={sorted(expected)} got={sorted(d) if isinstance(d, dict) else type(d).__name__}")
        param_names = {f.name for f in dataclass_fields(RegimeParams)}
        if not isinstance(d['params'], dict) or set(d['params']) != param_names:
            got = sorted(d['params']) if isinstance(d['params'], dict) else type(d['params']).__name__
            raise ValueError(f"curriculum parameter keys mismatch: expected={sorted(param_names)} got={got}")
        expected_regimes = set(self.REGIMES)
        expected_falls = expected_regimes | {'posture'}
        expected_frontier = {'vx_hi', 'vx_lo', 'vy', 'yaw', 'combo', 'accel'}
        for name, keys in (('competence', expected_regimes), ('fall_ema', expected_falls),
                           ('_falls_ema', expected_falls), ('_episodes_ema', expected_falls),
                           ('frontier_precision_ema', expected_frontier), ('frontier_bad_streak', expected_frontier)):
            if not isinstance(d[name], dict) or set(d[name]) != keys:
                raise ValueError(f"curriculum {name} keys mismatch")
        self.params = RegimeParams(**d['params'])
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


