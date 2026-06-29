from . import BaseController
import itertools
import numpy as np


class Controller(BaseController):
  """
  Guarded per-rollout MPC controller.

  The controller first builds a strong preview-PID nominal action, then lets a
  short-horizon MPC search make only small corrections around that nominal
  plan. This is intentionally conservative because the controller API does not
  expose the real tinyphysics model inside update().
  """

  def __init__(self):
    self.u_min = -2.0
    self.u_max = 2.0
    self.max_delta = 0.12

    self.horizon = 7
    self.sequence_len = 3
    self.dt = 0.1

    self.k_ff_target = 0.55
    self.k_ff_roll = -0.55
    self.k_ff_slope = 0.10
    self.k_p = 0.195
    self.k_i = 0.100
    self.k_d = -0.053
    self.integral_decay = 0.98
    self.integral_limit = 20.0
    self.preview_weights = np.array([0.42, 0.28, 0.18, 0.08, 0.04], dtype=float)

    self.plant_gain = 1.82
    self.base_alpha = 0.42

    self.track_weight = 52.0
    self.jerk_weight = 1.25
    self.action_weight = 0.05
    self.delta_weight = 8.0
    self.deviation_weight = 18.0
    self.terminal_weight = 10.0

    self.mpc_blend = 0.22
    self.max_mpc_deviation = 0.08
    self.search_offsets = np.array([-0.08, -0.04, 0.0, 0.04, 0.08], dtype=float)

    self.prev_action = 0.0
    self.prev_error = 0.0
    self.error_integral = 0.0

  def _clip_action(self, action):
    return float(np.clip(action, self.u_min, self.u_max))

  def _slew_limit_from(self, action, previous):
    delta = np.clip(action - previous, -self.max_delta, self.max_delta)
    return self._clip_action(previous + delta)

  def _pad(self, values, fallback, length):
    seq = list(values[:length]) if values else []
    if not seq:
      seq = [float(fallback)]
    while len(seq) < length:
      seq.append(seq[-1])
    return np.asarray(seq[:length], dtype=float)

  def _preview(self, values, fallback):
    seq = self._pad(values, fallback, len(self.preview_weights))
    return float(np.dot(seq, self.preview_weights))

  def _future_array(self, now, future_values):
    return self._pad([now] + list(future_values[:self.horizon - 1]), now, self.horizon)

  def _alpha(self, v_ego):
    v = float(np.clip(v_ego, 5.0, 40.0))
    return float(np.clip(self.base_alpha * (24.0 / (v + 8.0)), 0.22, 0.50))

  def _preview_pid_action(self, target_lataccel, current_lataccel, state, future_plan):
    target_now = float(target_lataccel)
    roll_now = float(getattr(state, "roll_lataccel", 0.0))
    target_preview = self._preview(getattr(future_plan, "lataccel", []), target_now)
    roll_preview = self._preview(getattr(future_plan, "roll_lataccel", []), roll_now)
    target_slope = target_preview - target_now
    feedforward = (
      self.k_ff_target * target_preview
      + self.k_ff_roll * roll_preview
      + self.k_ff_slope * target_slope
    )

    error = target_now - float(current_lataccel)
    self.error_integral = (
      self.integral_decay * self.error_integral
      + error
    )
    self.error_integral = float(np.clip(
      self.error_integral,
      -self.integral_limit,
      self.integral_limit,
    ))
    error_diff = error - self.prev_error
    self.prev_error = error
    feedback = (
      self.k_p * error
      + self.k_i * self.error_integral
      + self.k_d * error_diff
    )

    return self._slew_limit_from(feedforward + feedback, self.prev_action)

  def _nominal_actions(self, targets, rolls, first_action):
    slopes = np.diff(np.r_[targets, targets[-1]])
    actions = (
      self.k_ff_target * targets
      + self.k_ff_roll * rolls
      + self.k_ff_slope * slopes
    )
    actions[0] = first_action

    limited = np.empty_like(actions)
    previous = self.prev_action
    for i, action in enumerate(actions):
      limited[i] = self._slew_limit_from(action, previous)
      previous = limited[i]
    return limited

  def _candidate_sequences(self, nominal):
    choices = []
    previous = self.prev_action
    for i in range(self.sequence_len):
      centered = nominal[i] + self.search_offsets
      limited = [self._slew_limit_from(action, previous) for action in centered]
      choices.append(np.unique(np.round(limited, 4)))
      previous = nominal[i]
    return itertools.product(*choices)

  def _simulate(self, sequence, current_lataccel, targets, rolls, v_egos, nominal):
    lat = float(current_lataccel)
    prev_lat = lat
    prev_action = self.prev_action
    cost = 0.0

    for i in range(self.horizon):
      desired = sequence[i] if i < len(sequence) else nominal[i]
      action = self._slew_limit_from(desired, prev_action)

      alpha = self._alpha(v_egos[i])
      steady_lataccel = self.plant_gain * action + rolls[i]
      lat = lat + alpha * (steady_lataccel - lat)

      jerk = (lat - prev_lat) / self.dt
      error = targets[i] - lat
      deviation = action - nominal[i]
      delta = action - prev_action

      cost += self.track_weight * error * error
      cost += self.jerk_weight * jerk * jerk
      cost += self.action_weight * action * action
      cost += self.delta_weight * delta * delta
      cost += self.deviation_weight * deviation * deviation

      prev_lat = lat
      prev_action = action

    terminal_error = targets[-1] - lat
    return cost + self.terminal_weight * terminal_error * terminal_error

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    baseline_action = self._preview_pid_action(
      target_lataccel,
      current_lataccel,
      state,
      future_plan,
    )

    roll_now = float(getattr(state, "roll_lataccel", 0.0))
    v_now = float(getattr(state, "v_ego", 20.0))
    targets = self._future_array(
      float(target_lataccel),
      getattr(future_plan, "lataccel", []),
    )
    rolls = self._future_array(
      roll_now,
      getattr(future_plan, "roll_lataccel", []),
    )
    v_egos = self._future_array(
      v_now,
      getattr(future_plan, "v_ego", []),
    )

    nominal = self._nominal_actions(targets, rolls, baseline_action)
    best_sequence = min(
      self._candidate_sequences(nominal),
      key=lambda sequence: self._simulate(
        sequence,
        current_lataccel,
        targets,
        rolls,
        v_egos,
        nominal,
      ),
    )

    mpc_action = float(best_sequence[0])
    mpc_action = float(np.clip(
      mpc_action,
      baseline_action - self.max_mpc_deviation,
      baseline_action + self.max_mpc_deviation,
    ))
    action = (
      (1.0 - self.mpc_blend) * baseline_action
      + self.mpc_blend * mpc_action
    )
    action = self._slew_limit_from(action, self.prev_action)

    self.prev_action = action
    return action
