from . import BaseController
import numpy as np


class Controller(BaseController):
  """
  Preview feedforward with a PID feedback trim.

  The feedforward term uses the visible future target and road roll to start
  steering before the current-error PID loop needs to react.
  """

  def __init__(self):
    self.u_min = -2.0
    self.u_max = 2.0
    self.max_delta = 0.12

    self.k_ff_target = 0.55
    self.k_ff_roll = -0.55
    self.k_ff_slope = 0.10
    self.k_ff_accel = 0.0

    self.k_p = 0.195
    self.k_i = 0.100
    self.k_d = -0.053
    self.integral_decay = 0.98
    self.integral_limit = 20.0

    #weigh the next 5 preview points with decreasing weight
    self.preview_weights = np.array([0.42, 0.28, 0.18, 0.08, 0.04], dtype=float)
    self.prev_error = 0.0
    self.error_integral = 0.0
    self.prev_action = 0.0

  def _clip_action(self, action):
    return float(np.clip(action, self.u_min, self.u_max))

  def _slew_limit(self, action):
    #clip action if it exceeds the max delta from the previous action
    delta = np.clip(action - self.prev_action, -self.max_delta, self.max_delta)
    return self._clip_action(self.prev_action + delta)

  def _pad(self, values, fallback, length):
    #pads to desired length with last value or fallback if empty
    seq = list(values[:length]) if values else []
    if not seq:
      seq = [float(fallback)]
    while len(seq) < length:
      seq.append(seq[-1])
    return np.asarray(seq[:length], dtype=float)

  def _preview(self, values, fallback):
    seq = self._pad(values, fallback, len(self.preview_weights))
    return float(np.dot(seq, self.preview_weights))

  def _feedforward(self, target_lataccel, state, future_plan):
    target_now = float(target_lataccel)
    target_preview = self._preview(getattr(future_plan, "lataccel", []), target_now)
    roll_preview = self._preview(
      getattr(future_plan, "roll_lataccel", []),
      float(getattr(state, "roll_lataccel", 0.0)),
    )
    accel = float(getattr(state, "a_ego", 0.0))
    target_slope = target_preview - target_now

    return (
      self.k_ff_target * target_preview
      + self.k_ff_roll * roll_preview
      + self.k_ff_slope * target_slope
      + self.k_ff_accel * accel
    )

  def _pid_trim(self, target_lataccel, current_lataccel):

    #typical PID control with clipping
    error = float(target_lataccel) - float(current_lataccel)
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

    return (
      self.k_p * error
      + self.k_i * self.error_integral
      + self.k_d * error_diff
    )

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    #return steer action based on ff and feedback trim

    #ff is planned steering based on future target and road roll
    #feedback trim is PID based on current error
    #slew limit is applied to the final action to avoid large jumps in steering
    action = (
      self._feedforward(target_lataccel, state, future_plan)
      + self._pid_trim(target_lataccel, current_lataccel)
    )

    action = self._slew_limit(action)
    self.prev_action = action
    return action
