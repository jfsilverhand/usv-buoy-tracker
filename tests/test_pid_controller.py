import pytest

import pid_controller
from pid_controller import PIDController


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def clock(monkeypatch):
    c = FakeClock()
    monkeypatch.setattr(pid_controller.time, "time", c)
    return c


def step(pid, clock, error, dt):
    clock.t += dt
    return pid.compute(error)


def test_proportional_only(clock):
    pid = PIDController(kp=2.0, ki=0.0, kd=0.0)
    assert step(pid, clock, 0.5, 0.1) == pytest.approx(1.0)
    assert step(pid, clock, -0.25, 0.1) == pytest.approx(-0.5)


def test_deadzone_returns_zero_and_resets_integral(clock):
    pid = PIDController(kp=0.0, ki=1.0, kd=0.0, deadzone=0.05)
    step(pid, clock, 1.0, 0.5)
    assert pid._integral > 0
    assert step(pid, clock, 0.01, 0.5) == 0.0
    assert pid._integral == 0


def test_output_saturates(clock):
    pid = PIDController(kp=1000.0, ki=0.0, kd=0.0, output_min=-100, output_max=100)
    assert step(pid, clock, 1.0, 0.1) == 100
    assert step(pid, clock, -1.0, 0.1) == -100


def test_integral_is_clamped(clock):
    pid = PIDController(kp=0.0, ki=1.0, kd=0.0, output_max=1000)
    for _ in range(500):
        out = step(pid, clock, 1.0, 0.5)
    assert out == pytest.approx(50.0)  # integral clamp (anti-windup)


def test_derivative_term(clock):
    pid = PIDController(kp=0.0, ki=0.0, kd=1.0)
    assert step(pid, clock, 0.5, 0.1) == pytest.approx(5.0)   # (0.5 - 0) / 0.1
    assert step(pid, clock, 0.5, 0.1) == pytest.approx(0.0)   # error unchanged


def test_large_time_gap_uses_nominal_dt(clock):
    pid = PIDController(kp=0.0, ki=0.0, kd=1.0)
    assert step(pid, clock, 0.33, 5.0) == pytest.approx(0.33 / 0.033)


def test_reset_clears_state(clock):
    pid = PIDController(kp=0.0, ki=1.0, kd=1.0)
    step(pid, clock, 1.0, 0.1)
    pid.reset()
    assert pid._integral == 0.0 and pid._prev_error == 0.0
