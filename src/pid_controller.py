"""
PID Controller - Gimbal hedef takibi icin
"""
import time

class PIDController:
    def __init__(self, kp=0.3, ki=0.01, kd=0.05,
                 output_min=-100, output_max=100,
                 deadzone=0.02):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        self.deadzone = deadzone
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = time.time()
        self._integral_max = 50.0
    
    def compute(self, error):
        now = time.time()
        dt = now - self._prev_time
        self._prev_time = now
        
        if dt <= 0 or dt > 1.0:
            dt = 0.033
        
        if abs(error) < self.deadzone:
            self._integral = 0
            return 0.0
        
        p = self.kp * error
        
        self._integral += error * dt
        self._integral = max(-self._integral_max,
                            min(self._integral_max, self._integral))
        i = self.ki * self._integral
        
        d = self.kd * (error - self._prev_error) / dt if dt > 0 else 0
        self._prev_error = error
        
        output = p + i + d
        return max(self.output_min, min(self.output_max, output))
    
    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = time.time()
