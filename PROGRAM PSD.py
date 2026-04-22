import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint

# =========================
# PARAMETER SISTEM
# =========================
m = 1
c = 2
k = 5

# =========================
# PARAMETER PID
# =========================
Kp = 50
Ki = 10
Kd = 20

# =========================
# WAKTU SIMULASI
# =========================
t = np.linspace(0, 20, 1000)

# =========================
# SISTEM TANPA KONTROL
# =========================
def system_no_control(x, t):
    x1, x2 = x
    dx1dt = x2
    dx2dt = (-c * x2 - k * x1) / m
    return [dx1dt, dx2dt]

# kondisi awal
x0 = [1, 0]

# simulasi tanpa kontrol
sol_no = odeint(system_no_control, x0, t)

# =========================
# SISTEM DENGAN PID
# =========================
def system_pid(x, t):
    x1, x2, e_int = x
    setpoint = 0
    
    # error
    e = setpoint - x1
    
    # kontrol PID
    u = Kp * e + Ki * e_int - Kd * x2
    
    dx1dt = x2
    dx2dt = (u - c * x2 - k * x1) / m
    de_int_dt = e
    
    return [dx1dt, dx2dt, de_int_dt]

# kondisi awal PID
x0_pid = [1, 0, 0]

# simulasi PID
sol_pid = odeint(system_pid, x0_pid, t)

# ambil posisi
x_no = sol_no[:, 0]
x_pid = sol_pid[:, 0]

# =========================
# TABEL HASIL
# =========================
sample_times = [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]

data = []
for ts in sample_times:
    no_val = np.interp(ts, t, x_no)
    pid_val = np.interp(ts, t, x_pid)
    data.append([ts, no_val, pid_val])

df = pd.DataFrame(data, columns=["Waktu (s)", "Tanpa Kontrol", "Dengan PID"])

print("\n=== TABEL HASIL SIMULASI ===")
print(df)

# =========================
# GRAFIK
# =========================
plt.figure(figsize=(10,6))

plt.plot(t, x_no, label='Tanpa Kontrol')
plt.plot(t, x_pid, label='Dengan PID')

plt.xlabel('Waktu (detik)')
plt.ylabel('Posisi x(t)')
plt.title('Respon Sistem Massa–Pegas–Redaman')

plt.legend()
plt.grid(True)

plt.show()
