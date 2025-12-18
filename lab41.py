import numpy as np
import matplotlib.pyplot as plt

n = 14
N = 100 * n
T_values = [4, 8, 16, 32, 64, 128]
num_points = 20001
k_max = 300

t = np.linspace(-N, N, num_points)
dt = t[1] - t[0]

def f(t):
    return np.abs(t) ** ((2 * n + 1) / 3)

def fourier_coefficient(k, T):
    omega_k = 2 * np.pi * k / T
    integrand = f(t) * np.exp(-1j * omega_k * t)
    F = np.sum(integrand) * dt
    return np.real(F), np.imag(F), omega_k

axis_settings = {
    4:  (30, 1),
    8:  (20, 1),
    16: (10, 1),
    32: (5, 0.5),
    64: (2, 0.2),
    128:(1, 0.1)
}

for T in T_values:
    Re_list = []
    Amp_list = []
    omega_list = []

    omega_max, step = axis_settings[T]

    for k in range(k_max + 1):
        Re, Im, omega_k = fourier_coefficient(k, T)
        if omega_k > omega_max:
            break

        omega_list.append(omega_k)
        Re_list.append(Re)
        Amp_list.append(np.sqrt(Re**2 + Im**2))

    xticks = np.arange(0, omega_max + step, step)

    plt.figure()
    plt.plot(omega_list, Re_list, marker='o')
    plt.xlim(0, omega_max)
    plt.xticks(xticks)
    plt.xlabel("ωₖ")
    plt.ylabel("Re F(ωₖ)")
    plt.title(f"Re F(ωₖ) при T = {T}")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(omega_list, Amp_list, marker='o')
    plt.xlim(0, omega_max)
    plt.xticks(xticks)
    plt.xlabel("ωₖ")
    plt.ylabel("|F(ωₖ)|")
    plt.title(f"|F(ωₖ)| при T = {T}")
    plt.grid(True)
    plt.show()

print("END OF PROGRAM")
