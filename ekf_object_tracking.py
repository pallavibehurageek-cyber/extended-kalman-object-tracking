"""
3-D Object Tracking using an Extended Kalman Filter (EKF)

A moving aircraft observes a ground target through
noisy range/azimuth/elevation measurements. The EKF estimates target position
and velocity and produces diagnostic plots.


"""

import numpy as np
import matplotlib.pyplot as plt


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def state_transition(dt: float) -> np.ndarray:
    """Constant-velocity 6-state transition matrix."""
    F = np.eye(6)
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt
    return F


def process_noise(dt: float, accel_std: float) -> np.ndarray:
    """Discrete white-acceleration process noise."""
    q = accel_std ** 2
    G = np.array([
        [0.5 * dt**2, 0, 0],
        [0, 0.5 * dt**2, 0],
        [0, 0, 0.5 * dt**2],
        [dt, 0, 0],
        [0, dt, 0],
        [0, 0, dt],
    ])
    return q * (G @ G.T)


def measurement_model(x: np.ndarray, sensor_pos: np.ndarray) -> np.ndarray:
    """Nonlinear spherical measurement: range, azimuth, elevation."""
    dx, dy, dz = x[:3] - sensor_pos
    horizontal = max(np.hypot(dx, dy), 1e-9)
    rho = np.sqrt(dx * dx + dy * dy + dz * dz)
    az = np.arctan2(dy, dx)
    el = np.arctan2(dz, horizontal)
    return np.array([rho, az, el])


def measurement_jacobian(x: np.ndarray, sensor_pos: np.ndarray) -> np.ndarray:
    """Jacobian of [range, azimuth, elevation] w.r.t. the 6-state."""
    dx, dy, dz = x[:3] - sensor_pos
    rho2 = max(dx * dx + dy * dy + dz * dz, 1e-12)
    rho = np.sqrt(rho2)
    r2 = max(dx * dx + dy * dy, 1e-12)
    r = np.sqrt(r2)

    H = np.zeros((3, 6))
    # d(range)/d(position)
    H[0, 0] = dx / rho
    H[0, 1] = dy / rho
    H[0, 2] = dz / rho

    # d(azimuth)/d(position)
    H[1, 0] = -dy / r2
    H[1, 1] = dx / r2

    # d(elevation)/d(position)
    H[2, 0] = -dx * dz / (rho2 * r)
    H[2, 1] = -dy * dz / (rho2 * r)
    H[2, 2] = r / rho2
    return H


def make_aircraft_path(t: np.ndarray) -> np.ndarray:
    """Smooth 3-D aircraft trajectory used only to generate synthetic data."""
    x = 1800.0 + 220.0 * np.cos(0.07 * t)
    y = -700.0 + 320.0 * np.sin(0.07 * t)
    z = 1100.0 + 80.0 * np.sin(0.05 * t)
    return np.column_stack((x, y, z))


def run_ekf(seed: int = 7):
    rng = np.random.default_rng(seed)

    n = 120
    dt = 1.0
    t = np.arange(n) * dt

    # Ground target: slightly moving so the velocity estimate is meaningful.
    target_true = np.column_stack((
        1500.0 + 0.15 * t,
        800.0 + 0.10 * t,
        0.0 + 0.02 * t,
    ))
    target_velocity = np.gradient(target_true, axis=0) / dt

    aircraft = make_aircraft_path(t)

    # Sensor standard deviations.
    sigma_range = 2.0          # metres
    sigma_az = np.deg2rad(0.20)
    sigma_el = np.deg2rad(0.20)
    R = np.diag([sigma_range**2, sigma_az**2, sigma_el**2])

    z = np.zeros((n, 3))
    for k in range(n):
        ideal = measurement_model(
            np.r_[target_true[k], target_velocity[k]], aircraft[k]
        )
        z[k] = ideal + rng.normal(0.0, [sigma_range, sigma_az, sigma_el])
        z[k, 1] = wrap_angle(z[k, 1])
        z[k, 2] = np.clip(z[k, 2], -np.pi / 2 + 1e-5, np.pi / 2 - 1e-5)

    # Initial state from the first spherical measurement.
    rho, az, el = z[0]
    xy = rho * np.cos(el)
    initial_pos = aircraft[0] + np.array([
        xy * np.cos(az),
        xy * np.sin(az),
        rho * np.sin(el),
    ])
    xhat = np.r_[initial_pos, np.zeros(3)]
    P = np.diag([30.0**2, 30.0**2, 30.0**2, 10.0**2, 10.0**2, 10.0**2])

    F = state_transition(dt)
    Q = process_noise(dt, accel_std=0.5)

    estimates = np.zeros((n, 6))
    pos_sigma = np.zeros(n)
    raw_error = np.zeros(n)
    filt_error = np.zeros(n)

    for k in range(n):
        # Predict
        xhat = F @ xhat
        P = F @ P @ F.T + Q

        # Update
        h = measurement_model(xhat, aircraft[k])
        H = measurement_jacobian(xhat, aircraft[k])
        innovation = z[k] - h
        innovation[1] = wrap_angle(innovation[1])
        innovation[2] = wrap_angle(innovation[2])
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        xhat = xhat + K @ innovation
        P = (np.eye(6) - K @ H) @ P
        P = 0.5 * (P + P.T)

        estimates[k] = xhat
        pos_sigma[k] = float(np.sqrt(max(np.trace(P[:3, :3]), 0.0)))

        measured_rho, measured_az, measured_el = z[k]
        xy = measured_rho * np.cos(measured_el)
        raw_pos = aircraft[k] + np.array([
            xy * np.cos(measured_az),
            xy * np.sin(measured_az),
            measured_rho * np.sin(measured_el),
        ])
        raw_error[k] = np.linalg.norm(raw_pos - target_true[k])
        filt_error[k] = np.linalg.norm(xhat[:3] - target_true[k])

    return t, aircraft, target_true, z, estimates, pos_sigma, raw_error, filt_error


def main() -> None:
    t, aircraft, truth, z, est, sigma, raw_error, filt_error = run_ekf()
    target_velocity = np.gradient(truth, axis=0)

    fig, ax = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle(
        "3-D Object Tracking with Extended Kalman Filter (EKF)",
        fontweight="bold",
    )

    # A: position estimate
    ax[0, 0].plot(t, est[:, 0], label="EKF X")
    ax[0, 0].plot(t, est[:, 1], label="EKF Y")
    ax[0, 0].plot(t, est[:, 2], label="EKF Z")
    ax[0, 0].plot(t, truth[:, 0], "--", linewidth=1, label="True X")
    ax[0, 0].set_title("A  Per-axis position estimate")
    ax[0, 0].set_xlabel("Update step")
    ax[0, 0].set_ylabel("Position (m)")
    ax[0, 0].legend(fontsize=8)
    ax[0, 0].grid(True, alpha=0.25)

    # B: uncertainty
    ax[0, 1].plot(t, sigma, label="Position 1-sigma")
    ax[0, 1].axhline(5.0, linestyle="--", label="5 m reference")
    ax[0, 1].set_title("B  Filter uncertainty")
    ax[0, 1].set_xlabel("Update step")
    ax[0, 1].set_ylabel("3-D sigma (m)")
    ax[0, 1].legend(fontsize=8)
    ax[0, 1].grid(True, alpha=0.25)

    # C: raw vs filtered error
    ax[0, 2].plot(t, raw_error, label="Raw sensor error")
    ax[0, 2].plot(t, filt_error, label="EKF estimate error")
    ax[0, 2].axhline(5.0, linestyle="--", label="5 m reference")
    ax[0, 2].set_title("C  Raw vs filtered position error")
    ax[0, 2].set_xlabel("Update step")
    ax[0, 2].set_ylabel("3-D error (m)")
    ax[0, 2].legend(fontsize=8)
    ax[0, 2].grid(True, alpha=0.25)

    # D: estimated velocity
    ax[1, 0].plot(t, est[:, 3], label="vx")
    ax[1, 0].plot(t, est[:, 4], label="vy")
    ax[1, 0].plot(t, est[:, 5], label="vz")
    ax[1, 0].plot(t, target_velocity[:, 0], "--", linewidth=1)
    ax[1, 0].plot(t, target_velocity[:, 1], "--", linewidth=1)
    ax[1, 0].plot(t, target_velocity[:, 2], "--", linewidth=1)
    ax[1, 0].set_title("D  Estimated target velocity")
    ax[1, 0].set_xlabel("Update step")
    ax[1, 0].set_ylabel("Velocity (m/s)")
    ax[1, 0].legend(fontsize=8)
    ax[1, 0].grid(True, alpha=0.25)

    # E: 3-D trajectory
    ax[1, 1].plot(truth[:, 0], truth[:, 1], label="True trajectory")
    ax[1, 1].plot(est[:, 0], est[:, 1], label="EKF trajectory")
    ax[1, 1].plot(aircraft[:, 0], aircraft[:, 1], label="Aircraft path")
    ax[1, 1].set_title("E  Ground-track trajectory")
    ax[1, 1].set_xlabel("X (m)")
    ax[1, 1].set_ylabel("Y (m)")
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].grid(True, alpha=0.25)
    ax[1, 1].axis("equal")

    # F: convergence statistics
    ax[1, 2].plot(t, np.cumsum(filt_error < 5.0), label="Within 5 m")
    ax[1, 2].plot(t, np.arange(1, len(t) + 1), "--", label="All updates")
    ax[1, 2].set_title("F  Cumulative within-spec updates")
    ax[1, 2].set_xlabel("Update step")
    ax[1, 2].set_ylabel("Count")
    ax[1, 2].legend(fontsize=8)
    ax[1, 2].grid(True, alpha=0.25)

    plt.tight_layout()
    plt.show()

    print(f"Final EKF position error: {filt_error[-1]:.2f} m")
    print(f"Final 1-sigma position uncertainty: {sigma[-1]:.2f} m")
    print(f"Updates within 5 m: {(filt_error < 5.0).sum()}/{len(t)}")


if __name__ == "__main__":
    main()
