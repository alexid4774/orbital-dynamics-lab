"""Conversions between Cartesian and Keplerian orbital elements. All calculations use SI units.
Keplerian elements are ordered as:[a, e, i, omega, raan, nu]."""

import numpy as np
import pyorb


def _validate_mu(mu: float) -> None:

    """Validate the gravitational parameter.
    Args:
        mu: Standard gravitational parameter in m^3/s^2.
    Raises:
        ValueError: If mu is not finite or positive.
    """

    if not np.isfinite(mu) or mu <= 0:
        raise ValueError("Gravitational parameter mu must be finite and positive.")


def cartesian_to_keplerian(state: np.ndarray, mu: float) -> np.ndarray:

    """Convert a Cartesian state to Keplerian orbital elements.
    Args:
        state: [x, y, z, vx, vy, vz] in SI units.
        mu: Gravitational parameter in m^3/s^2.
    Returns:
        Array [a, e, i, omega, raan, nu] in SI units and radians.
    Raises:
        ValueError: If the input is invalid or the orbit is degenerate.
    """

    _validate_mu(mu)
    state = np.asarray(state, dtype = float)

    if state.shape != (6,):
        raise ValueError("Cartesian state must contain exactly six values.")

    if not np.all(np.isfinite(state)):
        raise ValueError("Cartesian state must contain finite values.")

    r = state[:3]
    v = state[3:]

    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)

    if r_norm == 0:
        raise ValueError("Position vector must not be zero.")

    h_norm = np.linalg.norm(np.cross(r, v))

    if h_norm <= 1e-12 * r_norm * v_norm:
        raise ValueError("Radial trajectories are not supported.")

    elements = np.asarray(pyorb.cart_to_kep(state, mu = mu, degrees = False), dtype = float)

    if elements.shape != (6,) or not np.all(np.isfinite(elements)):
        raise ValueError("Failed to calculate finite Keplerian elements.")

    if elements[1] >= 1:
        raise ValueError("Only elliptical orbits (e < 1) are currently supported.")

    return elements


def keplerian_to_cartesian(elements: np.ndarray, mu: float) -> np.ndarray:

    """Convert Keplerian orbital elements to a Cartesian state.
    Args:
        elements: [a, e, i, omega, raan, nu].
            Distances are in metres and angles in radians.
        mu: Gravitational parameter in m^3/s^2.
    Returns:
        Cartesian state [x, y, z, vx, vy, vz] in SI units.
    Raises:
        ValueError: If orbital elements are invalid.
    """

    _validate_mu(mu)

    elements = np.asarray(elements, dtype=float)

    if elements.shape != (6,):
        raise ValueError("Keplerian elements must contain exactly six values.")

    if not np.all(np.isfinite(elements)):
        raise ValueError("Keplerian elements must contain finite values.")

    a, e, inclination, _, _, _ = elements

    if a <= 0:
        raise ValueError("Semi-major axis must be positive.")

    if not 0 <= e < 1:
        raise ValueError("Only elliptical orbits (0 <= e < 1) are supported.")

    if not 0 <= inclination <= np.pi:
        raise ValueError("Inclination must be between 0 and pi radians.")

    state = np.asarray(pyorb.kep_to_cart(elements, mu=mu, degrees=False), dtype = float)

    if state.shape != (6,) or not np.all(np.isfinite(state)):
        raise ValueError("Failed to calculate a finite Cartesian state.")

    return state


def sample_orbit(elements: np.ndarray, mu: float, samples: int = 181) -> np.ndarray:

    """Generate positions along an elliptical orbit using vectorized PyOrb.
    Args:
        elements: Keplerian elements [a, e, i, omega, raan, nu].
            Distances are in metres and angles in radians.
        mu: Gravitational parameter in m^3/s^2.
        samples: Number of points along the orbit.
    Returns:
        Array of shape (samples, 3) with positions in metres.
    Raises:
        ValueError: If the input parameters are invalid.
    """

    if (isinstance(samples, (bool, np.bool_)) or not isinstance(samples, (int, np.integer)) or samples < 2):
        raise ValueError("Number of samples must be an integer >= 2.")

    elements = np.asarray(elements, dtype=float)

    keplerian_to_cartesian(elements, mu)

    orbital_elements = np.repeat(elements[:, np.newaxis], samples, axis = 1)

    orbital_elements[5, :] = np.linspace(0.0, 2.0 * np.pi, samples)

    states = np.asarray(pyorb.kep_to_cart(orbital_elements, mu = mu, degrees = False), dtype = float)

    if states.shape != (6, samples) or not np.all(np.isfinite(states)):
        raise ValueError("Failed to generate finite orbital positions.")

    return states[:3, :].T.copy()


def describe_orbital_angles(elements: np.ndarray, eccentricity_tolerance: float = 1e-8, inclination_tolerance: float = 1e-8) -> dict[str, float | str]:

    """Describe physically meaningful orbital angles.
    Args:
        elements: Keplerian elements [a, e, i, omega, raan, nu].
            Angles are in radians.
        eccentricity_tolerance: Threshold for treating an orbit as circular.
        inclination_tolerance: Threshold for treating an orbit as equatorial.
    Returns:
        Dictionary containing the orbit type and meaningful angles.
    Raises:
        ValueError: If orbital elements or tolerances are invalid.
    """

    elements = np.asarray(elements, dtype=float)

    if elements.shape != (6,) or not np.all(np.isfinite(elements)):
        raise ValueError("Expected six finite Keplerian elements.")

    if (not np.isfinite(eccentricity_tolerance) or eccentricity_tolerance <= 0 or not np.isfinite(inclination_tolerance)
        or inclination_tolerance <= 0):
        raise ValueError("Tolerances must be finite and positive.")

    _, e, i_rad, omega_rad, raan_rad, nu_rad = elements

    if e < 0 or not 0 <= i_rad <= np.pi:
        raise ValueError("Invalid eccentricity or inclination.")

    circular = e <= eccentricity_tolerance
    equatorial = min(i_rad, abs(np.pi - i_rad)) <= inclination_tolerance

    def normalized(angle: float) -> float:
        return float(angle % (2.0 * np.pi))

    if circular and equatorial:
        return {"type": "circular_equatorial", "true_longitude_rad": normalized(raan_rad + omega_rad + nu_rad)}

    if circular:
        return {"type": "circular_inclined", "raan_rad": normalized(raan_rad), "argument_of_latitude_rad": normalized(omega_rad + nu_rad)}

    if equatorial:
        return {"type": "elliptical_equatorial", "longitude_of_periapsis_rad": normalized(raan_rad + omega_rad),
            "true_anomaly_rad": normalized(nu_rad)}

    return {"type": "elliptical_inclined", "argument_of_periapsis_rad": normalized(omega_rad),
        "raan_rad": normalized(raan_rad), "true_anomaly_rad": normalized(nu_rad)}
