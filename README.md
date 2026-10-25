# Orbital Dynamics Lab

A Python project for orbital mechanics calculations and interactive 3D visualization.
The project provides conversions between Cartesian state vectors and Keplerian
orbital elements, along with visualization of elliptical orbits around Earth.
The architecture is designed to support future numerical orbit propagation, force models, and orbital maneuvers.

## Technology Stack

- Python 3.11–3.12
- NumPy
- PyOrb
- SciPy (for future numerical propagation)
- PySide6
- PyVista
- PyVistaQt
- uv for dependency management

## Installation

Clone the repository:
    git clone https://github.com/alexid4774/Theoretical-mechanics.git
    cd Theoretical-mechanics
Install dependencies:
    uv sync
The project uses pyproject.toml and uv.lock for dependency management.

## Running the Applications

### Orbital Element Converter

    uv run orbit-converter

All calculations in the computational core use SI units.

### Orbit Visualizer

    uv run orbit-visualizer

Supported eccentricity range:

    0 <= e <= 0.99
