import os
os.environ.setdefault('QT_API', 'pyside6')

import math
import sys

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication, QWidget, QHBoxLayout, QVBoxLayout, QFormLayout,
    QLabel, QSlider, QDoubleSpinBox, QPushButton, QGroupBox, QMessageBox)
from orbital_dynamics.conversions import keplerian_to_cartesian, sample_orbit
from orbital_dynamics.constants import EARTH_MU, EARTH_RADIUS


EARTH_RADIUS_KM = EARTH_RADIUS / 1000


class OrbitVisualizer(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Интерактивная визуализация орбиты')
        self.resize(1180, 740)

        layout = QHBoxLayout(self)
        self.plotter = QtInteractor(self)
        self.plotter.set_background('#0c1425')
        layout.addWidget(self.plotter.interactor, 3)

        controls = QVBoxLayout()
        layout.addLayout(controls, 1)
        controls.addWidget(QLabel('Кеплеровы элементы орбиты'))

        group = QGroupBox('Изменение параметров')
        form = QFormLayout(group)
        controls.addWidget(group)
        self.sliders = {}
        self.spins = {}

        specs = [('a', 'Большая полуось, км', 7000, 6800, 30000, 10, 0),
            ('e', 'Эксцентриситет', 0.05, 0.0, 0.99, 0.01, 2),
            ('i', 'Наклонение, °', 30, 0, 180, 1, 0),
            ('omega', 'Аргумент перицентра, °', 45, 0, 360, 1, 0),
            ('raan', 'Долгота восх. узла, °', 60, 0, 360, 1, 0),
            ('nu', 'Истинная аномалия, °', 90, 0, 360, 1, 0)]
        
        for key, label, default, low, high, step, decimals in specs:
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(round(low / step))
            slider.setMaximum(round(high / step))
            slider.setValue(round(default / step))
            spin = QDoubleSpinBox()
            spin.setRange(low, high)
            spin.setSingleStep(step)
            spin.setDecimals(decimals)
            spin.setValue(default)
            slider.valueChanged.connect(lambda n, s = spin, scale = step: s.setValue(n * scale))
            spin.valueChanged.connect(lambda n, s = slider, scale = step: s.setValue(round(n / scale)))
            spin.valueChanged.connect(self.update_orbit)
            row = QVBoxLayout()
            row.addWidget(spin)
            row.addWidget(slider)
            form.addRow(label, row)
            self.sliders[key] = slider
            self.spins[key] = spin

        self.status = QLabel()
        self.status.setWordWrap(True)
        controls.addWidget(self.status)
        reset = QPushButton('Вернуть начальные значения')
        reset.clicked.connect(self.reset_values)
        controls.addWidget(reset)
        controls.addStretch()

        earth = pv.Sphere(radius = EARTH_RADIUS_KM, theta_resolution = 48, phi_resolution = 48)
        self.plotter.add_mesh(earth, color = '#2768ad', smooth_shading = True, name = 'earth')
        self.plotter.add_axes()
        self.update_orbit()
        self.plotter.view_isometric()
        self.plotter.reset_camera()

    def elements(self, nu = None):
        p = self.spins
        return np.array([p['a'].value() * 1000,
            p['e'].value(),
            math.radians(p['i'].value()),
            math.radians(p['omega'].value()),
            math.radians(p['raan'].value()),
            math.radians(p['nu'].value() if nu is None else nu),], dtype = float)

    def draw_trajectory(self, positions_m: np.ndarray) -> None:

        """Draw a trajectory from Cartesian positions in metres.
        """

        positions_m = np.asarray(positions_m, dtype = float)

        if (positions_m.ndim != 2 or positions_m.shape[1] != 3 or len(positions_m) < 2
            or not np.all(np.isfinite(positions_m))):
            raise ValueError("Trajectory must contain at least two finite 3D positions.")

        positions_km = positions_m / 1000.0

        trajectory = pv.lines_from_points(positions_km, close = False)

        self.plotter.add_mesh(trajectory, color = "#54d9f0", line_width = 4, name = "orbit")

    def draw_satellite(self, position_m: np.ndarray) -> None:

        """Draw a satellite at a Cartesian position in metres.
        """

        position_m = np.asarray(position_m, dtype = float)

        if position_m.shape != (3,) or not np.all(np.isfinite(position_m)):
            raise ValueError("Satellite position must contain three finite values.")

        position_km = position_m / 1000.0

        satellite = pv.Sphere(radius = 250, center = position_km)

        self.plotter.add_mesh(satellite, color = "#ffc35a", name = "satellite")

    def update_orbit(self, *args) -> None:

        """Update the displayed Keplerian orbit.
        """

        try:
            elements = self.elements()
            positions_m = sample_orbit(elements, EARTH_MU, samples = 181)

            state = keplerian_to_cartesian(elements, EARTH_MU)

            self.draw_trajectory(positions_m)
            self.draw_satellite(state[:3])

            a_km = elements[0] / 1000.0
            e = elements[1]

            periapsis_km = a_km * (1.0 - e)
            apoapsis_km = a_km * (1.0 + e)

            note = ("\nВнимание: орбита проходит внутри Земли!" if periapsis_km < EARTH_RADIUS_KM else "")

            self.status.setText(
                f"Перицентр: {periapsis_km:.1f} км от центра Земли\n"
                f"Апоцентр: {apoapsis_km:.1f} км от центра Земли"
                f"{note}")

            self.plotter.render()

        except (ValueError, RuntimeError, FloatingPointError) as exc:
            QMessageBox.critical(self, "Ошибка вычисления!", str(exc))

    def reset_values(self):
        defaults = {'a': 7000, 'e': 0.05, 'i': 30,'omega': 45, 'raan': 60, 'nu': 90}
        for key, value in defaults.items():
            self.spins[key].setValue(value)


def main() -> None:

    """Launch the OrbitVisualizer application.
    """

    app = QApplication(sys.argv)
    window = OrbitVisualizer()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
