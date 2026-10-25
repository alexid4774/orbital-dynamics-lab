import sys

import numpy as np
from PySide6.QtWidgets import (QApplication, QComboBox, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QPushButton, QTextEdit, QVBoxLayout, QWidget)
from orbital_dynamics.conversions import cartesian_to_keplerian, keplerian_to_cartesian, describe_orbital_angles
from orbital_dynamics.constants import EARTH_MU

EARTH_MU_KM = EARTH_MU / 1e9


class OrbitConverter(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Конвертер орбитальных элементов')
        self.resize(650, 560)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel('Преобразование орбитальных элементов'))

        self.mode = QComboBox()
        self.mode.addItems(['Декартовы в Кеплеровы', 'Кеплеровы в Декартовы'])
        self.mode.currentIndexChanged.connect(self.change_mode)
        layout.addWidget(self.mode)

        self.group = QGroupBox('Исходные данные')
        self.form = QFormLayout(self.group)
        self.entries = []
        self.labels = []
        for i in range(6):
            entry = QLineEdit()
            label = QLabel()
            self.entries.append(entry)
            self.labels.append(label)
            self.form.addRow(label, entry)
        layout.addWidget(self.group)

        mu_row = QHBoxLayout()
        mu_row.addWidget(QLabel('Гравитационный параметр mu (км^3/с^2):'))
        self.mu_entry = QLineEdit(str(EARTH_MU_KM))
        mu_row.addWidget(self.mu_entry)
        layout.addLayout(mu_row)

        self.button = QPushButton('Рассчитать')
        self.button.clicked.connect(self.calculate)
        layout.addWidget(self.button)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        layout.addWidget(QLabel('Результат:'))
        layout.addWidget(self.output)
        self.change_mode()

    def change_mode(self):

        if self.mode.currentIndex() == 0:
            labels = ['x (км)', 'y (км)', 'z (км)',
                      'V_x (км/с)', 'V_y (км/с)', 'V_z (км/с)']
            defaults = ['7000', '0', '0', '0', '7.5', '1.0']
        else:
            labels = ['a (км)', 'e', 'i (°)', 'omega (°)', 'Omega (°)', 'nu (°)']
            defaults = ['7000', '0.01', '30', '45', '60', '90']

        for label, entry, default, widget in zip(labels, self.entries, defaults, self.labels):
            widget.setText(label + ':')
            entry.setText(default)
        self.output.clear()

    def calculate(self):

        try:

            values = []
            for entry in self.entries:
                text = entry.text()
                text = text.replace(',', '.')
                number = float(text)
                values.append(number)
            mu = float(self.mu_entry.text().replace(',', '.'))

            if self.mode.currentIndex() == 0:
                result = self.to_keplerian(values, mu)
            else:
                result = self.to_cartesian(values, mu)
            self.output.setPlainText(result)

        except (ValueError, RuntimeError, OverflowError, FloatingPointError) as error:
            QMessageBox.critical(self, 'Ошибка вычисления!', str(error))

    def to_keplerian(self, values, mu):
        """Convert Cartesian input to formatted Keplerian elements."""
        state_si = np.asarray(values, dtype=float) * 1000.0
        mu_si = mu * 1e9

        elements = cartesian_to_keplerian(state_si, mu_si)
        a_m, e, i_rad, omega_rad, raan_rad, nu_rad = elements

        angle_info = describe_orbital_angles(elements)

        lines = [f"Большая полуось a: {a_m / 1000:.8f} км",
            f"Эксцентриситет e: {e:.10f}",
            f"Наклонение i: {np.degrees(i_rad):.8f}°"]

        orbit_type = angle_info["type"]

        if orbit_type == "circular_equatorial":
            longitude_deg = np.degrees(angle_info["true_longitude_rad"])
            lines.append(f"Истинная долгота l: {longitude_deg:.8f}°")
            lines.append("Круговая экваториальная орбита: "
                "Omega, omega и nu по отдельности не определены.")

        elif orbit_type == "circular_inclined":
            raan_deg = np.degrees(angle_info["raan_rad"])
            latitude_deg = np.degrees(angle_info["argument_of_latitude_rad"])
            lines.append(f"Долгота восходящего узла Omega: {raan_deg:.8f}°")
            lines.append(f"Аргумент широты u: {latitude_deg:.8f}°")
            lines.append("Круговая орбита: omega и nu по отдельности не определены.")

        elif orbit_type == "elliptical_equatorial":
            periapsis_deg = np.degrees(angle_info["longitude_of_periapsis_rad"])
            anomaly_deg = np.degrees(angle_info["true_anomaly_rad"])
            lines.append(f"Долгота перицентра: {periapsis_deg:.8f}°")
            lines.append(f"Истинная аномалия nu: {anomaly_deg:.8f}°")
            lines.append("Экваториальная орбита: Omega не определена.")

        else:
            angles_deg = np.degrees([omega_rad, raan_rad, nu_rad]) % 360.0

            lines.extend([f"Аргумент перицентра omega: {angles_deg[0]:.8f}°",
                f"Долгота восходящего узла Omega: {angles_deg[1]:.8f}°",
                f"Истинная аномалия nu: {angles_deg[2]:.8f}°"])

        return "\n".join(lines)

    def to_cartesian(self, values, mu):

        """Convert Keplerian input to formatted Cartesian coordinates.
        Input distances are in kilometres and angles in degrees.
        The computational core uses SI units.
        """

        a_km, e, i_deg, omega_deg, raan_deg, nu_deg = values

        elements_si = np.array([a_km * 1000.0, e, *np.radians([i_deg, omega_deg, raan_deg, nu_deg])])

        state_si = keplerian_to_cartesian(elements_si, mu * 1e9)
        state_km = state_si / 1000.0

        names = ("x", "y", "z", "V_x", "V_y", "V_z")
        lines = []

        for name, value in zip(names, state_km):
            unit = "км" if name in ("x", "y", "z") else "км/с"
            lines.append(f"{name}: {value:.8f} {unit}")

        return "\n".join(lines)


def main() -> None:
    """Launch the OrbitConverter application."""
    app = QApplication(sys.argv)
    window = OrbitConverter()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
