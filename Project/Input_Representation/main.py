import sys
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import QApplication

from Input_Representation.input_widget import InputGUI as ig

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ig()
    window.show()

    app.exec_()
