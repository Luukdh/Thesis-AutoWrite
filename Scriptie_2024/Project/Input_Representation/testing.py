import sys
import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd

from PyQt5.QtWidgets import *

class InputCollector(QWidget):
    def __init__(self):
        super().__init__()

        self.label = QLabel()
        self.label.setStyleSheet("background-image: url(alpha.png);")
        

        
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AutoWrite")
        self.setGeometry(0, 0, 800, 600)
        # self.show()

        self.input_collector = InputCollector()
        self.setCentralWidget(self.input_collector)

        # self.input_collector.setStyleSheet("background-image: url(alpha.png);")
        # self.input_collector.setStyleSheet("background-color: rgb(50, 255, 255);")


app = QApplication(sys.argv)
window = MainWindow()
window.show()

app.exec_()