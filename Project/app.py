import sys
import matplotlib.pyplot as plt
import numpy as np
import pickle

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QMessageBox
)

from PyQt5.QtGui import (
    QPainter,
    QPen,
    QColor,
    QPixmap,
    QColor,
    QBrush
)
import PyQt5.QtCore as QtCore
from PyQt5.QtCore import (
    Qt
)

import torch
import torch.nn as nn

from Model_and_Training.unet import UNet
from Input_Representation.input_widget import NormalInputWindow


'''
This class creates a window that records strokes and transforms them into neat strokes.
'''
class AppWindow(NormalInputWindow):
    def __init__(self, width=1200, height=600, parent_window=None):
        super().__init__(width, height, parent_window)
        self.initUI()

    def initUI(self):
        self.end_last_stroke = None

        device = torch.device("cpu")
        self.model = UNet(3).to(device)
        self.model.load_state_dict(torch.load("./Model_and_Training/model.ckpt"))

    def processStroke(self, stroke):
        '''
        This function takes a normal input stroke and transforms it into a neat stroke
        '''
        stroke_len = 256
        return torch.Tensor(np.array([self.normalize(self.interpolate(stroke.T, stroke_len))]))

    def processOutput(self, output):
        '''
        This function takes the output of the model and transforms it into a stroke
        '''
        return output.detach().numpy()[0].T

    def denormalize(self, stroke, start):
        '''
        This function takes a neat stroke and denormalizes it to the original canvas.
        '''
        x, y = start
        arr_x = np.array([val + x for val in stroke.T[0]])
        arr_y = np.array([val + y for val in stroke.T[1]])
        return np.stack([arr_x, arr_y, stroke.T[2]], axis=0).T

    def mouseReleaseEvent(self, event):
        """
        If the mouse is released that means the end of the current stroke.
        The stroke is added to the history.
        """
        np_stroke = np.array(self.current_stroke)
        # Check if stroke is valid.
        if np_stroke.size < 0:
            return
        
        # Check if stroke is out of bounds.
        median = np.median(np_stroke.T, axis=1)
        if median[0] < 0 or median[1] < 0 or median[0] > 1200 or median[1] > 600:
            print(f"Stroke is out of bounds.")
            self.current_stroke = []
            return
        
        print(f"np_stroke: {np_stroke}")
        start = np_stroke[-1, :-1]
        tn_stroke = self.processStroke(np_stroke)
        print(f"tn_stroke: {tn_stroke}")
        model_output = self.model(tn_stroke)
        processed_output = self.processOutput(model_output)
        output = self.denormalize(processed_output, start)
        print(f"Output: {output}")
        self.end_last_stroke = np.array(output[-1, :-1])

        self.history.append(output)
        if len(self.canvas_history) == 0:
            new_pixmap = self.blank_canvas.copy()
        else:
            new_pixmap = self.canvas_history[-1].copy()
        
        painter = QPainter(new_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(Qt.black))
        painter.pen().setWidth(2)
        last_point = None
        for point in output:
            if last_point is None:
                last_point = point
                continue
            painter.drawLine(int(last_point[0]), int(last_point[1]), int(point[0]), int(point[1]))
            last_point = point
        painter.end()

        self.current_stroke = []
        self.last_x = None
        self.last_y = None

        self.widget.setPixmap(new_pixmap.copy())
        self.canvas_history.append(new_pixmap)
        self.update()

        event.accept()
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AppWindow(1200, 600)
    window.show()

    app.exec_()
