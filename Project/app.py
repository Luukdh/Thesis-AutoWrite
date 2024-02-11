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

from Model_and_Training.models_class.unet_das_2 import UNet
from Input_Representation.input_widget import NormalInputWindow

'''
This class represents a line that is being drawn on.
'''
class Line:
    def __init__(self, h=None, lc=None):
        self.height = h
        self.last_coord = lc
        self.strokes = []
    
    def __str__(self):
        return f"Line: {self.height}, {self.last_coord}, {len(self.strokes)}"


'''
This class creates a window that records strokes and transforms them into neat strokes.
'''
class AppWindow(NormalInputWindow):
    def __init__(self, width=1200, height=600, parent_window=None, line_sep=50):
        super().__init__(width, height, parent_window)
        self.line_sep = line_sep
        self.initUI()

    def initUI(self):
        self.lines = []

        device = torch.device("cpu")
        self.model = UNet(3).to(device)
        self.model.load_state_dict(torch.load("./Model_and_Training/models/das_final_model_states/model_das_2_0.ckpt"))

    def undoEvent(self):
        """ Handles undo events. """
        if len(self.history) > 0:
            rm_stroke = self.history.pop()
            self.eraseLastStroke()
        else:
            print("No more strokes to undo.")
            return
        for line in self.lines:
            if rm_stroke is line.strokes[-1]:
                line.strokes.pop()
                if len(line.strokes) == 0:
                    self.lines.remove(line)
                else:
                    _, _, max_x, _ = self.findStrokeBox(line.strokes[-1])
                    line.last_coord = max_x
                break

    def clearEvent(self):
        """ Handles clear events. """
        self.clearCanvas()
        self.canvas_history = []
        self.lines = []

    def print_stroke(self, stroke):
        '''
        This function prints a stroke.
        '''
        print(f"Stroke: ")
        for channel in stroke:
            print(channel)
    
    def normalize_scale(self, stroke):
        '''
        This function normalizes the scale of a stroke.
        '''
        stroke[:, 0] = stroke[:, 0] / 1200
        stroke[:, 1] = stroke[:, 1] / 600
        return stroke

    def processStroke(self, stroke):
        '''
        This function takes a normal input stroke and transforms it into a neat stroke
        '''
        stroke_len = 256
        # self.print_stroke(stroke.T)
        interp_stroke = self.interpolate(stroke.T, stroke_len)
        # self.print_stroke(interp_stroke)
        norm_stroke = self.normalize(interp_stroke)
        # self.print_stroke(norm_stroke)
        return torch.Tensor(np.array([norm_stroke]))

    def processOutput(self, output):
        '''
        This function takes the output of the model and transforms it into a stroke
        '''
        return output.detach().numpy()[0].T

    def denormalize(self, stroke, line, min_x, max_x, min_y, max_y):
        '''
        This function takes a neat stroke and denormalizes it to the original canvas.
        '''
        period = 20
        m_x = min_x + abs(min_x - max_x) / 2
        m_y = min_y + abs(min_y - max_y) / 2
        if not line.height:
            line.height = m_y
            line.last_coord = m_x
            base_x = line.last_coord
        elif min_x > line.last_coord + period:
            base_x = line.last_coord + period
        else:
            base_x = line.last_coord
        base_y = line.height
        begin_arr = abs(np.min(stroke.T[0]))
        arr_x = np.array([val + base_x + begin_arr for val in stroke.T[0]])
        arr_y = np.array([val + base_y for val in stroke.T[1]])
        return np.stack([arr_x, arr_y, stroke.T[2]], axis=0).T

    def findStrokeBox(self, stroke):
        '''
        This function finds the stroke box of a stroke.
        '''
        min_x = np.min(stroke.T[0])
        min_y = np.min(stroke.T[1])
        max_x = np.max(stroke.T[0])
        max_y = np.max(stroke.T[1])
        return min_x, min_y, max_x, max_y

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
        if np_stroke.size > 0:
            median = np.median(np_stroke.T, axis=1)
            if median[0] < 0 or median[1] < 0 or median[0] > 1200 or median[1] > 600:
                print(f"Stroke is out of bounds.")
                return
        else:
            return        
        # Find the stroke box.
        min_x, min_y, max_x, max_y = self.findStrokeBox(np_stroke)
        print(f"Stroke box: {min_x}, {min_y}; {max_x}, {max_y}")
        # Check current lines.
        current_line = None
        for line in self.lines:
            if not (line.height - self.line_sep < max_y and line.height + self.line_sep > min_y):
                continue
            else:
                current_line = line
                break
        if not current_line:
            print(f"New line created.")
            current_line = Line()
            self.lines.append(current_line)

        # Apply the input preprocessing.
        tn_stroke = self.processStroke(np_stroke)
        
        model_output = self.model(tn_stroke)
        processed_output = self.processOutput(model_output)
        output = self.denormalize(processed_output, current_line, min_x, max_x, min_y, max_y)
        # Add stroke to line.
        current_line.strokes.append(output)
        # Find new stroke box.
        _, _, max_x, _ = self.findStrokeBox(output)
        print(max_x)
        current_line.last_coord = max_x
        print(current_line)

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
