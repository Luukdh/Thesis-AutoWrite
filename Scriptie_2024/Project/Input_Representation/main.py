import sys
import matplotlib.pyplot as plt
import numpy as np

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton
)

from PyQt5.QtGui import (
    QPainter,
    QColor,
    QPixmap,
    QColor
)
import PyQt5.QtCore as QtCore
from PyQt5.QtCore import (
    Qt
)


class InputWidget(QLabel):
    def __init__(self):
        super().__init__()

'''
This class creates a window that allows the user to record their own perfect alphabet.
'''
class GetAlphabet(QMainWindow):
    """ Input widget. """
    def __init__(self, width=1200, height=600):
        """ Initializer. """
        super().__init__()
        self.setWindowTitle("Create the perfect Alphabet!")
        self.setGeometry(0, 0, width + 50, height + 100)

        self.explanation = QLabel()
        self.explanation.setText("Draw the perfect alphabet!")

        self.clearButton = QPushButton("Clear")
        self.clearButton.setFixedSize(200, 40)
        self.clearButton.clicked.connect(self.clearEvent)

        self.undoButton = QPushButton("Undo")
        self.undoButton.setFixedSize(200, 40)
        self.undoButton.clicked.connect(self.undoEvent)

        self.top_bar = QWidget()
        self.top_bar_layout = QHBoxLayout()
        self.top_bar_layout.addWidget(
            self.explanation,
            alignment=QtCore.Qt.AlignCenter
        )
        self.top_bar_layout.addWidget(
            self.clearButton,
            alignment=QtCore.Qt.AlignCenter
        )
        self.top_bar_layout.addWidget(
            self.undoButton,
            alignment=QtCore.Qt.AlignCenter
        )
        self.top_bar.setLayout(self.top_bar_layout)

        self.widget = InputWidget()
        self.widget.setFixedSize(width, height)
        self.widget.setAlignment(
                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter
        )
        self.setCursor(QtCore.Qt.CrossCursor)
        self.setStyleSheet("InputWidget {background-image: url(alpha.png); background-repeat: no-repeat; background-position: center;}")

        canvas = QPixmap(width, height)
        colour = QColor(255, 255, 255, 220)
        canvas.fill(colour) # Fill canvas with color (default is white).
        self.widget.setPixmap(canvas)
        
        self.centralWidget = QWidget()
        self.centralWidgetLayout = QVBoxLayout()
        self.centralWidgetLayout.addWidget(
            self.top_bar,
            alignment=QtCore.Qt.AlignCenter
        )
        self.centralWidgetLayout.addWidget(
            self.widget,
            alignment=QtCore.Qt.AlignCenter
        )

        self.centralWidget.setLayout(self.centralWidgetLayout)
        self.setCentralWidget(self.centralWidget)

        self.initialize_offsets()

        self.last_x, self.last_y, self.last_p = None, None, None
        self.current_stroke = []
        self.history = []
        self.canvas_history = []
        self.blank_canvas = self.widget.pixmap().copy()

    def initialize_offsets(self):
        """
        Keep track of the offset of the drawing canvas in the widget and the
        cursor offset while drawing.
        """
        self.canvas_offset_left = (self.widget.width() -
                self.widget.pixmap().width())/2
        self.canvas_offset_top = (self.widget.height() -
                self.widget.pixmap().height())/2
        self.cursor_offset_top = 90
        self.cursor_offset_left = 25

    def getHistory(self):
        """ Returns current input. """
        return self.history

    def getCurrentStroke(self):
        """ Returns the values stored for the current stroke. """
        return self.current_stroke

    def getLastX(self):
        """ Returns last x coordinate. """
        return self.last_x

    def getLastY(self):
        """ Returns last y coordinate. """
        return self.last_y

    def clearCanvas(self):
        """ Clears drawing canvas. """
        self.widget.setPixmap(self.blank_canvas.copy())
        self.history = []
        self.update()

    def tabletEvent(self, event):
        """ Handles tablet events. """
        current_x = int(event.x() - self.canvas_offset_left -
                self.cursor_offset_left)
        current_y = int(event.y() - self.canvas_offset_top -
                self.cursor_offset_top)
        

        if self.last_x is None: # First event.
            self.last_x = current_x
            self.last_y = current_y
            self.last_p = event.pressure()
            return # Ignore the first time.

        painter = QPainter(self.widget.pixmap())
        painter.drawLine(self.last_x, self.last_y, current_x, current_y)
        painter.end()
        self.update()

        # Update the origin for next time.
        self.last_x = current_x
        self.last_y = current_y
        self.last_p = event.pressure()
        self.current_stroke.append(
                [
                    self.last_x,
                    self.last_y,
                    self.last_p
                ]
        )

    def printHistory(self):
        """ Prints the history of the input. """
        h = self.history
        for stroke in h:
            print(stroke, "\n")

    def mouseReleaseEvent(self, event):
        """
        If the mouse is released that means the end of the current stroke.
        The stroke is added to the history.
        """
        if len(self.current_stroke) > 0:
            self.history.append(self.current_stroke)

        self.current_stroke = []
        self.last_x = None
        self.last_y = None

        # self.printHistory()
        print(f"Last stroke: {self.history[-1]}\n")
        new_pixmap = self.widget.pixmap().copy()
        self.canvas_history.append(new_pixmap)

    def resizeEvent(self, event):
        """ Handles resizing of the canvas. """
        self.canvas_offset_left =(self.widget.width() -
                self.widget.pixmap().width())/2
        self.canvas_offset_top =(self.widget.height() -
                self.widget.pixmap().height())/2
        
    def eraseLastStroke(self):
        """ Erases the last stroke. """
        self.canvas_history.pop()
        if len(self.canvas_history) == 0:
            new_pixmap = self.blank_canvas
        else:
            new_pixmap = self.canvas_history[-1]
        self.widget.setPixmap(new_pixmap.copy())
        self.update()

    def undoEvent(self):
        """ Handles undo events. """
        if len(self.history) > 0:
            self.history.pop()
            self.eraseLastStroke()

    def clearEvent(self):
        """ Handles clear events. """
        self.clearCanvas()
        self.canvas_history = []

class AutoWrite(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AutoWrite")
        self.setGeometry(0, 0, 800, 600)

        # self.setStyleSheet("background-image: url(alpha.png);")
        self.setStyleSheet("background-color: rgb(50, 200, 100);")
        
        self.AlphaButton = QPushButton("Create Alphabet!")
        self.AlphaButton.setFixedSize(400, 80)
        self.AlphaButton.clicked.connect(self.create_alphabet)
        

        self.centralWidget = QWidget()
        self.centralWidgetLayout = QVBoxLayout()

        self.centralWidgetLayout.addWidget(
            self.AlphaButton,
            alignment=QtCore.Qt.AlignCenter
        )
        self.centralWidget.setLayout(self.centralWidgetLayout)
        self.setCentralWidget(self.centralWidget)
        
    def create_alphabet(self):
        self.newWindow = GetAlphabet()
        self.newWindow.show()

    
    def interpolate(self, stroke, len):
        '''
        This funtion takes a normal input stroke and returns a stroke 
        with the a contant length of "len" using linear interpolation.
        '''
        new_stroke = []
        for i in np.linspace(0, len(stroke) - 1, len):
            new_coord = stroke[int(i)] + (i - int(i)) * (stroke[int(i) + 1] - stroke[int(i)])
            new_stroke.append(new_coord)
        return new_stroke
     
    def resample(self, stroke, len):
        '''
        This function takes a normal input stroke and resamples it to have a constant length
        of "len" by removing random points or duplicating random points using random sampling. 
        '''   
        new_stroke = []
        for i in np.linspace(0, len(stroke) - 1, len):
            new_stroke.append(stroke[int(i)])
        return new_stroke
    

app = QApplication(sys.argv)
window = AutoWrite()
window.show()

app.exec_()
