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
    def __init__(self, width=1200, height=600, alpha_type="normal", parent_window=None):
        """ Initializer. """
        super().__init__()
        self.alpha_type = alpha_type
        self.parent_window = parent_window

        self.setWindowTitle("Create the perfect Alphabet!")
        self.setGeometry(0, 0, width + 50, height + 100)

        self.explanation = QLabel()
        if type == "dirty":
            self.explanation.setText("Draw a \"dirty\" alphabet!")
        else:
            self.explanation.setText("Draw the perfect alphabet!")

        self.clearButton = QPushButton("Clear")
        self.clearButton.setFixedSize(200, 40)
        self.clearButton.clicked.connect(self.clearEvent)

        self.undoButton = QPushButton("Undo")
        self.undoButton.setFixedSize(200, 40)
        self.undoButton.clicked.connect(self.undoEvent)

        self.saveButton = QPushButton("Save")
        self.saveButton.setFixedSize(200, 40)
        self.saveButton.clicked.connect(self.saveEvent)

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
        self.top_bar_layout.addWidget(
            self.saveButton,
            alignment=QtCore.Qt.AlignRight
        )
        self.top_bar.setLayout(self.top_bar_layout)

        self.widget = InputWidget()
        self.widget.setFixedSize(width, height)
        self.widget.setAlignment(
                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter
        )
        self.setCursor(QtCore.Qt.CrossCursor)
        self.setStyleSheet("InputWidget {background-image: url(./images/alpha.png); background-repeat: no-repeat; background-position: center;}")

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
            m_stroke = self.find_median_coordinate(self.current_stroke)
            if m_stroke[0] < 0 or m_stroke[1] < 0 or m_stroke[0] > 1200 or m_stroke[1] > 600:
                print(f"Stroke is out of bounds.")
                return
            self.history.append(self.current_stroke)
        else:
            return

        self.current_stroke = []
        self.last_x = None
        self.last_y = None

        # self.printHistory()
        # print(f"Last stroke: {self.history[-1]}\n")
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

    def interpolate(self, stroke, length):
        '''
        This funtion takes a normal input stroke and returns a stroke 
        with the a contant length of "len" using linear interpolation.
        '''
        new_stroke = []
        for i in np.linspace(0, len(stroke) - 1, length, endpoint=False):
            new_x = stroke[int(i)][0] + (i - int(i)) * (stroke[int(i) + 1][0] - stroke[int(i)][0])
            new_y = stroke[int(i)][1] + (i - int(i)) * (stroke[int(i) + 1][1] - stroke[int(i)][1])
            new_p = stroke[int(i)][2]
            new_stroke.append([int(new_x), int(new_y), new_p])
        return new_stroke
    
    def normalize(self, stroke):
        '''
        This function takes a normal input stroke and normalizes these coordinates
        to the middle of the roster square. 
        '''
        dx, dy = 50, 50
        new_stroke = []
        for i in range(len(stroke)):
            new_x = stroke[i][0] % 100 - dx
            new_y = stroke[i][1] % 100 - dy
            new_p = stroke[i][2]
            new_stroke.append([int(new_x), int(new_y), new_p])
        return new_stroke
     
    def resample(self, stroke, length):
        '''
        This function takes a normal input stroke and resamples it to have a constant length
        of "len" by removing random points or duplicating random points using random sampling. 
        '''   
        new_stroke = []
        for i in np.linspace(0, len(stroke) - 1, length):
            new_stroke.append(stroke[int(i)])
        return new_stroke
    
    def find_median_coordinate(self, coordinates_list):
        if not coordinates_list:
            return None

        # Transpose the coordinates list to separate x and y values
        x_values, y_values, _ = zip(*coordinates_list)

        # Calculate the median for each dimension
        median_x = sorted(x_values)[len(x_values) // 2]
        median_y = sorted(y_values)[len(y_values) // 2]

        return [median_x, median_y]
    
    def find_coordinate_index(self, len, width, height, coord):
        x, y = coord
        x_index = x // width
        y_index = y // height

        return x_index + (y_index * len)
    
    def orderStrokes(self):
        '''
        This function takes the input strokes and places them in an alphabet.
        '''
        list_char = [
            "A", "a", "B", "b", "C", "c", "D", "d", "E", "e", "F", "f",
            "G", "g", "H", "h", "I", "i", "J", "j", "K", "k", "L", "l",
            "M", "m", "N", "n", "O", "o", "P", "p", "Q", "q", "R", "r",
            "S", "s", "T", "t", "U", "u", "V", "v", "W", "w", "X", "x",
            "Y", "y", "Z", "z", "0", "1", "2", "3", "4", "5", "6", "7",
            "8", "9", ".", ",", ";", ":", "!", "\"", "/", "?", "#", "@"
        ]
        num_char = 12 * 6
        if self.alpha_type == "dirty":
            self.characters = self.parent_window.dirty_alphabet
        else:
            self.characters = self.parent_window.alphabet
        for i, stroke in enumerate(self.history):
            m_stroke = self.find_median_coordinate(stroke)
            index = self.find_coordinate_index(12, 100, 100, m_stroke)
            if list_char[index] in self.characters:
                self.characters[list_char[index]].append(stroke)
            else:
                self.characters[list_char[index]] = [stroke]

        return self.characters
    
    def notFinishedDialog(self):
        msg = QMessageBox()
        msg.setWindowTitle("Not Finished!")
        msg.setText("Please draw all characters!")
        msg.setIcon(QMessageBox.Warning)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def saveEvent(self):
        """ Handles save events. """
        stroke_len = 256
        if len(self.history) == 0:
            return
        self.history = [self.interpolate(stroke, stroke_len) for stroke in self.history]

        ordered_characters = self.orderStrokes()
        normalized_characters = {}
        for key, value in ordered_characters.items():
            normalized_characters[key] = [self.normalize(stroke) for stroke in value]
         
        # Create popup if not all characters are drawn.
        if len(ordered_characters) != 72:
            print("Please draw all characters!")
            self.notFinishedDialog()
            return

        # Save the drawn alphabet.
        if self.alpha_type == "dirty":
            self.parent_window.dirty_alphabet = normalized_characters
        else:
            self.parent_window.alphabet = normalized_characters
        
        self.close()


class InputGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AutoWrite")
        self.setGeometry(0, 0, 800, 600)

        # self.setStyleSheet("background-image: url(alpha.png);")
        self.setStyleSheet("background-color: rgb(50, 200, 100);")
        
        self.AlphaButton = QPushButton("Create Alphabet!")
        self.AlphaButton.setFixedSize(400, 80)
        self.AlphaButton.clicked.connect(self.create_alphabet)

        self.DirtyAlphaButton = QPushButton("Create \"Dirty\" Alphabet!")
        self.DirtyAlphaButton.setFixedSize(400, 80)
        self.DirtyAlphaButton.clicked.connect(self.create_dirty_alphabet)

        self.SaveAlphaButton = QPushButton("Save Alphabets as Pickles.")
        self.SaveAlphaButton.setFixedSize(400, 80)
        self.SaveAlphaButton.clicked.connect(self.save_alphabet)

        self.centralWidget = QWidget()
        self.centralWidgetLayout = QVBoxLayout()

        self.centralWidgetLayout.addWidget(
            self.AlphaButton,
            alignment=QtCore.Qt.AlignCenter
        )
        self.centralWidgetLayout.addWidget(
            self.DirtyAlphaButton,
            alignment=QtCore.Qt.AlignCenter
        )
        self.centralWidgetLayout.addWidget(
            self.SaveAlphaButton,
            alignment=QtCore.Qt.AlignCenter
        )
        self.centralWidget.setLayout(self.centralWidgetLayout)
        self.setCentralWidget(self.centralWidget)

        self.alphabet = {}
        self.clicked = False
        self.dirty_alphabet = {}
        
    def create_alphabet(self):
        if self.clicked:
            print("Last recorded alphabet deleted.")
            self.alphabet = {}
        self.clicked = True
        self.newWindow = GetAlphabet(parent_window=self)
        self.newWindow.show()

    def create_dirty_alphabet(self):
        self.newWindow = GetAlphabet(parent_window=self, alpha_type="dirty")
        self.newWindow.show()

    def print_alphabet(self):
        for key, value in self.alphabet.items():
            print(f"{key}: {value}\n")

    def save_alphabet(self):
        with open("./data/other/alphabet.pkl", "wb") as f:
            pickle.dump(self.alphabet, f)
        with open("./data/other/dirty_alphabet.pkl", "wb") as f:
            pickle.dump(self.dirty_alphabet, f)
        print("Alphabets saved as pickles.")
        # self.print_alphabet()
    

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = InputGUI()
    window.show()

    app.exec_()