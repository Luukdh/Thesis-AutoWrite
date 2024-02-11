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

class InputWidget(QLabel):
    def __init__(self):
        super().__init__()

class InputWindow(QMainWindow):
    def __init__(self, width=1200, height=600, alpha_type=None, parent_window=None):
        """ Initializer. """
        super().__init__()
        self.setGeometry(0, 0, width + 50, height + 100)

        self.explanation = QLabel()

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

        # Create the input window.
        self.widget = InputWidget()
        self.widget.setFixedSize(width, height)
        self.widget.setAlignment(
                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter
        )
        # Set the cursor style.
        self.setCursor(QtCore.Qt.CrossCursor)

        # Call the initializer for the GUI.
        if alpha_type is not None:
            self.initAlphaGUI(alpha_type, parent_window, width, height)
        else:
            self.initNormalGUI(width, height, parent_window)
        
        self.top_bar.setLayout(self.top_bar_layout)

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

    def initNormalGUI(self, width, height, parent_window):
        self.setWindowTitle("AutoWrite")
        self.explanation.setText("Write using one stroke per letter!")

    def initAlphaGUI(self, alpha_type, parent_window, width, height):
        self.setWindowTitle("Create an Alphabet!")
        self.setFixedSize(width + 50, height + 100)

        if alpha_type == "dirty":
            self.explanation.setText("Draw a \"dirty\" alphabet!")
        else:
            self.explanation.setText("Draw the perfect alphabet!")

        self.setStyleSheet("InputWidget {background-image: url(./images/honey_final.png);\
                            background-repeat: no-repeat; background-position: center;}")

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
        
    def clearCanvas(self):
        """ Clears drawing canvas. """
        self.widget.setPixmap(self.blank_canvas.copy())
        self.history = []
        self.update()

    def tabletEvent(self, event):
        """ Handles tablet events. """
        current_x = event.posF().x() - self.canvas_offset_left - self.cursor_offset_left
        current_y = event.posF().y() - self.canvas_offset_top - self.cursor_offset_top

        if current_x < 0 or current_y < 0 or current_x > 1200 or current_y > 600:
            print(f"point is out of bounds.")
            return

        if self.last_x is None: # First event.
            self.last_x = current_x
            self.last_y = current_y
            self.last_p = event.pressure()
            return # Ignore the first time.

        painter = QPainter(self.widget.pixmap())
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(Qt.black))
        painter.pen().setWidth(2)
        painter.drawLine(int(self.last_x), int(self.last_y), int(current_x), int(current_y))
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
        # Accepting the event causes malfunction.
        # event.accept()
    
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
        x, y, p = stroke
        size = stroke.shape[1]
        interp_x = np.interp(np.linspace(0, size, length, endpoint=True), np.arange(size), x)
        interp_y = np.interp(np.linspace(0, size, length, endpoint=True), np.arange(size), y)
        interp_p = np.interp(np.linspace(0, size, length, endpoint=True), np.arange(size), p)
        return np.stack([interp_x, interp_y, interp_p], axis=0)


class NormalInputWindow(InputWindow):
    def __init__(self, width=1200, height=600, parent_window=None):
        super().__init__(width, height, parent_window=parent_window)
    
    def resizeEvent(self, event):
        """ Handles resizing of the canvas. """
        self.canvas_offset_left =(self.widget.width() -
                self.widget.pixmap().width())/2
        self.canvas_offset_top =(self.widget.height() -
                self.widget.pixmap().height())/2  

    def normalize(self, stroke):
        '''
        This function takes a normal input stroke and normalizes these coordinates
        to the middle of the roster square. 
        '''
        # Find median.
        x, y = np.median(stroke[:-1], axis=1)
        # Translate to origin.
        arr_x = np.array([val - x for val in stroke[0]])
        arr_y = np.array([val - y for val in stroke[1]])
        return np.stack([arr_x, arr_y, stroke[2]], axis=0) 


class AlphaInputWindow(InputWindow):
    def __init__(self, width=1200, height=600, alpha_type=None, parent_window=None):
        super().__init__(width, height, alpha_type=alpha_type, parent_window=parent_window)
        self.alpha_type = alpha_type
        self.parent_window = parent_window

        self.saveButton = QPushButton("Save")
        self.saveButton.setFixedSize(200, 40)
        self.saveButton.clicked.connect(self.saveEvent)

        self.top_bar_layout.addWidget(
            self.saveButton,
            alignment=QtCore.Qt.AlignRight
        )

    def normalize(self, stroke):
        '''
        This function takes a normal input stroke and normalizes these coordinates
        to the base line of the roster square or the median of the stroke. 
        '''
        # Find the median of the stroke.
        median = np.median(stroke[:-1], axis=1)
        if self.alpha_type == "dirty":
            # Normalization point is the median of the stroke.
            x, y = median
        else:
            # Normalization point is the base line of the square.
            x = (median[0] // 100) * 100 + 50
            y = (median[1] // 100) * 100 + 70
        # Translate to origin.
        arr_x = np.array([val - x for val in stroke[0]])
        arr_y = np.array([val - y for val in stroke[1]])
        return np.stack([arr_x, arr_y, stroke[2]], axis=0)

    def mouseReleaseEvent(self, event):
        """
        If the mouse is released that means the end of the current stroke.
        The stroke is added to the history.
        """
        out_of_bounds = False
        np_stroke = np.array(self.current_stroke)
        if np_stroke.size > 0:
            median = np.median(np_stroke.T, axis=1)
            if median[0] < 0 or median[1] < 0 or median[0] > 1200 or median[1] > 600:
                print(f"Stroke is out of bounds.")
                out_of_bounds = True
            self.history.append(np_stroke)
        else:
            return

        self.current_stroke = []
        self.last_x = None
        self.last_y = None
        self.last_p = None

        new_pixmap = self.widget.pixmap().copy()
        self.canvas_history.append(new_pixmap)

        if out_of_bounds:
            self.undoEvent()

    def find_coordinate_index(self, len, width, height, coord):
        '''
        Find the index of the character that represents the roster square.
        '''
        x, y, _ = coord
        x_index = int(x) // width
        y_index = int(y) // height

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
            median = np.median(stroke, axis=1)
            index = self.find_coordinate_index(12, 100, 100, median)
            if list_char[index] not in self.characters:
                self.characters[list_char[index]] = np.array([self.normalize(stroke)])
            else:
                char_strokes = self.characters[list_char[index]]
                self.characters[list_char[index]] = np.append(char_strokes, [self.normalize(stroke)], axis=0)            
        print(f"Characters: {self.characters}")
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
        # Create popup if not all characters are drawn.
        if len(self.history) < 72:
            print("Please draw all characters!")
            self.notFinishedDialog()
            return
        
        # Interpolate the strokes to have a constant length.
        self.history = [self.interpolate(stroke.T, stroke_len) for stroke in self.history]
        ordered_characters = self.orderStrokes()
        # Save the drawn alphabet.
        if self.alpha_type == "dirty":
            self.parent_window.dirty_alphabet = ordered_characters
        else:
            self.parent_window.alphabet = ordered_characters
        
        self.close()

class InputGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AutoWrite")
        self.setGeometry(0, 0, 800, 600)

        self.setStyleSheet("background-color: rgb(200, 200, 200);")
        
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
        self.newWindow = AlphaInputWindow(parent_window=self, alpha_type="normal")
        self.newWindow.show()

    def create_dirty_alphabet(self):
        self.newWindow = AlphaInputWindow(parent_window=self, alpha_type="dirty")
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
    

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = InputGUI()
    window.show()

    app.exec_()