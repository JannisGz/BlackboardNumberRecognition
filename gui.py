import os
import sys
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PIL import Image as pI
from src.classification import Classifier


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        """
        Creates a new MainWindow for the number recognition app. It consists of a blank canvas, two buttons for
        submitting and clearing the canvas and numerous text fields.
        The canvas is used to draw single digit numbers on it. With the 'Submit' button the canvas can be forwarded to
        the Classifier. The Classifier uses its trained machine learning model to predict what number was drawn on the
        canvas.
        The prediction is then displayed on the MainWindow. Additionally a percentage value is displayed which
        represents how certain the Classifier is with its prediction.
        The 'Clear' button can be used to reset the canvas.
        """
        super().__init__()

        self.classifier = Classifier()

        self.canvas = Canvas()
        self.prediction = QtWidgets.QLabel("")
        self.prediction.setAlignment(Qt.AlignCenter)
        self.certainty = QtWidgets.QLabel("")
        self.certainty.setAlignment(Qt.AlignCenter)
        self.submit = QtWidgets.QPushButton("Submit")
        self.submit.clicked.connect(self.submit_image)
        self.clear = QtWidgets.QPushButton("Clear")
        self.clear.clicked.connect(self.canvas.clear)

        window = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout()
        window.setLayout(layout)
        layout.addWidget(self.canvas)

        panel = QtWidgets.QVBoxLayout()
        panel.addWidget(self.submit)
        panel.addWidget(self.clear)
        panel.addWidget(self.prediction)
        panel.addWidget(self.certainty)

        layout.addLayout(panel)

        self.setCentralWidget(window)

    def submit_image(self):
        """
        Creates an image from the current state of the canvas. The image is resized to a 28 x 28 pixel image and
        forwarded to the Classifier. The Window is then updated with the according labels.
        """
        self.canvas.grab().save("img.png")

        im = pI.open(r"img.png")
        newsize = (28, 28)
        resized_img = im.resize(newsize)
        resized_img.save("img2.png")

        predicted_label, certainty = self.classifier.predict(resized_img)

        self.prediction.setText(str(predicted_label))
        self.certainty.setText("{0:.0%}".format(certainty))

        os.remove("img.png")
        os.remove("img2.png")


class Canvas(QtWidgets.QLabel):

    def __init__(self):
        """
        Creates a new Canvas. The Canvas is a 250 x 250 square with a black background. The cursor can be used as a pen
        to draw white lines.
        """
        super().__init__()

        # Set size
        pixmap = QtGui.QPixmap(250, 250)
        self.setPixmap(pixmap)

        # Draw black background
        self.clear()

        # Initialize drawing coordinates
        self.last_x, self.last_y = None, None

    def mouseMoveEvent(self, e):
        """
        Drawing event: Gets called when the cursor is moved (held down) on the Canvas. Draws on the Canvas.

        :param e: Cursor coordinates
        """
        if self.last_x is None:  # Initial mouse click event
            self.last_x = e.x()
            self.last_y = e.y()
        else:  # The mouse is already clicked and was moved to a new position on the canvas
            painter = QtGui.QPainter(self.pixmap())
            p = painter.pen()
            p.setWidth(20)
            p.setColor(QtGui.QColor('white'))
            painter.setPen(p)
            painter.drawLine(self.last_x, self.last_y, e.x(), e.y())
            painter.end()
            self.update()

            self.last_x = e.x()
            self.last_y = e.y()

    def mouseReleaseEvent(self, e):
        """
        Release event: Gets called when the cursor is released (no longer held down). Ends the drawing until the cursor
        is held down again.

        :param e: Cursor coordinates
        """
        self.last_x = None
        self.last_y = None

    def clear(self):
        """
        Resets the Canvas to a black background. Deletes all drawings on the Canvas.
        """
        painter = QtGui.QPainter(self.pixmap())
        pen = QtGui.QPen()
        pen.setWidth(1000)
        pen.setColor(QtGui.QColor('black'))
        painter.setPen(pen)
        painter.drawPoint(0, 0)
        painter.end()
        self.update()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()

    app.exec_()
