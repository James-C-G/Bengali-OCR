import numpy as np
import tensorflow as tf

from settings import *

from kivy.app import App
from kivy.core.window import Window
from kivy.uix.widget import Widget
from kivy.graphics import Line
from kivy.config import Config

from cv2 import imread, imwrite, resize, IMREAD_GRAYSCALE


Config.set("input", "mouse", "mouse,multitouch_on_demand")  # Disable right click default


class DrawInput(Widget):  # Drawing widget
    def on_touch_down(self, touch):
        print(touch)
        if touch.button == 'right':  # On right click clear canvas
            self.canvas.clear()
            print("CLEARED CANVAS!", touch)

        with self.canvas:  # While mouse is held, draw white line at mouse coords
            touch.ud["line"] = Line(width=2, points=(touch.x, touch.y))

    def on_touch_move(self, touch):
        print(touch)
        touch.ud["line"].points += (touch.x, touch.y)  # While mouse is held and is moving, continue drawing

    def on_touch_up(self, touch):
        print("RELEASED!", touch)  # Print info upon mouse release


class SimpleKivy4(App):  # Kivy app
    def build(self):
        Window.bind(on_request_close=self.on_request_close)  # When closing the windows, run defined method
        Window.size = (112, 112)  # Size of window (4 * 28 = 112) for scalability
        Window.clearcolor = (0, 0, 0, 1)  # Set background to black
        self.parent = DrawInput()  # Initialise drawing widget
        return self.parent  # Return app to run

    def on_request_close(self, *args):
        self.parent.export_to_png("user_image.png")  # On app close export drawn image to png

        # Load, resize, and save image
        image = imread("user_image.png")
        image = resize(image, (28, 28))
        imwrite("user_image.png", image)


def test_image(img_path, model_path):
    # Load image as greyscale and normalise it
    img_array = imread(img_path, IMREAD_GRAYSCALE)
    new_array = img_array / 255.0

    # Reshape image into numpy array
    x = np.array(new_array).reshape((-1, 28, 28, 1))

    # Load model
    model = tf.keras.models.load_model(model_path)

    # Get prediction on passed image
    predictions = model.predict(x)

    # Convert predictions into a score between 0 and 1 (percentage)
    score = tf.nn.softmax(predictions[0])

    # Sort into descending order
    top_classes = np.argsort(-score)
    top_percentages = -np.sort(-score)

    # Output top 3 classes with their confidence percentage
    print(f"This image most likely belongs to either: \n"
          f"Class {top_classes[0]} with a {round(100 * top_percentages[0], 3)}% confidence.\n"
          f"Class {top_classes[1]} with a {round(100 * top_percentages[1], 3)}% confidence.\n"
          f"Class {top_classes[2]} with a {round(100 * top_percentages[2], 3)}% confidence.")


if __name__ == "__main__":
    SimpleKivy4().run()
    test_image("user_image.png", CHAR_MODEL)






