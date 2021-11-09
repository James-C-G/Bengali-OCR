# **Bengali Optical Character Recognition Application**

## Table of Contents
1. [General Info](#1-general-info)
2. [Technologies Used](#2-technologies-used)
3. [Features](#3-features)
4. [Pre-requisites](#4-pre-requisites)
   1. [Demo Application](#41-demo-application)
   2. [Build Model](#42-build-model)
5. [Usage](#5-usage)
6. [Status](#6-status)


## 1. General Info
This application was originally developed for my third year disseration, it provides a 
Bengali optical character recognition (OCR) application through the use of four main 
models - one for characters and modifiers, and two for compound characters. 
Each of these models have been built using one of two datasets. The Ekush and Matri Vasha 
datasets that were used can be found [here](https://shahariarrabby.github.io/ekush/#download).

The datasets were converted into numpy arrays, resized, and normalised. Afterwards, they 
were stored in pickles for easier usage. Additionally, the models were built using the 
tensorflow networks provided and saved into the ".h5" format to maintain the weights of the 
network after building.

The application also allows the user to draw with a canvas by holding left click and moving 
their mouse to draw a white line, while right click clears the canvas. Once done, the user
exits the application, and the image is then saved and classified using one of the specified 
models. Once classified, the application shows the top three possible classes the image lies
within along with the corresponding confidence the model has for those decisions.

## 2. Technologies Used 
* [Python](https://www.python.org/)
* [TensorFlow](https://www.tensorflow.org/)
* [Google Colabs](https://research.google.com/colaboratory)
* [Kivy](https://kivy.org/#home)

## 3. Features
Currently implemented features:
* Convert datasets into numpy arrays 
* Normalise and reshape the numpy arrays
* Store numpy arrays in pickles
* Build convolutional neural networks with the datasets
* Save convolutional neural networks for later use
* Draw on and clear a canvas
* Save drawing made
* Classify image drawn using specified model
* Provide the confidence of the classification

## 4. Pre-requisites
### 4.1. Demo Application
To run the OCR application you must install both TensorFlow and Kivy. This can be easily through
the use of `pip`:

```
pip install --upgrade tensorflow

pip install kivy
```

### 4.2. Build Model
Additionally, if you wish to build the models yourself, you must first download the datasets - 
which can't be provided due to their size. To do this, you must first download the Ekush
datasets [here](https://drive.google.com/drive/folders/16iwJuCFrHE0JGGfyuP3KSpvGu1p_SHd0). 
Download both the male and female CSV's besides the digits. Once downloaded, place each of 
the CSV's in their respected `csv` folder in the `datasets` folder e.g. `malechar1.csv` and 
`malechar2.csv` both go into `datasets/ekush/male/chars/csv/`. Then you must download the 
images for the Matri Vasha datasets 
[here](https://drive.google.com/drive/folders/1XAPVD66BzH22W33pdcEmMhGQXwtau0CK). Once downloaded
both `female.zip` and `male.zip` from the link, extract and place the class folders in the 
`images` folder (e.g. `datasets/matrivasha/male/images/0/0_BAR_11_1_12.jpg`).

Once the datasets have been downloaded, they need to be formatted and put into pickles for
easier usage. To do this, simply run `initialise_ekush()` and `initialise_matrivasha()`
in `init_dataset.py` - however running the file on its own will do this automatically (provided
the datasets have been set up properly).

## 5. Usage
To build the models simply select the model you wish to build `{model_name}_model.py` and 
run it. This will build the model using the dataset for the specific model and save it in the
models' folder - all the models have been built already for ease of use.

To use the demo application, simply select the dataset you wish to draw a character from, and set 
the model you wish to use in `main.py`. The models you can choose from are `CHAR_MODEL, 
MODIFIER_MODEL, COMPOUND_MODEL, and MATRIVASHA_MODEL`. Once you have chosen your model, set it here:

```python
if __name__ == "__main__":
    main(CHAR_MODEL)
```

Then you can run `main.py` to start the drawing application. Draw onto the canvas with left click
and clear with right click. Once done drawing exit the application and view the console to see 
the results.

## 6. Status
Last updated 14/05/2021

Version 1.4
