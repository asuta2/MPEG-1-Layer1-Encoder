# MPEG-1-Layer1-Encoder

This project is a simple showcase of the MPEG-1 Layer-1 algorithm created with the intent to explain the basic steps of the algorithm and prove its effectiveness in terms of Digital Signal Processing. The encoder is wrapped inside a simple Graphical Interface for easier conversion and accessibility that lets the user compress either a prerecorded audio file or a freshly created audio file. 

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)

## Installation

To use this project, you will need to install the following libraries:

- pyaudio
- tkinter
- numpy
- wave

You can install them using pip:
``` python
    pip install pyaudio
    pip install tkinter
    pip install numpy
    pip install wave 
```
## Usage
To convert an audio file into MP3, you can either supply the program with a premade .WAV audio file or recorded in the GUI.
In order to begin the convertion, choose the necessary options and press the "Start" button. This will begin the convertion as shown on the progress bar until it finishes.

To run the project, simply type the following in the terminal:
``` 
python main.py
```

**Note:** Currently, only fixed bitrate .WAV files are supported and the user should have that in mind when converting a premade file. This problem does not occur when using the built in audio recorder.


