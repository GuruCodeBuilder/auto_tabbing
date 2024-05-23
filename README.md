# auto_guitar_tab

Convert wav files of electric guitar recordings into tablature that can be read.

This project is experimental.

This project will make use of fast fourier transforms (FFT), machine learning, and web development to create an application for guitarists.
This will be a web program made using Vue.js and a backend like Django or Flask. The database will be sqlite, MySQL, or postgreSQL

The input would be a .wav file inputted by the use. There are few pre-conditions for this .wav file. The listed requirements thus far include:

- the file is .wav
- the .wav file must be audio from the guitar directly inputted into a computer audio proccessing system.
- this implies the guitar being used is electric or can in some way directly transfer sound (.wav) files to the computer
- The guitar is a standard 6 string guitar
  It is recommended that the audio is clear of noise for more precise accuracy.

The output will be a displayed tablature in the GUI. The use will have the option to listen to the chord to check for its accuracy. They also have options to edit the tablatures being made with different audio proccessing tools, and the GUI will in real time reflect changes.

Methodology:
First, the .wav file will be proccessed in from the web through some input. Then the file will go through a FFT and be seperated into more fundemental components. Then, machine learning algorithms will be used to interpret which notes are being played and will compile these individual notes. Then another elgorithm will determine the placement of these notes on the fretboard of a guitar. This data will be sent from the backend back to the frontend, and the front end will interpret this data and display it on a GUI.

Considerations in Methodology:

- Considering algorithms like the Constant Q-transform, which is closely related to the Fourier Transform and is suited for musical representation.
- (look into differences between Constant Q-transform and Variable Q-transform)
- The Q-transform may eliminate need for machine learning or reduce it to a method of filtering noise out. This may not even need to be implemented directly
  and likely exists is some other algorithm already released as a open source library

Other possible features include:

- Sign in/Sign out from users
- Profiles for each user to save different songs and tablatures created
