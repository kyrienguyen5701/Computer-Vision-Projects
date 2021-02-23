# Headpose Estimation

Headpose Estimation is a web application that uses machine learning models (the head-pose model is converted from the original [PyTorch model](https://github.com/natanielruiz/deep-head-pose), the rest are from face-api.js) to estimate the direction in which the user's head is facing to. 

This is running in the browser in realtime using [TensorFlow.js](https://www.tensorflow.org/js)

*This is not an officially supported Google product.*

## Build And Run

Install dependencies:

```sh
pip install -r requirements.txt
```

Launch the app:
```sh
python app.py
```

After launching the app, open your browser and go to `localhost:8080`

## Platform support

Demos are supported on Desktop Chrome and iOS Safari.

It should also run on Chrome on Android and potentially more Android mobile browsers though support has not been tested yet.