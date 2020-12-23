# Phone detector

Phone detector is a web application that uses the webcam if the user is using a smartphone to take pictures of the screen (to avoid leakage of secret document) or to use a selfie to deceive the facial recognition (to prevent false check-in) at real-time based on the recognition result of CocoSSD.

This is running in the browser in realtime using [TensorFlow.js](https://www.tensorflow.org/js)

*This is not an officially supported Google product.*

## Build And Run

Install dependencies and prepare the build directory:

```sh
yarn
```

To watch files for changes, and launch a dev server:

```sh
yarn watch
```

## Platform support

Demos are supported on Desktop Chrome and iOS Safari.

It should also run on Chrome on Android and potentially more Android mobile browsers though support has not been tested yet.

// TODO: Make the app recognize the owner of the computer