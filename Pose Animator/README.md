# Pose Animator

## Introduction

Update 10/19/2020: This project is closed since its purpose was met - to help me understand the animating algorithm and how models like PoseNet and FaceMesh work. Further development moves to *Kawaii Face Animator*

This clone project is purely for educational purpose. Check the original project in the previous commit or [here](https://github.com/yemount/pose-animator)

In this project, my goal is to animate the character's face so that it matches with the movement of the user's face on camera at real time. To do this, I deal with two subtasks:

- Detect landmarks of the user's face on camera using face-api.js
- Transform the pre-defined rig of the character SVG to the input landmarks map, using Linear Blend Skinning (LBS) algorithm

This app does not animate the body's pose. I intentionally disabled it while I was playing around and modifying the program.

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