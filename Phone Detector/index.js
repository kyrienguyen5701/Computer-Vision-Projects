import * as cocoSsd from '@tensorflow-models/coco-ssd';
import '@tensorflow/tfjs-backend-cpu'
import '@tensorflow/tfjs-backend-webgl';
import * as tf from '@tensorflow/tfjs-core';

let videoWidth, videoHeight, rafID, ctx, canvas, model;
const VIDEO_WIDTH = 640;
const VIDEO_HEIGHT = 480;

const state = {
    backend: 'webgl',
    baseModel: 'mobilenet_v2'
}

const stats = new Stats();
stats.showPanel(0);
document.body.appendChild(stats.dom);

function setupDatGui() {
    const gui = new dat.GUI();
    gui.add(state, 'backend', ['webgl', 'cpu'])
        .onChange(async backend => {
            window.cancelAnimationFrame(rafID);
            await tf.setBackend(backend);
            detectPhoneRealTime(video);
        })
    gui.add(state, 'baseModel', ['mobilenet_v2', 'mobilenet_v1', 'lite_mobilenet_v2'])
        .onChange(async baseModel => {
            window.cancelAnimationFrame(rafID);
            model = await cocoSsd.load({base: baseModel});
            detectPhoneRealTime(video); 
        })
}

async function setupCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error(
            'Browser API navigator.mediaDevices.getUserMedia not available');
      }
    
    const video = document.getElementById('video');
    const stream = await navigator.mediaDevices.getUserMedia({
        'audio': false,
        'video': {
            facingMode: 'user',
            // Only setting the video to a specified size in order to accommodate a
            // point cloud, so on mobile devices accept the default size.
            width: VIDEO_WIDTH,
            height: VIDEO_HEIGHT
        },
    });
    video.srcObject = stream;
    
    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            resolve(video);
        };
    });
}

async function loadVideo() {
    const video = await setupCamera();
    video.play();
    return video;
}

async function main() {
    await tf.setBackend(state.backend);
    model = await cocoSsd.load({base: state.baseModel});
    
    let video;
    try {
        video = await loadVideo();
    }
    catch (e) {
        let info = document.getElementById('info');
        info.textContent = e.message;
        info.style.display = 'block';
        throw e;
    }

    setupDatGui();

    videoWidth = video.videoWidth;
    videoHeight = video.videoHeight;

    canvas = document.getElementById('output');
    canvas.width = videoWidth;
    canvas.height = videoHeight;
    video.width = videoWidth;
    video.height = videoHeight;

    ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, videoWidth, videoHeight);
    ctx.strokeStyle = 'blue';
    ctx.fillStyle = 'blue';
    ctx.font = '10px Arial';
    
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);

    detectPhoneRealTime(video);
}

const detectPhoneRealTime = async(video) => {
    async function frameDetections() {
        stats.begin();
        ctx.drawImage(
            video, 0, 0, videoWidth, videoHeight, 0, 0,
            canvas.width, canvas.height
        );
        let result = await model.detect(video);
        result = result.filter((o) => o.class === 'cell phone');
        if (result.length > 0) {
            console.log('Phone detected');
            for (let i = 0; i < result.length; i++) {
                ctx.beginPath();
                ctx.rect(...result[i].bbox);
                ctx.lineWidth = 1;
                ctx.stroke();
                ctx.fillText(
                    result[i].score.toFixed(3) + ' ' + result[i].class, result[i].bbox[0],
                    result[i].bbox[1] > 10 ? result[i].bbox[1] - 5 : 10
                );
            }
        }
        else console.log('No phone detected');
        stats.end();
        rafID = requestAnimationFrame(frameDetections);
    }
    frameDetections();  
}

navigator.getUserMedia = navigator.getUserMedia ||
    navigator.webkitGetUserMedia || navigator.mozGetUserMedia;

main();