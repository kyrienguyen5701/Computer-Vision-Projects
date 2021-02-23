let video;
let canvas;
let ctx;
let model;

function startVideo() {
    const constraints = {
		audio: false,
		video: true
	}
    if (navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia(constraints)
        .then(function(stream){
            video.srcObject = stream;
        })
        .catch(error => { console.log('error') });
    } else {
        console.log('can not get user media');
    }

    video.onloadeddata = (event) => {
        estimateHeadPose();
	};
}

async function loadModels() {
    await faceapi.nets.ssdMobilenetv1.loadFromUri(MODELS_PATH);
    await faceapi.nets.faceLandmark68Net.loadFromUri(MODELS_PATH);
    model = await tf.loadGraphModel('/static/models/head_pose/model.json');
}

window.onload = async function() {
	console.log('Page loaded ...');
    video = document.getElementById('videoInput');
    canvas = document.getElementById('imageCanvas');
    ctx = canvas.getContext('2d');
    ctx.strokeStyle = 'blue';
    ctx.fillStyle = 'blue';
    console.log('Start loading models');
    await loadModels();
    console.log('Finish loading models');
    console.log('Start video');
    startVideo();
    console.log('Video is running');
}

function estimateHeadPose() {
    setInterval(async () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        let faces = await faceapi.detectAllFaces(video).withFaceLandmarks();
        if (faces.length > 0) {
            const face = faces[0];
            const box = face.detection.box;
            const x_min = box.x;
            const y_min = box.y;
            const width = box.width;
            const height = box.height;
            const x_max = x_min + width;
            const y_max = y_min + height;
            let nx1 = parseFloat(x_min) / video.width;
            let ny1 = parseFloat(y_min) / video.height;
            let nx2 = parseFloat(x_max) / video.width;
            let ny2 = parseFloat(y_max) / video.height;
            let minSize = parseFloat(Math.min(width, height));
            let maxSize = parseFloat(Math.max(width, height));
            let ratio = maxSize / minSize;
            ctx.beginPath();
            ctx.rect(x_min, y_min, width, height);
            ctx.closePath();
            ctx.stroke();
            let idx_tensor = [];
            for (let i = 0; i < 66; i++) {
                idx_tensor.push(i);
            }
            tf.engine().startScope();
            let img = tf.browser.fromPixels(video).expandDims().toFloat()
            img = tf.image.resizeBilinear(img, [224, parseInt(224 * ratio)])
            const normalizedBox = tf.tensor([ny1, nx1, ny2, nx2]).expandDims().toFloat()
            const normalizedBoxIndex = tf.tensor([0]).toInt()
            img = tf.image.cropAndResize(img, normalizedBox, normalizedBoxIndex, [224, 224])
            img = img.div(tf.scalar(255));
            img = img.sub(tf.tensor([0.485, 0.456, 0.406]))
            img = img.div(tf.tensor([0.229, 0.224, 0.225]))
            const outputs = model.predict(img)
            let yaw_predicted = outputs[0].softmax(1).squeeze()
            let pitch_predicted = outputs[1].softmax(1).squeeze()
            let roll_predicted = outputs[2].softmax(1).squeeze()
            idx_tensor = tf.tensor(idx_tensor)
            yaw_predicted = tf.sum(yaw_predicted.mul(idx_tensor))
            yaw_predicted = yaw_predicted.arraySync()
            pitch_predicted = tf.sum(pitch_predicted.mul(idx_tensor))
            pitch_predicted = pitch_predicted.arraySync()
            roll_predicted = tf.sum(roll_predicted.mul(idx_tensor))
            roll_predicted = roll_predicted.arraySync()
            tf.engine().endScope()
            yaw_predicted = yaw_predicted * 3 - 99
            pitch_predicted = pitch_predicted * 3 - 99
            roll_predicted = roll_predicted * 3 - 99

            // console.log(yaw_predicted, "\n", pitch_predicted, "\n", roll_predicted)
            const nose = face.landmarks.positions[33];
            const tdx = nose.x;
            const tdy = nose.y - 10/480 * canvas.height;
            const axes = getAxes(yaw_predicted, pitch_predicted, roll_predicted, tdx, tdy, height / 2)
            Object.values(axes).forEach(axis => {
                ctx.strokeStyle = axis.c;
                ctx.fillStyle = axis.c;
                ctx.beginPath();
                ctx.moveTo(tdx.toFixed(0), tdy.toFixed(0));
                ctx.lineTo(axis.x.toFixed(0), axis.y.toFixed(0));
                ctx.closePath();
                ctx.stroke();
            })    
        }
    }, 0)
}
