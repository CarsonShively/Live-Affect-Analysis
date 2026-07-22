const video = document.getElementById("camera");
const canvas = document.getElementById("capture-canvas");
const overlay = document.getElementById("overlay");
const context = canvas.getContext("2d");

let requestInProgress = false;

async function startCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: {
            facingMode: "user"
        },
        audio: false
    });

    video.srcObject = stream;
}

async function sendFrame() {
    if (requestInProgress || video.readyState < 2) {
        return;
    }

    requestInProgress = true;

    try {
        canvas.width = 320;
        canvas.height = 240;

        context.drawImage(
            video,
            0,
            0,
            canvas.width,
            canvas.height
        );

        const blob = await new Promise((resolve) => {
            canvas.toBlob(resolve, "image/jpeg", 0.75);
        });

        const formData = new FormData();
        formData.append("file", blob, "frame.jpg");

        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const result = await response.json();

        overlay.textContent = result.person_detected
            ? result.categories[0].label
            : "No person detected";
    } catch (error) {
        console.error(error);
        overlay.textContent = "Prediction error";
    } finally {
        requestInProgress = false;
    }
}

startCamera().then(() => {
    setInterval(sendFrame, 333);
});