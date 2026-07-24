const video = document.getElementById("camera");
const canvas = document.getElementById("capture-canvas");
const overlay = document.getElementById("overlay");
const flipCameraButton = document.getElementById("flip-camera");
const context = canvas.getContext("2d");

let requestInProgress = false;
let currentStream = null;
let currentFacingMode = "user";

async function startCamera() {
    // Stop the currently active camera before starting another one.
    if (currentStream) {
        currentStream.getTracks().forEach((track) => track.stop());
    }

    currentStream = await navigator.mediaDevices.getUserMedia({
        video: {
            facingMode: currentFacingMode
        },
        audio: false
    });

    video.srcObject = currentStream;
    await video.play();
}

async function flipCamera() {
    currentFacingMode =
        currentFacingMode === "user"
            ? "environment"
            : "user";

    try {
        await startCamera();
    } catch (error) {
        console.error("Camera switch failed:", error);

        // Switch the setting back if the other camera could not open.
        currentFacingMode =
            currentFacingMode === "user"
                ? "environment"
                : "user";

        overlay.textContent = "Unable to switch camera";
    }
}

function displayPrediction(result) {
    if (result.status === "No Person Detected") {
        overlay.textContent = "No person detected";
        return;
    }

    if (!result.success) {
        overlay.textContent =
            result.status || result.details || "Prediction error";
        return;
    }

    overlay.textContent =
        `Mood: ${result.mood}/10\n` +
        `Energy: ${result.energy}/10\n` +
        `Happiness: ${result.happiness}%\n` +
        `Calmness: ${result.calmness}%\n` +
        `Sadness: ${result.sadness}%\n` +
        `Fear: ${result.fear}%\n` +
        `Anger: ${result.anger}%`;
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

        if (!blob) {
            return;
        }

        const formData = new FormData();
        formData.append("file", blob, "frame.jpg");

        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Prediction request failed: ${response.status}`);
        }

        const result = await response.json();
        displayPrediction(result);

    } catch (error) {
        console.error("Prediction failed:", error);
        overlay.textContent = "Prediction error";
    } finally {
        requestInProgress = false;
    }
}

flipCameraButton.addEventListener("click", flipCamera);

startCamera()
    .then(() => {
        setInterval(sendFrame, 333);
    })
    .catch((error) => {
        console.error("Camera startup failed:", error);
        overlay.textContent = "Camera unavailable";
    });