document.getElementById("predictBtn").addEventListener("click", function() {
  const audioFileInput = document.getElementById("engine_audio");
  const audioFile = audioFileInput.files[0];

  if (!audioFile) {
    alert("Please upload an engine audio file.");
    return;
  }

  // Collect all form data including hidden fields but exclude files except engine_audio
  const form = document.getElementById("audioForm");
  const formData = new FormData();

  // Append hidden inputs and text inputs
  form.querySelectorAll('input[type="hidden"], input[type="number"]').forEach(input => {
    formData.append(input.name, input.value);
  });

  // Append engine_audio file
  formData.append("engine_audio", audioFile);

  fetch("/predict-image-audio", {
    method: "POST",
    body: formData,
  })
  .then(response => response.json())
  .then(data => {
    document.getElementById("predicted_price").value = data.predicted_price;

    const audioClipsDiv = document.getElementById("audioClips");
    audioClipsDiv.style.display = "block";

    const engineAudioPlayer = document.getElementById("engineAudioPlayer");
    const noiseAudioPlayer = document.getElementById("noiseAudioPlayer");

    if (data.cleaned_audio_url) {
      engineAudioPlayer.src = data.cleaned_audio_url;
      engineAudioPlayer.load();
    } else {
      engineAudioPlayer.src = "";
    }

    if (data.noise_audio_url) {
      noiseAudioPlayer.src = data.noise_audio_url;
      noiseAudioPlayer.load();
    } else {
      noiseAudioPlayer.src = "";
    }
  })
  .catch(error => {
    alert("Prediction failed. Please check your inputs.");
    console.error("Error:", error);
  });
});
