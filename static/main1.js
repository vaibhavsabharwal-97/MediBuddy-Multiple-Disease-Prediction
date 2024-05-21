const fileUpload = document.getElementById('file-upload');
const fileDrag = document.getElementById('file-drag');
const imagePreview = document.getElementById('image-preview');
const imageDisplay = document.getElementById('image-display');
const predResult = document.getElementById('pred-result');
const loader = document.getElementById('loader');

// Show preview of the selected image
fileUpload.addEventListener('change', (event) => {
  const file = event.target.files[0];
  const reader = new FileReader();

  reader.onload = () => {
    imagePreview.src = reader.result;
    imagePreview.classList.remove('hidden');
  };

  if (file) {
    reader.readAsDataURL(file);
  }
});

// Drag and drop functionality
fileDrag.addEventListener('dragover', (event) => {
  event.preventDefault();
});

fileDrag.addEventListener('drop', (event) => {
  event.preventDefault();
  const file = event.dataTransfer.files[0];
  const reader = new FileReader();

  reader.onload = () => {
    imagePreview.src = reader.result;
    imagePreview.classList.remove('hidden');
  };

  if (file) {
    reader.readAsDataURL(file);
  }
});

// Submit the image for prediction
function submitImage() {
  if (imagePreview.src) {
    imageDisplay.src = imagePreview.src;
    predResult.textContent = 'Loading...';
    predResult.classList.remove('hidden');
    loader.classList.remove('hidden');

    // Replace this with your actual prediction logic
    setTimeout(() => {
      const prediction = Math.random() < 0.5 ? 'Pneumonia' : 'Normal';
      predResult.textContent = `Prediction: ${prediction}`;
      loader.classList.add('hidden');
    }, 2000);
  }
}

// Clear the image and prediction
function clearImage() {
  imagePreview.src = '';
  imagePreview.classList.add('hidden');
  imageDisplay.src = '';
  predResult.textContent = '';
  predResult.classList.add('hidden');
}