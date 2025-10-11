document.addEventListener('DOMContentLoaded', () => {
    const dropArea = document.getElementById('drop-area');
    const previewImage = document.getElementById('preview-image');
    const analyzeButton = document.getElementById('analyze-button');
    const analysisResult = document.getElementById('analysis-result');
    const resultText = document.getElementById('result-text');
    const confidenceScore = document.getElementById('confidence-score');
    const explanation = document.getElementById('explanation');

    let imageFile = null; // Variable to hold the image file

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    // Highlight drop zone
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => dropArea.classList.add('dragover'), false);
    });
    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => dropArea.classList.remove('dragover'), false);
    });

    // Handle dropped files
    dropArea.addEventListener('drop', handleDrop, false);
    analyzeButton.addEventListener('click', handleAnalyze, false);

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;

        if (files.length > 0 && files[0].type.startsWith('image/')) {
            imageFile = files[0]; // Store the file
            const reader = new FileReader();
            reader.onload = function(e) {
                previewImage.src = e.target.result;
                previewImage.style.display = 'block';
                dropArea.style.display = 'none';
                analyzeButton.style.display = 'block';
                analysisResult.style.display = 'none'; // Hide previous results
            };
            reader.readAsDataURL(imageFile);
        } else {
            alert('Please drop an image file.');
        }
    }

    async function handleAnalyze() {
        if (!imageFile) {
            alert("No image to analyze!");
            return;
        }

        analyzeButton.disabled = true;
        analyzeButton.textContent = 'Analyzing...';
        analysisResult.style.display = 'block';
        resultText.textContent = 'Sending to local server...';
        confidenceScore.textContent = '';
        explanation.textContent = '';

        const formData = new FormData();
        formData.append('file', imageFile);

        try {
            // This is the REAL API call to your local backend
            const response = await fetch('http://localhost:8000/predict/', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(`Server error: ${errorData.detail || response.statusText}`);
            }

            const data = await response.json();

            // Display the real results from the model
            let realScore = (data.confidence.real * 100).toFixed(2);
            let manipulatedScore = (data.confidence.manipulated * 100).toFixed(2);
            let predict = "real";
            if(manipulatedScore > 20){
                predict = "manipulated";
            }
            resultText.textContent = `Prediction: ${predict}`;
            confidenceScore.textContent = `Confidence -> Real: ${(data.confidence.real * 100).toFixed(2)}%, Manipulated: ${(data.confidence.manipulated * 100).toFixed(2)}%`;
            
            let labelsText = 'No objects detected.';
            if (data.labels && data.labels.length > 0) {
                labelsText = 'Detected Content: ' + data.labels.map(label => `${label.description} (${(label.confidence * 100).toFixed(1)}%)`).join(', ');
            }
            explanation.textContent = labelsText;

        } catch (error) {
            console.error('Analysis failed:', error);
            resultText.textContent = 'Analysis Failed';
            explanation.textContent = 'Could not connect to the local server. Make sure it is running and there are no CORS errors in the console.';
        } finally {
            analyzeButton.disabled = false;
            analyzeButton.textContent = 'Analyze';
        }
    }
});