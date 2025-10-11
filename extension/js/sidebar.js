document.addEventListener('DOMContentLoaded', () => {
    // --- Element Selectors ---
    const dropArea = document.getElementById('drop-area');
    const previewImage = document.getElementById('preview-image');
    const analyzeButton = document.getElementById('analyze-button');
    const analysisResult = document.getElementById('analysis-result');
    const resultText = document.getElementById('result-text');
    const confidenceScore = document.getElementById('confidence-score');
    const explanation = document.getElementById('explanation');
    // New elements for URL analysis
    const urlInput = document.getElementById('url-input');
    const analyzeUrlButton = document.getElementById('analyze-url-button');

    let mediaFile = null;

    // --- Event Listeners ---
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });
    ['dragenter', 'dragover'].forEach(eventName => dropArea.addEventListener(eventName, () => dropArea.classList.add('dragover'), false));
    ['dragleave', 'drop'].forEach(eventName => dropArea.addEventListener(eventName, () => dropArea.classList.remove('dragover'), false));
    
    dropArea.addEventListener('drop', handleDrop, false);
    analyzeButton.addEventListener('click', handleFileAnalyze, false);
    analyzeUrlButton.addEventListener('click', handleUrlAnalyze, false); // New listener

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function showLoadingState(button, text) {
        button.disabled = true;
        button.textContent = 'Analyzing...';
        analysisResult.style.display = 'block';
        resultText.textContent = text;
        confidenceScore.textContent = '';
        explanation.textContent = '';
    }

    function showResults(data) {
        // Use 'let' and get raw numbers for the swap logic
        let realConfidence = data.confidence.real * 100;
        let manipulatedConfidence = data.confidence.manipulated * 100;
        
        // 1. Determine initial prediction
        let result = "Real";
        if (manipulatedConfidence > 20) {
            result = "Manipulated";
        }

        // 2. Apply your swap condition
        if (manipulatedConfidence < 50 && result === "Manipulated") {
            let temp = manipulatedConfidence;
            manipulatedConfidence = realConfidence;
            realConfidence = temp;
        }
        
        // 3. Display final results
        resultText.textContent = `Prediction: ${result}`;
        
        // Re-enabled to show the results of the swap
        confidenceScore.textContent = `Confidence -> Real: ${realConfidence.toFixed(2)}%, Manipulated: ${manipulatedConfidence.toFixed(2)}%`;
        
        explanation.textContent = "Analysis complete.";
    }

    

    function showError(error) {
        resultText.textContent = 'Analysis Failed';
        explanation.textContent = `Error: ${error.message}. Is the local server running?`;
    }

    function resetButtons() {
        analyzeButton.disabled = false;
        analyzeButton.textContent = 'Analyze File';
        analyzeUrlButton.disabled = false;
        analyzeUrlButton.textContent = 'Analyze URL';
    }

    function handleDrop(e) {
        // ... (This function is the same as before)
        const file = e.dataTransfer.files[0];
        if (!file) return;
        dropArea.innerHTML = `<p>Drag and drop an Image or Video here</p>`;
        analysisResult.style.display = 'none';
        if (file.type.startsWith('image/')) {
            mediaFile = file;
            const reader = new FileReader();
            reader.onload = e => { previewImage.src = e.target.result; };
            reader.readAsDataURL(mediaFile);
            previewImage.style.display = 'block';
            dropArea.style.display = 'none';
            analyzeButton.style.display = 'block';
        } else if (file.type.startsWith('video/')) {
            mediaFile = file;
            previewImage.style.display = 'none';
            dropArea.innerHTML = `<p><b>Video Selected:</b><br>${mediaFile.name}</p>`;
            dropArea.style.display = 'flex';
            analyzeButton.style.display = 'block';
        } else {
            alert('Please drop a valid image or video file.');
        }
    }

    async function handleFileAnalyze() {
        if (!mediaFile) return;

        showLoadingState(analyzeButton, 'Uploading file...');
        const formData = new FormData();
        formData.append('file', mediaFile);
        
        const isVideo = mediaFile.type.startsWith('video/');
        const endpoint = isVideo 
            ? 'http://localhost:8000/predict-video-from-upload/' 
            : 'http://localhost:8000/predict-image/';

        try {
            const response = await fetch(endpoint, { method: 'POST', body: formData });
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Server error');
            }
            const data = await response.json();
            showResults(data);
        } catch (error) {
            showError(error);
        } finally {
            resetButtons();
        }
    }

    // *** NEW FUNCTION TO HANDLE URL ANALYSIS ***
    async function handleUrlAnalyze() {
        const url = urlInput.value;
        if (!url || !url.startsWith('http')) {
            alert('Please enter a valid video URL.');
            return;
        }

        showLoadingState(analyzeUrlButton, 'Sending URL to server...');

        try {
            const response = await fetch('http://localhost:8000/predict-video-from-url/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ url: url }), // Send URL as JSON
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Server error');
            }
            const data = await response.json();
            showResults(data);
        } catch (error) {
            showError(error);
        } finally {
            resetButtons();
        }
    }
});