document.addEventListener('DOMContentLoaded', (event) => {
    // DOM elements
    const predictionList = document.getElementById('prediction-list');
    const container = document.querySelector('.image-container');
    const heatmap1 = document.getElementById('heatmap1');
    const heatmap2 = document.getElementById('heatmap2');
    const heatmap3 = document.getElementById('heatmap3');
    const orgImg = document.getElementById('org-img');
    const predictionImg = document.getElementById('prediction');
    const prevBtn = document.getElementById('prev-img-btn');
    const nextBtn = document.getElementById('next-img-btn');
    const imageCounter = document.getElementById('image-counter');

    let imageList = [];
    let selectedPredictions = null;
    let currentIndex = 0;

    // Fetch directories and image list
    fetchDirectories(predictionList);

    // Heatmap display logic on mousemove
    container.addEventListener('mousemove', function(event) {
        const containerWidth = container.offsetWidth;
        const mouseX = event.clientX - container.getBoundingClientRect().left;
        const thirdWidth = containerWidth / 3;
        if (orgImg.style.display === 'block'){
            if (mouseX < thirdWidth) {
                heatmap1.style.display = 'block';
                heatmap2.style.display = 'none';
                heatmap3.style.display = 'none';
            } else if (mouseX < thirdWidth * 2) {
                heatmap1.style.display = 'none';
                heatmap2.style.display = 'block';
                heatmap3.style.display = 'none';
            } else {
                heatmap1.style.display = 'none';
                heatmap2.style.display = 'none';
                heatmap3.style.display = 'block';
            }
        }
    });

    // Hide heatmaps on mouseleave
    container.addEventListener('mouseleave', function() {
        heatmap1.style.display = 'none';
        heatmap2.style.display = 'none';
        heatmap3.style.display = 'none';
    });

    // Image navigation buttons
    prevBtn.addEventListener('click', () => {
        if (imageList.length > 0) {
            if (currentIndex > 0) {
                currentIndex--;
            } else {
                currentIndex = imageList.length - 1; // Loop to the last image
            }
            updateImages();
        }
    });

    nextBtn.addEventListener('click', () => {
        if (imageList.length > 0) {
            if (currentIndex < imageList.length - 1) {
                currentIndex++;
            } else {
                currentIndex = 0; // Loop to the first image
            }
            updateImages();
        }
    });

    // Display filename on image counter hover
    imageCounter.addEventListener('mouseover', () => {
        imageCounter.textContent = getFilenameFromPath(orgImg.src);
    });

    imageCounter.addEventListener('mouseout', () => {
        updateImageCounter();
    });

    // Fetch the list of directories
    async function fetchDirectories(targetElement) {
        try {
            const response = await fetch('http://localhost:8000/list_directories/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ path: 'data/predictions' }),
            });

            if (!response.ok) {
                throw new Error('Network response was not ok ' + response.statusText);
            }

            const data = await response.json();
            updateList(targetElement, data.directories);
            enableClickListenersData(targetElement);
        } catch (error) {
            console.error('Error fetching directories:', error);
        }
    }

    // Update the directory list in the DOM
    function updateList(targetElement, info) {
        targetElement.innerHTML = '';
        info.forEach(data => {
            const li = document.createElement('li');
            li.textContent = data;
            targetElement.appendChild(li);
        });
    }

    // Enable click listeners on directory list items
    function enableClickListenersData(listElement) {
        const items = listElement.querySelectorAll("li");
        if (items.length > 0) {
            items[0].classList.add("active");
            selectedPredictions = items[0].textContent;
            fetchImageList(selectedPredictions);
        }

        items.forEach(item => {
            item.onclick = function() {
                items.forEach(i => i.classList.remove("active"));
                if (!item.classList.contains("active")) {
                    item.classList.add("active");
                    selectedPredictions = item.textContent;
                } else {
                    selectedPredictions = null;
                }
                fetchImageList(selectedPredictions);
            };
        });
    }

    // Fetch the image list based on the selected prediction
    async function fetchImageList(selectedPredictions) {
        if (!selectedPredictions) return;

        try {
            const response = await fetch(`http://localhost:8000/list_images/${selectedPredictions}`);
            if (response.ok) {
                const data = await response.json();
                imageList = data.images || [];
                currentIndex = 0;
                updateImages();
            } else {
                console.error("Failed to fetch image list:", response.status);
            }
        } catch (error) {
            console.error("Error fetching image list:", error);
        }
    }

    // Update the displayed images
    function updateImages() {
        if (imageList.length > 0) {
            orgImg.src = imageList[currentIndex];
            predictionImg.src = imageList[currentIndex].replace('ORG_', 'COMB_');
            heatmap1.src = imageList[currentIndex].replace('ORG_', 'PROB_mask1_');
            heatmap2.src = imageList[currentIndex].replace('ORG_', 'PROB_mask2_');
            heatmap3.src = imageList[currentIndex].replace('ORG_', 'PROB_mask3_');
            updateImageCounter();
        } else {
            console.error("No images found in the list");
        }
    }

    function updateImageCounter() {
        let filePath = orgImg.src; // Get the current image's file path
        let status;
        let disableForm = false;
    
        if (filePath.includes('model_correct')) {
            status = `<span style="color: green;">correct</span>`;
            disableForm = true;
        } else if (filePath.includes('model_incorrect')) {
            status = `<span style="color: red;">incorrect</span>`;
            disableForm = true;
        } else if (filePath.includes('user_uploads_correct')) {
            status = `<span style="color: blue;">uploaded</span>`;
            disableForm = true;
        } else {
            status = `<span style="color: gray;">unchecked</span>`;
        }
        
        imageCounter.innerHTML = `Image (${currentIndex + 1}/${imageList.length}) ${status}`;
    
        // Get feedback form and its elements
        const feedbackForm = document.getElementById('feedback-form');
        if (feedbackForm) {
            const checkboxes = feedbackForm.querySelectorAll('.form-check-input');
            const submitButton = document.getElementById('submit-feedback');
            if (disableForm) {
                feedbackForm.style.pointerEvents = 'none';
                feedbackForm.style.cursor = 'not-allowed';
                checkboxes.forEach(checkbox => checkbox.disabled = true);
                if (submitButton) {
                    submitButton.disabled = true;
                    submitButton.classList.add('disabled-button');
                }
            } else {
                feedbackForm.style.pointerEvents = 'auto';
                feedbackForm.style.cursor = 'auto';
                checkboxes.forEach(checkbox => checkbox.disabled = false);
                if (submitButton) {
                    submitButton.disabled = false;
                    submitButton.classList.remove('disabled-button');
                }
            }
        } else {
            console.error('Feedback form element not found.');
        }
    }
    


    // Get filename from path
    function getFilenameFromPath(path) {
        return path.split('/').pop();
    }

const downloadPredictionsButton = document.getElementById('download-predictions');

downloadPredictionsButton.addEventListener('click', async function() {
    this.disabled = true; // Disable the button to prevent multiple clicks
    const loadingOverlay = document.getElementById('loading-overlay');

    function showLoadingPopup() {
        loadingOverlay.style.display = 'flex';
    }

    function hideLoadingPopup() {
        loadingOverlay.style.display = 'none';
    }

    try {
        showLoadingPopup();

        const folder = encodeURIComponent(selectedPredictions); // Assuming selectedFolder is set somewhere in your script
        const response = await fetch(`http://localhost:8000/download-predictions/?folder=${folder}`, {
            method: 'GET'
        });


        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = `${decodeURIComponent(folder)}.zip`;  // Set the download attribute
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);  // Clean up
            window.URL.revokeObjectURL(url);
        } else {
            const errorText = await response.text();
            console.error(`Failed to download: ${response.status} - ${response.statusText}: ${errorText}`);
            alert(`Failed to download: ${response.status} - ${response.statusText}`);
        }
    } catch (error) {
        console.error('Error:', error);
        alert(`Error: ${error.message}`);
    } finally {
        hideLoadingPopup();
        this.disabled = false; // Re-enable the button after operation
    }
});


    document.getElementById('download-img-pred-btn').addEventListener('click', async () => {
        const orgImgSrc = document.getElementById('org-img').src;
        const parentDir = orgImgSrc.split('/').slice(-2, -1)[0];  // Get the parent directory
        const filenameWithExt = orgImgSrc.split('/').pop();       // Get the filename with extension
        const filenameWithoutExt = filenameWithExt.replace(/\.[^/.]+$/, "");  // Remove the extension
        const filename = `${parentDir}_${filenameWithoutExt.substring(4)}`;  // Exclude the first 4 characters

        // Get the basename of the src
        const imageId2 = orgImgSrc.substring(orgImgSrc.lastIndexOf('/') + 1);
        
        // Remove the first 4 characters and the extension
        const imageId = imageId2.substring(4, imageId2.lastIndexOf('.'));
        const formData = new FormData();
        console.log(imageId)
        formData.append('dataset_model', selectedPredictions);
        formData.append('imageId', imageId);

        try {
            const response = await fetch('http://localhost:8000/download_mask/', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                a.download = `${filename}_preds.zip`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
            } else {
                console.error('Failed to download mask:', response.statusText);
            }
        } catch (error) {
            console.error('Error:', error);
        }
    });

    document.getElementById("submit-feedback").addEventListener("click", function(event) {
        event.preventDefault();  // Prevent default form submission behavior
        var datasetModel = selectedPredictions; // Set by your script
        const orgImgSrc = decodeURIComponent(document.getElementById('org-img').src); // Decode URI component to handle %20
        const filenameWithExt = orgImgSrc.split('/').pop();       // Get the filename with extension
        const filenameWithoutExt = filenameWithExt.replace(/\.[^/.]+$/, "");  // Remove the extension
        const filename = `${filenameWithoutExt.substring(4)}`;  // Exclude the first 4 characters

        var feedback = "";
    
        if (document.getElementById("mark-as-correct").checked) {
            feedback = "correct";
        } else if (document.getElementById("mark-as-false").checked) {
            feedback = "incorrect";
        } else {
            alert("Please mark as correct or incorrect.");
            return;
        }
    
        fetch('http://localhost:8000/set_feedback/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                dataset_model: datasetModel,
                imageId: filename, // Replace spaces with underscores
                feedback: feedback
            }),
        })
        .then(response => response.json())
        .then(data => {
            console.log('Success:', data);
        })
        .catch((error) => {
            console.error('Error:', error);
        });
    });

   

});
