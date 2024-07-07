let selectedRawData = null;
let selectedProcessedData = null;

function enableClickListeners(listElement, type) {
    const items = listElement.querySelectorAll("li");
    items.forEach(item => {
        item.onclick = function() {
            if (item.classList.contains("active")) {
                item.classList.remove("active");
                if (type === 'raw') {
                    selectedRawData = null;
                    get_raw_data_info(selectedRawData);
                } else if (type === 'processed') {
                    selectedProcessedData = null;
                }
            } else {
                items.forEach(i => i.classList.remove("active"));
                item.classList.add("active");
                if (type === 'raw') {
                    selectedRawData = item.textContent;
                    get_raw_data_info(selectedRawData);
                } else if (type === 'processed') {
                    selectedProcessedData = item.textContent;
                }
            }
            console.log(`Selected Raw Data: ${selectedRawData}`);
            console.log(`Selected Processed Data: ${selectedProcessedData}`);
        };
    });
}

async function get_raw_data_info(dir) {
    try {
        const response = await fetch('http://localhost:8000/get_raw_data_info/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ path: dir }),
        });

        if (!response.ok) {
            throw new Error('Network response was not ok ' + response.statusText);
        }

        const data = await response.json();
        displayRawInfo(data);
        const classList = document.getElementById('class-list');
        updateClassList(classList, data.classes);
        console.log(data.images_count);
    } catch (error) {
        console.error('Error fetching directories:', error);
    }
}

function displayRawInfo(data) {
    document.getElementById('image-count').textContent = `Amount of images found: ${data.images_count}`;
    document.getElementById('mask-count').textContent = `Amount of masks found: ${data.masks_count}`;
    document.getElementById('class-count').textContent = `Amount of classes found: ${data.class_count}`;
}

function updateList(targetElement, info) {
    targetElement.innerHTML = ''; // Clear any existing content
    info.forEach(data => {
        const li = document.createElement('li');
        li.textContent = data;
        targetElement.appendChild(li);
    });
}
function updateClassList(targetElement, info) {
    targetElement.innerHTML = ''; // Clear any existing content

    info.forEach(data => {
        // Create list item as dropdown item
        const li = document.createElement('li');
        li.className = 'dropdown-item';

        // Create checkbox
        const input = document.createElement('input');
        input.type = 'checkbox';
        input.id = data;
        input.value = data;

        // Create label
        const label = document.createElement('label');
        label.htmlFor = data;
        label.textContent = data;
        label.style.cursor = 'pointer'; // Make the cursor a pointer when hovering over the label

        // Append checkbox and label to list item
        li.appendChild(input);
        li.appendChild(label);

        // Append list item to the target element
        targetElement.appendChild(li);
    });
}

function updateClassList(targetElement, info) {
    targetElement.innerHTML = ''; // Clear any existing content

    info.forEach(data => {
        // Create list item as dropdown item
        const li = document.createElement('li');
        li.className = 'dropdown-item';

        // Create checkbox
        const input = document.createElement('input');
        input.type = 'checkbox';
        input.id = data;
        input.value = data;

        // Create label
        const label = document.createElement('label');
        label.htmlFor = data;
        label.textContent = data;
        label.style.cursor = 'pointer'; // Make the cursor a pointer when hovering over the label

        // Append checkbox and label to list item
        li.appendChild(input);
        li.appendChild(label);

        // Append list item to the target element
        targetElement.appendChild(li);
    });

    // Prevent dropdown from closing when clicking inside
    document.querySelector('.dropdown-menu').addEventListener('click', function(e) {
        e.stopPropagation();
    });
}


async function fetchDirectories(targetElement, dir, type) {
    try {
        const response = await fetch('http://127.0.0.1:8000/list_directories/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ path: dir }),
        });

        if (!response.ok) {
            throw new Error('Network response was not ok ' + response.statusText);
        }

        const data = await response.json();
        updateList(targetElement, data.directories);
        enableClickListeners(targetElement, type);
        console.log(data.directories);
    } catch (error) {
        console.error('Error fetching directories:', error);
    }
}

function refreshDirectories() {
    const rawDataList = document.getElementById('raw-data');
    const processedDataList = document.getElementById('processed-data');
    fetchDirectories(rawDataList, 'data/raw', 'raw');
    fetchDirectories(processedDataList, 'data/processed', 'processed');
}

document.addEventListener('DOMContentLoaded', function() {
    refreshDirectories(); // Initial fetch of directories

    const deleteProcessedButton = document.getElementById('delete-processed-data');

    const deleteRawButton = document.getElementById('delete-raw-data');
    const processButton = document.getElementById('process-data');

    const loadingOverlay = document.getElementById('loading-overlay');

    function showLoadingPopup() {
        loadingOverlay.style.display = 'flex';
    }

    function hideLoadingPopup() {
        loadingOverlay.style.display = 'none';
    }

    if (deleteProcessedButton) {
        deleteProcessedButton.addEventListener('click', async function(event) {
            event.preventDefault();
            this.disabled = true; // Disable the button to prevent multiple clicks
            showLoadingPopup();

            try {
                const response = await fetch(`http://localhost:8000/delete_data/processed/${encodeURIComponent(selectedProcessedData)}`, {
                    method: 'DELETE'
                });
                if (response.ok) {
                    alert('Directory deleted successfully');
                    refreshDirectories();
                } else {
                    alert(`Failed to delete: ${response.statusText}`);
                }
            } catch (error) {
                alert('Error:', error);
            } finally {
                hideLoadingPopup();
                this.disabled = false; // Re-enable the button
            }
        });
    }

    if (deleteRawButton) {
        deleteRawButton.addEventListener('click', async function(event) {
            event.preventDefault();
            this.disabled = true; // Disable the button to prevent multiple clicks
            showLoadingPopup();

            try {
                const response = await fetch(`http://localhost:8000/delete_data/raw/${encodeURIComponent(selectedRawData)}`, {
                    method: 'DELETE'
                });
                if (response.ok) {
                    alert('Directory deleted successfully');
                    refreshDirectories();
                } else {
                    alert(`Failed to delete: ${response.statusText}`);
                }
            } catch (error) {
                alert('Error:', error);
            } finally {
                hideLoadingPopup();
                this.disabled = false; // Re-enable the button
            }
        });
    }

    if (processButton) {
        processButton.addEventListener('click', async function(event) {
            event.preventDefault();
            this.disabled = true;

            const selectedClasses = [];
            document.querySelectorAll('#class-list input[type="checkbox"]:checked').forEach(checkbox => {
                selectedClasses.push(checkbox.value);
            });
            console.log(selectedClasses);

            const response = await handleFetch(`http://localhost:8000/process_data/${encodeURIComponent(selectedRawData)}`, {
                method: "POST",
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    masks: selectedClasses
                })
            });

            if (response) {
                const data = await response.json();
                alert(data.message);
                refreshDirectories();
            }

            this.disabled = false;
        });
    }



    // █░█ █▀█ █░░ █▀█ ▄▀█ █▀▄   ▀█ █ █▀█   
    // █▄█ █▀▀ █▄▄ █▄█ █▀█ █▄▀   █▄ █ █▀▀
    const uploadBox = document.getElementById('uploadBox');
    const fileInput = document.getElementById('fileInput');

    // Handle click on the upload box
    uploadBox.addEventListener('click', () => {
        fileInput.click();
    });

    // Handle file selection
    fileInput.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (file) {
            uploadFile(file);
        }
    });

    // Handle drag and drop
    uploadBox.addEventListener('dragover', (event) => {
        event.preventDefault();
        event.stopPropagation();
        uploadBox.classList.add('dragging');
    });

    uploadBox.addEventListener('dragleave', (event) => {
        event.preventDefault();
        event.stopPropagation();
        uploadBox.classList.remove('dragging');
    });

    uploadBox.addEventListener('drop', (event) => {
        event.preventDefault();
        event.stopPropagation();
        uploadBox.classList.remove('dragging');

        const file = event.dataTransfer.files[0];
        if (file) {
            uploadFile(file);
        }
    });

    function uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);
    
        fetch('http://localhost:8000/upload/', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.text().then(text => { throw new Error(text) });
            }
            return response.json();
        })
        .then(data => {
            alert('Success: ' + JSON.stringify(data));
        })
        .catch((error) => {
            alert('Error: ' + error.message);
        });
    }


});


async function fetchInitialData() {
    // Simulate fetching data with a delay
    return new Promise(resolve => setTimeout(resolve, 2000)); // Simulates a fetch delay
}

