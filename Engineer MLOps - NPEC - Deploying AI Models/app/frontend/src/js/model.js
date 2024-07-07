let selectedModel = null;
let selectedProcessedData = null;
let selectedArchitecture = null;

const processedDataList = document.getElementById('processed-data');

function updateList(targetElement, info) {
    targetElement.innerHTML = ''; // Clear any existing content
    info.forEach(data => {
        const li = document.createElement('li');
        li.textContent = data;
        targetElement.appendChild(li);
    });
}

async function fetchDirectories(targetElement) {
    try {
        const response = await fetch('http://localhost:8000/list_directories/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ path: 'data/processed' }),
        });

        if (!response.ok) {
            throw new Error('Network response was not ok ' + response.statusText);
        }

        const data = await response.json();
        updateList(targetElement, data.directories);
        enableClickListenersData(targetElement);
        console.log(data.directories);
    } catch (error) {
        console.error('Error fetching directories:', error);
    }
}

async function fetchModels(modelList) {
    try {
        const response = await fetch('http://localhost:8000/models', {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            },
        });

        if (!response.ok) {
            throw new Error('Network response was not ok ' + response.statusText);
        }

        const data = await response.json();
        updateList(modelList, data);
        enableClickListenersModel(modelList);
        console.log(data);
    } catch (error) {
        console.error('Error fetching models:', error);
    }
}

document.addEventListener('DOMContentLoaded', function() {
    const modelList = document.getElementById('model-list');
    let selectedProcessedData = null;
    const processedDataList = document.getElementById('processed-data');

    fetchModels(modelList);
    fetchDirectories(processedDataList)
});


function enableClickListenersModel(listElement) {
    const items = listElement.querySelectorAll("li");
    items.forEach(item => {
        item.onclick = function() {
            if (item.classList.contains("active")) {
                item.classList.remove("active");
                selectedModel = null;
            } else {
                items.forEach(i => i.classList.remove("active"));
                item.classList.add("active");
                selectedModel = item.textContent;
                retrieveModelInfo(selectedModel)
            }
            console.log(`Selected Item: ${selectedModel}`);
        };
    });
}




function enableClickListenersData(listElement) {
    const items = listElement.querySelectorAll("li");
    items.forEach(item => {
        item.onclick = function() {
            if (item.classList.contains("active")) {
                item.classList.remove("active");
                selectedProcessedData = null;
            } else {
                items.forEach(i => i.classList.remove("active"));
                item.classList.add("active");
                selectedProcessedData = item.textContent;
            }
            console.log(`Selected dataset: ${selectedProcessedData}`);
        };
    });
}

async function retrieveModelInfo(selectedModel) {
    const modelSummaryTextarea = document.getElementById('model-summary');
    const modelLogsTextarea = document.getElementById('model-log');
    const loadingOverlay = document.getElementById('loading-overlay');

    function showLoadingPopup() {
        loadingOverlay.style.display = 'flex';
    }

    function hideLoadingPopup() {
        loadingOverlay.style.display = 'none';
    }

    if (!selectedModel) {
        console.error('No model selected.');
        modelSummaryTextarea.value = 'Default summary text';
        modelLogsTextarea.value = 'Default logs text';
        return;
    }

    try {
        showLoadingPopup();

        // Fetch model summary
        const summaryResponse = await fetch(`http://localhost:8000/models/${encodeURIComponent(selectedModel)}/summary`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            },
        });

        if (!summaryResponse.ok) {
            throw new Error('Network response was not ok ' + summaryResponse.statusText);
        }

        const summaryData = await summaryResponse.json();
        modelSummaryTextarea.value = summaryData.summary;

        // Fetch model logs
        const logsResponse = await fetch(`http://localhost:8000/models/{logs_name}/logs?model_name=${encodeURIComponent(selectedModel)}`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            },
        });

        if (!logsResponse.ok) {
            throw new Error('Network response was not ok ' + logsResponse.statusText);
        }

        const logsData = await logsResponse.json();
        modelLogsTextarea.value = logsData.logs.join('\n');

    } catch (error) {
        console.error('Error:', error);
        if (error.message.includes('summary')) {
            modelSummaryTextarea.value = 'Error retrieving model summary';
        } else if (error.message.includes('logs')) {
            modelLogsTextarea.value = 'Error retrieving model logs';
        }
    } finally {
        hideLoadingPopup();
    }
}


const resNet50Button = document.getElementById('res-net-50');
const resNet101Button = document.getElementById('res-net-101');

resNet50Button.addEventListener('click', function () {
    if (selectedArchitecture === 0) {
        selectedArchitecture = null;
        resNet50Button.classList.remove('active');
        console.log(selectedArchitecture)

    } else {
        selectedArchitecture = 0;
        resNet50Button.classList.add('active');
        resNet101Button.classList.remove('active');
        console.log(selectedArchitecture)

    }
});

resNet101Button.addEventListener('click', function () {
    if (selectedArchitecture === 1) {
        selectedArchitecture = null;
        resNet101Button.classList.remove('active');
        console.log(selectedArchitecture)

    } else {
        selectedArchitecture = 1;
        resNet101Button.classList.add('active');
        resNet50Button.classList.remove('active');
        console.log(selectedArchitecture)
    }
    const trainBtn = document.getElementById('start-training');

document.getElementById("start-training").addEventListener("click", function() {
    fetch("/train", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            depth_sel: selectedArchitecture,
            data_dir: selectedProcessedData
        })
    })
    .then(response => response.json())
    .then(data => {
        console.log(data.message);
    })
    .catch(error => {
        console.error("Error:", error);
    });
});

document.getElementById('download-summary').addEventListener('click', function() {
    downloadContent('model-summary', 'Model_Summary.txt');
});

document.getElementById('download-log').addEventListener('click', function() {
    downloadContent('model-log', 'Training_Log.txt');
});

function downloadContent(textareaId, filename) {
    var text = document.getElementById(textareaId).value;
    var blob = new Blob([text], { type: 'text/plain' });
    var url = window.URL.createObjectURL(blob);
    var element = document.createElement('a');
    element.setAttribute('href', url);
    element.setAttribute('download', filename);

    element.style.display = 'none';
    document.body.appendChild(element);

    element.click();

    document.body.removeChild(element);
    window.URL.revokeObjectURL(url);
}
});

