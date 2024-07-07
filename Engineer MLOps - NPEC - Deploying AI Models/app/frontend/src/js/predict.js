let selectedModel = null;
let selectedProcessedData = null;

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
    const modelList1 = document.getElementById('model-list-1');
    const modelList2 = document.getElementById('model-list-2');
    let selectedModel = null;
    let selectedProcessedData = null;
    const processedDataList = document.getElementById('processed-data');

    fetchModels(modelList1);
    fetchModels(modelList2);

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

document.getElementById('predict-data').addEventListener('click', async () => {
    try {
        const response = await fetch('http://localhost:8000/predict_folder/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                patches_folder: { patches_folder: selectedProcessedData },
                model_name: { model_name: selectedModel }
            })
        });

        if (!response.ok) {
            throw new Error('Network response was not ok ' + response.statusText);
        }

        const data = await response.json();
        console.log('Success:', data);
    } catch (error) {
        console.error('Error:', error);
    }
});


document.getElementById('predict-img').addEventListener('click', async () => {
    try {
        const response = await fetch('http://localhost:8000/predict_single_image/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model_name: selectedModel  // Directly passing the model name as a string
            })
        });

        if (!response.ok) {
            throw new Error('Network response was not ok ' + response.statusText);
        }

        const data = await response.json();
        console.log('Success:', data);
    } catch (error) {
        console.error('Error:', error);
    }
});

document.getElementById('upload-image').addEventListener('change', async (event) => {
    const files = event.target.files;
    const formData = new FormData();

    for (let i = 0; i < files.length; i++) {
        formData.append('images', files[i], files[i].name);
    }

    try {
        const response = await fetch('http://localhost:8000/upload_images/', {  // Adjusted to HTTP
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Network response was not ok ' + response.statusText);
        }

        const data = await response.json();
        console.log('Success:', data);
    } catch (error) {
        console.error('Error:', error);
    }
});