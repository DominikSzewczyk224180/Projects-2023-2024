
function selectButton(element) {
    // Deselect all buttons
    const buttons = document.querySelectorAll('.model-button');
    buttons.forEach(button => {
        button.classList.remove('selected');
    });

    // Select the clicked button
    element.classList
    .add('selected');
}
document.getElementById('autoTune').addEventListener('change', function() {
    const modelConfig = document.getElementById('modelConfig');
    modelConfig.disabled = this.checked;  // Disable or enable the textarea based on checkbox
    if (this.checked) {
        modelConfig.classList.add('disabled');  // Add 'disabled' class when checkbox is checked
    } else {
        modelConfig.classList.remove('disabled');  // Remove 'disabled' class when checkbox is unchecked
    }
});

document.getElementById('resetButton').addEventListener('click', function() {
    const defaultText = `"learning_rate": 0.001,
"batch_size": 32,
"epochs": 100,
"optimizer": "adam",
"metrics": ["accuracy"],
"early_stopping": {
    "monitor": "val_loss",
    "min_delta": 0.01,
    "patience": 10,
    "verbose": 1,
    "mode": "min",
    "restore_best_weights": True
}`;
    document.getElementById('modelConfig').value = defaultText;
});


document.addEventListener('DOMContentLoaded', function () {
    var dropdownMenu = document.querySelector('.dropdown-menu');
    dropdownMenu.addEventListener('click', function (e) {
        e.stopPropagation();  // This stops the dropdown from toggling on click
    });

    // Optional: Script to show selected items as button text
    document.querySelectorAll('.dropdown-menu input[type="checkbox"]').forEach(function(item) {
        item.addEventListener('change', function() {
            var allChecked = document.querySelectorAll('.dropdown-menu input[type="checkbox"]:checked');
            var btnLabel = document.getElementById('dropdownMenuButton');
            var text = Array.from(allChecked).map(item => item.parentElement.textContent.trim()).join(', ');
            btnLabel.textContent = text || 'Select Options';  // Change button text or set to default
        });
    });

        const uploadBox = document.getElementById('uploadBox');
        const fileInput = document.getElementById('fileInput');
    
        uploadBox.addEventListener('click', () => {
            fileInput.click();
        });
    
        uploadBox.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadBox.classList.add('dragover');
        });
    
        uploadBox.addEventListener('dragleave', () => {
            uploadBox.classList.remove('dragover');
        });
    
        uploadBox.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadBox.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                uploadFile(files[0]);
            }
        });
    
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                uploadFile(fileInput.files[0]);
            }
        });
    
        function uploadFile(file) {
            const formData = new FormData();
            formData.append('file', file);
    
            fetch('/upload', {
                method: 'POST',
                body: formData,
            })
            .then(response => response.json())
            .then(data => {
                console.log('File uploaded successfully:', data);
            })
            .catch(error => {
                console.error('Error uploading file:', error);
            });
        }

    
});