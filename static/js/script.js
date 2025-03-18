console.log('script.js loaded');

let currentPath = null;
let folderModal = null;
let selectedFolder = null;
let ws = null;

// Initialize WebSocket connection
function connectWebSocket() {
    ws = new WebSocket(`ws://${window.location.host}/ws`);

    ws.onopen = function () {
        console.log('WebSocket connected');
    };

    ws.onmessage = function (event) {
        const status = JSON.parse(event.data);
        updateIndexingStatus(status);
    };

    ws.onclose = function () {
        console.log('WebSocket disconnected, attempting to reconnect...');
        setTimeout(connectWebSocket, 1000);
    };

    ws.onerror = function (error) {
        console.error('WebSocket error:', error);
    };
}

// Update indexing progress
function updateIndexingStatus(status) {
    const statusDiv = document.getElementById('indexingStatus');
    const progressBar = statusDiv.querySelector('.progress-bar');
    const details = document.getElementById('indexingDetails');

    if (status.status === 'idle') {
        // Fade out the status div
        statusDiv.style.opacity = '0';
        setTimeout(() => {
            statusDiv.style.display = 'none';
            statusDiv.style.opacity = '1';
        }, 500);
        return;
    }

    // Show and update the status
    statusDiv.style.display = 'block';
    statusDiv.style.opacity = '1';

    // Calculate progress percentage
    const percentage = status.total_files > 0
        ? Math.round((status.processed_files / status.total_files) * 100)
        : 0;

    progressBar.style.width = `${percentage}%`;
    progressBar.setAttribute('aria-valuenow', percentage);

    // Update status text
    let statusText = `Status: ${status.status}`;
    if (status.current_file) {
        statusText += ` | Current file: ${status.current_file}`;
    }
    if (status.total_files > 0) {
        statusText += ` | Progress: ${status.processed_files}/${status.total_files} (${percentage}%)`;
    }
    details.textContent = statusText;
}

// Initialize folder browser
async function initFolderBrowser() {
    folderModal = new bootstrap.Modal(document.getElementById('folderBrowserModal'));
    await loadFolderContents();
    await loadIndexedFolders();
}

// Open folder browser modal
function openFolderBrowser() {
    selectedFolder = null;
    folderModal.show();
    loadFolderContents();
}

function showDrives(breadcrumb, browser, data) {
    // Windows drives
    breadcrumb.innerHTML = '<li class="breadcrumb-item active">Drives</li>';
    data.drives.forEach(drive => {
        const escapedDrive = drive.replace(/\\/g, '\\\\').replace(/'/g, "\\'");
        browser.innerHTML += `
                    <div class="folder-item" onclick="loadFolderContents('${escapedDrive}')">
                        <i class="bi bi-hdd"></i>${drive}
                    </div>
                `;
    });
}

function showFolderContents(breadcrumb, browser, data) {
    // Folder contents
    currentPath = data.current_path;

    // Update breadcrumb
    const pathParts = currentPath.split(/[\\/]/);
    let currentBreadcrumb = '';
    pathParts.forEach((part, index) => {
        if (part) {
            // Check if the path contains backslashes to detect Windows
            const isWindows = currentPath.includes('\\');
            currentBreadcrumb += part + (isWindows ? '\\' : '/');
            const isLast = index === pathParts.length - 1;
            const escapedPath = currentBreadcrumb.replace(/\\/g, '\\\\').replace(/'/g, "\\'");
            breadcrumb.innerHTML += `
                                    <li class="breadcrumb-item ${isLast ? 'active' : ''}">
                                        ${isLast ? part : `<a href="#" onclick="loadFolderContents('${escapedPath}')">${part}</a>`}
                                    </li>
                                `;
        }
    });

    // Add parent directory
    if (data.parent_path) {
        addParentDirectory(browser, data);
    }

    // Add folders and files
    addFolderContents(browser, data);
}

function addParentDirectory(browser, data) {
    const escapedParentPath = data.parent_path.replace(/\\/g, '\\\\').replace(/'/g, "\\'");
    browser.innerHTML += `
                            <div class="folder-item" onclick="loadFolderContents('${escapedParentPath}')">
                                <i class="bi bi-arrow-up"></i>..
                            </div>
                        `;
}

function addFolderContents(browser, data) {
    data.contents.forEach(item => {
        const icon = item.type === 'directory' ? 'bi-folder' : 'bi-image';
        const escapedPath = item.path.replace(/\\/g, '\\\\').replace(/'/g, "\\'");
        browser.innerHTML += `
                                <div class="folder-item" onclick="${item.type === 'directory' ? `loadFolderContents('${escapedPath}')` : ''}" ondblclick="${item.type === 'directory' ? `selectFolder('${escapedPath}')` : ''}">
                                    <i class="bi ${icon}"></i>${item.name}
                                </div>
                            `;
    });
}

// Load folder contents
async function loadFolderContents(path = null) {
    try {
        const url = path ? `/browse/${encodeURIComponent(path)}` : '/browse';
        const response = await fetch(url);
        const data = await response.json();

        const browser = document.getElementById('folderBrowser');
        const breadcrumb = document.getElementById('folderBreadcrumb');

        browser.innerHTML = '';
        breadcrumb.innerHTML = '';

        if (data.drives) {
            showDrives(breadcrumb, browser, data);
        } else {
            showFolderContents(breadcrumb, browser, data);
        }
    } catch (error) {
        console.error('Error loading folder contents:', error);
    }
}

// Select folder for indexing
function selectFolder(path) {
    selectedFolder = path;
    addSelectedFolder();
}

// Add selected folder
async function addSelectedFolder() {

    folderModal.hide();

    if (!selectedFolder && currentPath) {
        selectedFolder = currentPath;
    }

    if (selectedFolder) {
        try {
            const encodedPath = encodeURIComponent(selectedFolder);
            const response = await fetch(`/folders?folder_path=${encodedPath}`, {
                method: 'POST'
            });

            if (response.ok) {
                await loadIndexedFolders();
                selectedFolder = null;
            } else {
                const error = await response.json();
                alert(`Error adding folder: ${error.detail || error.message || JSON.stringify(error)}`);
            }
        } catch (error) {
            console.error('Error adding folder:', error);
            alert('Error adding folder. Please try again.');
        }
    }
}

// Load indexed folders
async function loadIndexedFolders() {
    try {
        const response = await fetch('/folders');
        const folders = await response.json();

        const folderList = document.getElementById('folderList');
        folderList.innerHTML = '';

        folders.forEach(folder => {
            const escapedPath = folder.path.replace(/\\/g, '\\\\').replace(/'/g, "\\'");
            const item = document.createElement('a');
            item.href = '#';
            item.className = `list-group-item list-group-item-action ${folder.is_valid ? '' : 'text-danger'}`;
            item.innerHTML = `
                        <div class="d-flex justify-content-between align-items-center" style="gap: 8px;">
                            <div style="overflow: auto">
                                <i class="bi bi-folder"></i>
                                <span class="ms-2">${folder.path}</span>
                            </div>
                            <button class="btn btn-sm btn-outline-danger" onclick="removeFolder('${escapedPath}')">
                                <i class="bi bi-trash"></i>
                            </button>
                        </div>
                    `;
            folderList.appendChild(item);
        });

        // Load images from all folders
        await loadImages();
    } catch (error) {
        console.error('Error loading folders:', error);
    }
}

// Remove folder
async function removeFolder(path) {
    if (confirm('Are you sure you want to remove this folder?')) {
        try {
            const encodedPath = encodeURIComponent(path).replace(/%5C/g, '\\');
            const response = await fetch(`/folders/${encodedPath}`, {
                method: 'DELETE'
            });

            if (response.ok) {
                await loadIndexedFolders();
            } else {
                const error = await response.text();
                alert(`Error removing folder: ${error}`);
            }
        } catch (error) {
            console.error('Error removing folder:', error);
            alert('Error removing folder. Please try again.');
        }
    }
}

// Load images
async function loadImages(folder = null) {
    try {
        const url = folder ? `/images?folder=${encodeURIComponent(folder)}` : '/images';
        const response = await fetch(url);
        const images = await response.json();

        const imageGrid = document.getElementById('imageGrid');
        imageGrid.innerHTML = '';

        images.forEach(image => {
            const card = document.createElement('div');
            card.className = 'image-card';
            card.innerHTML = `
                        <img src="/files/${encodeURIComponent(image.root_folder)}/${encodeURIComponent(image.path)}" 
                             alt="${image.path}"
                             loading="lazy">
                    `;
            imageGrid.appendChild(card);
        });
    } catch (error) {
        console.error('Error loading images:', error);
    }
}

// Get current folder path
function getCurrentPath() {
    // Return the current path if we're in a folder, otherwise null
    return currentPath;
}

// Search images
async function searchImages(event) {
    event.preventDefault();
    const query = document.getElementById('searchInput').value;
    if (!query) return;

    try {
        // Only include folder parameter if we're inside the folder browser
        const searchUrl = `/search/text?query=${encodeURIComponent(query)}`;
        const response = await fetch(searchUrl);
        const results = await response.json();

        const imageGrid = document.getElementById('imageGrid');
        imageGrid.innerHTML = '';

        if (results.length === 0) {
            imageGrid.innerHTML = '<div class="no-results text-center p-4">No images found matching your search.</div>';
            return;
        }

        results.forEach(result => {
            const card = document.createElement('div');
            card.className = 'image-card';
            card.innerHTML = `
                        <img src="/files/${encodeURIComponent(result.root_folder)}/${encodeURIComponent(result.path)}" 
                             alt="${result.path}"
                             loading="lazy">
                        <div class="similarity-score">${result.similarity}%</div>
                    `;
            imageGrid.appendChild(card);
        });
    } catch (error) {
        console.error('Error searching images:', error);
        const imageGrid = document.getElementById('imageGrid');
        imageGrid.innerHTML = '<div class="error text-center p-4">An error occurred while searching. Please try again.</div>';
    }
}

// Search by image
async function searchByImage(event) {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
        const searchUrl = '/search/image';
        const response = await fetch(searchUrl, {
            method: 'POST',
            body: formData
        });
        const results = await response.json();

        const imageGrid = document.getElementById('imageGrid');
        imageGrid.innerHTML = '';

        if (results.length === 0) {
            imageGrid.innerHTML = '<div class="no-results text-center p-4">No similar images found.</div>';
            return;
        }

        results.forEach(result => {
            const card = document.createElement('div');
            card.className = 'image-card';
            card.innerHTML = `
                        <img src="/files/${encodeURIComponent(result.root_folder)}/${encodeURIComponent(result.path)}" 
                             alt="${result.path}"
                             loading="lazy">
                        <div class="similarity-score">${result.similarity}%</div>
                    `;
            imageGrid.appendChild(card);
        });

        // Reset file input
        event.target.value = '';
    } catch (error) {
        console.error('Error searching by image:', error);
        const imageGrid = document.getElementById('imageGrid');
        imageGrid.innerHTML = '<div class="error text-center p-4">An error occurred while searching. Please try again.</div>';
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    connectWebSocket();
    initFolderBrowser();
});