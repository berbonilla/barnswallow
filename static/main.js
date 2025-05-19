// 🚀 Set the height for all charts
const CHART_HEIGHT = 400;
const CHART_WIDTH = "100%";  

// First code base - Migration Forecast System
const API_BASE_URL = "http://127.0.0.1:8001";

// WebSocket for Real-time Updates
let trainingSocket = null;
let trainingLog = [];
const API_BASE_URL_2 = API_BASE_URL;
const statusMessageEl = document.getElementById('statusMessage');
const modelSelect = document.getElementById('model_name');
const dataSelect = document.getElementById('data_name');

const modelDropdown = document.getElementById('model-select');
const maeCtx = document.getElementById('maeChart').getContext('2d');
const accuracyCtx = document.getElementById('accuracyChart').getContext('2d');
let maeChart, accuracyChart;

if (!modelDropdown) {
  console.error("[ERROR] The <select> element with ID 'model-select' was not found in the DOM.");
} else {
  console.log("[INFO] <select> element found successfully.");
}

async function fetchModels() {
  console.log("[INFO] Fetching models from:", `${API_BASE_URL}/models`);

  try {
    const response = await fetch(`${API_BASE_URL}/models`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
    });

    console.log("[DEBUG] Response status:", response.status);
    if (!response.ok) {
      console.error(`[ERROR] Failed to fetch models. Status: ${response.status}`);
      alert(`Failed to load models. Status: ${response.status}`);
      return;
    }

    const models = await response.json();
    console.log("[INFO] Models fetched successfully:", models);

    if (!modelDropdown) {
      console.error("[ERROR] <select> element is not available in the DOM.");
      alert("Dropdown element not found!");
      return;
    }

    modelDropdown.innerHTML = '<option value="">Select a model...</option>';

    models.forEach((model, index) => {
      const option = document.createElement('option');
      option.value = model.id;
      option.textContent = model.name;
      modelDropdown.appendChild(option);
    });

    console.log("[INFO] Model dropdown populated successfully.");

  } catch (error) {
    console.error("[ERROR] Exception during model fetch:", error);
    alert(`Exception occurred: ${error.message}`);
  }
}

fetchModels();

modelDropdown.addEventListener('change', (e) => {
  const selectedModelId = e.target.value;
  if (selectedModelId) {
    console.log(`[INFO] Model selected: ${selectedModelId}`);
    fetchModelMetrics(selectedModelId);
  }
});

// 🚀 **Fetch Model Metrics and Render Graphs**
async function fetchModelMetrics(modelId) {
  console.log(`[INFO] Fetching metrics for model ID: ${modelId}`);
  
  try {
    const response = await fetch(`${API_BASE_URL}/model/${modelId}`);
    
    if (!response.ok) {
      console.error(`[ERROR] Failed to fetch model metrics. Status: ${response.status}`);
      alert(`Failed to load model metrics. Status: ${response.status}`);
      return;
    }

    const modelData = await response.json();
    console.log("[INFO] Model metrics fetched successfully:", modelData);

    // 🚀 **No need for JSON.parse, it is already an object**
    const metrics = modelData.metrics;

    console.log("[DEBUG] Metrics Object:", metrics);

    // **Prepare Data for Graphs**
    const labels = Object.keys(metrics);
    const maeValues = labels.map(key => metrics[key].MAE);
    const accuracyValues = maeValues.map(mae => Math.max(0, (1 - mae) * 100));

    console.log("[DEBUG] MAE Values:", maeValues);
    console.log("[DEBUG] Accuracy Values:", accuracyValues);

    // **Destroy Previous Charts if Exist**
    if (maeChart) maeChart.destroy();
    if (accuracyChart) accuracyChart.destroy();

    // **Render MAE Chart**
    maeChart = new Chart(maeCtx, {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          label: 'MAE',
          data: maeValues,
          backgroundColor: '#4f46e5',
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false
      }
    });

    // **Render Accuracy Chart**
    accuracyChart = new Chart(accuracyCtx, {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          label: 'Accuracy (%)',
          data: accuracyValues,
          backgroundColor: '#10b981',
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false
      }
    });

    console.log("[INFO] Graphs rendered successfully.");

  } catch (error) {
    console.error("[ERROR] Exception during metric fetch:", error);
  }
}

const modelDropdownHeatmap = document.getElementById('model-select-heatmap');
const heatmapGraphCtx = document.getElementById('heatmapGraph').getContext('2d');
const trainingLossGraphCtx = document.getElementById('trainingLossGraph').getContext('2d');
const actualVsPredictedContainer = document.getElementById('actual-predicted-container');
let heatmapChart, trainingLossChart;


// 🚀 Resize Observer to handle dynamic resizing
const resizeObserver = new ResizeObserver(entries => {
    entries.forEach(entry => {
        if (entry.contentBoxSize) {
            const width = entry.contentRect.width;
            const canvases = entry.target.querySelectorAll('canvas');
            canvases.forEach(canvas => {
                canvas.width = width;
            });
        }
    });
});

resizeObserver.observe(document.getElementById('heatmap-charts-container'));
resizeObserver.observe(document.getElementById('actual-predicted-container'));

async function fetchModelsForHeatmap() {
    console.log("[INFO] Fetching models for Heatmap from:", `${API_BASE_URL}/models`);

    try {
        const response = await fetch(`${API_BASE_URL}/models`, {
            method: 'GET',
            headers: {
                'Accept': 'application/json',
            },
        });

        const models = await response.json();
        console.log("[INFO] Models fetched successfully for Heatmap:", models);

        modelDropdownHeatmap.innerHTML = '<option value="">Select a model...</option>';

        models.forEach((model, index) => {
            const option = document.createElement('option');
            option.value = model.id;
            option.textContent = model.name;
            modelDropdownHeatmap.appendChild(option);
        });

    } catch (error) {
        console.error("[ERROR] Exception during model fetch for Heatmap:", error);
    }
}

fetchModelsForHeatmap();

modelDropdownHeatmap.addEventListener('change', (e) => {
    const selectedModelId = e.target.value;
    if (selectedModelId) {
        console.log(`[INFO] Model selected for Heatmap: ${selectedModelId}`);
        fetchModelMetricsForHeatmap(selectedModelId);
    }
});

async function fetchModelMetricsForHeatmap(modelId) {
    console.log(`[INFO] Fetching metrics for Heatmap with model ID: ${modelId}`);

    try {
        const response = await fetch(`${API_BASE_URL}/model/${modelId}`);
        if (!response.ok) {
            console.error(`[ERROR] Failed to fetch model metrics for Heatmap. Status: ${response.status}`);
            alert(`Failed to load model metrics. Status: ${response.status}`);
            return;
        }

        const modelData = await response.json();
        const metrics = modelData.metrics;

        // 🚀 **Prepare Heatmap Data**
        const labels = Object.keys(metrics);
        const maeValues = labels.map(key => metrics[key].MAE);
        const rmseValues = labels.map(key => metrics[key].RMSE);
        const mseValues = labels.map(key => metrics[key].MSE);

        const heatmapData = {
            labels,
            datasets: [
                { label: 'MAE', data: maeValues, backgroundColor: '#4f46e5' },
                { label: 'RMSE', data: rmseValues, backgroundColor: '#34d399' },
                { label: 'MSE', data: mseValues, backgroundColor: '#f87171' }
            ]
        };

        // 🟢 **Apply responsive width and height**
        const heatmapCanvas = document.getElementById('heatmapGraph');
        heatmapCanvas.style.height = `${CHART_HEIGHT}px`;
        heatmapCanvas.style.width = CHART_WIDTH;

        if (heatmapChart) heatmapChart.destroy();
        heatmapChart = new Chart(heatmapGraphCtx, {
            type: 'bar',
            data: heatmapData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            stepSize: 0.1   // Smaller steps for readability
                        }
                    },
                    x: {
                        ticks: {
                            autoSkip: false,  // Prevents label skipping
                            maxRotation: 45,  // Rotates for better visibility
                            minRotation: 0
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top'
                    },
                    tooltip: {
                        enabled: true
                    }
                }
            }
        });

        // 🟢 **Training Loss Graph**
        const trainingCanvas = document.getElementById('trainingLossGraph');
        trainingCanvas.style.height = `${CHART_HEIGHT}px`;
        trainingCanvas.style.width = CHART_WIDTH;

        if (trainingLossChart) trainingLossChart.destroy();
        trainingLossChart = new Chart(trainingLossGraphCtx, {
            type: 'line',
            data: {
                labels: Array.from({ length: 100 }, (_, i) => i + 1),
                datasets: [{
                    label: 'Training Loss',
                    data: Array.from({ length: 100 }, () => Math.random() * 0.1), // Placeholder data
                    borderColor: '#f97316',
                    fill: false,
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            stepSize: 0.01
                        }
                    }
                }
            }
        });

    } catch (error) {
        console.error("[ERROR] Exception during metric fetch for Heatmap:", error);
    }
}

// This is after initializing the map and heat layer
const canvas = document.querySelector('canvas'); // Get the first canvas element (assuming there is only one for heatmap)
if (canvas) {
    canvas.willReadFrequently = true; // Set willReadFrequently flag
}

let migrationMapInstance = null;
let routesMapInstance = null;
let patternsMapInstance = null;
let lastForecastResults = []; // Globally save so we can build maps lazily

function setStatus(message, isError = false) {
    statusMessageEl.textContent = message;
    statusMessageEl.className = isError ? 'error' : 'success';
    statusMessageEl.style.display = 'block';
}

async function loadAssets() {
    try {
        const response = await fetch(`${API_BASE_URL}/list_assets`);
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ error: `HTTP error! status: ${response.status}` }));
            throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
        }
        const assets = await response.json();
        modelSelect.innerHTML = '';
        dataSelect.innerHTML = '';
        if (assets.models && assets.models.length > 0) {
            assets.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = model;
                modelSelect.appendChild(option);
            });
        } else {
                modelSelect.innerHTML = `<option value="">No models found</option>`;
        }
        if (assets.datafiles && assets.datafiles.length > 0) {
            assets.datafiles.forEach(datafile => {
                const option = document.createElement('option');
                option.value = datafile;
                option.textContent = datafile;
                dataSelect.appendChild(option);
            });
        } else {
            dataSelect.innerHTML = `<option value="">No data files found</option>`;
        }
    } catch (error) {
        setStatus(`Error loading assets: ${error.message}`, true);
        modelSelect.innerHTML = `<option value="">Error loading</option>`;
        dataSelect.innerHTML = `<option value="">Error loading</option>`;
    }
}

function clearStatus() {
    statusMessageEl.textContent = '';
    statusMessageEl.style.display = 'none';
}

function destroyMap(mapInstance) {
    if (mapInstance) {
        mapInstance.remove();
    }
    return null;
}

function clearResults() {
    document.getElementById('arrival-content').innerHTML = '';
    document.getElementById('environment-content').innerHTML = '';
    document.getElementById('timeline-content').innerHTML = '';
    document.getElementById('weather-content').innerHTML = '';
    document.getElementById('routes-content-details').innerHTML = '';
    document.getElementById('map-content-details').innerHTML = '';
    document.getElementById('patterns-content-details').innerHTML = '';
    migrationMapInstance = destroyMap(migrationMapInstance);
    routesMapInstance = destroyMap(routesMapInstance);
    patternsMapInstance = destroyMap(patternsMapInstance);
    document.getElementById('migration-map-container').innerHTML = '';
    document.getElementById('routes-map-container').innerHTML = '';
    document.getElementById('patterns-map-container').innerHTML = '';
    lastForecastResults = [];
}

async function generateForecast() {
    clearStatus();
    clearResults();
    setStatus('Generating forecast...', false);
    const modelName = modelSelect.value;
    const dataName = dataSelect.value;
    const numDays = document.getElementById('num_days').value;
    if (!modelName || !dataName || modelName === "No models found" || modelName === "Error loading" || dataName === "No data files found" || dataName === "Error loading") {
        setStatus('Please select a valid model and data file.', true);
        return;
    }
    const formData = new FormData();
    formData.append('model_name', modelName);
    formData.append('data_name', dataName);
    formData.append('num_days', numDays);
    try {
        const response = await fetch(`${API_BASE_URL}/forecast`, {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || `HTTP error! status: ${response.status}`);
        }
        if (data.results && data.results.length > 0) {
            lastForecastResults = data.results;
            displayForecastResults(data.results);
            setStatus(`Forecast generated `, false);
        } else {
            setStatus(data.error || 'No results received or forecast is empty.', true);
        }
    } catch (error) {
        setStatus(`Error during forecast: ${error.message}`, true);
    }
}

function initializeMap(containerId, initialLatLng, initialZoom = 5) {
    const map = L.map(containerId).setView(initialLatLng, initialZoom);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        maxZoom: 18,
    }).addTo(map);
    return map;
}

function createRouteMap(containerId, results) {
  // If map already exists on this container, remove it properly
  if (mapInstances[containerId]) {
    mapInstances[containerId].remove();
    delete mapInstances[containerId];
  }

  const mapContainer = document.getElementById(containerId);
  // No need to clear innerHTML explicitly after .remove()
  // but can be safe to clear:
  mapContainer.innerHTML = '';

  // Initialize map with first valid coordinate or fallback center
  let initialLatLng = [0, 0];
  for (const r of results) {
    const lat = parseFloat(r.Latitude);
    const lng = parseFloat(r.Longitude);
    if (!isNaN(lat) && !isNaN(lng)) {
      initialLatLng = [lat, lng];
      break;
    }
  }

  const map = initializeMap(containerId, initialLatLng);
  mapInstances[containerId] = map;

  const latLngs = [];
  results.forEach(r => {
    const lat = parseFloat(r.Latitude);
    const lng = parseFloat(r.Longitude);
    if (isNaN(lat) || isNaN(lng)) {
      console.warn("Invalid Lat/Lng in results for route map:", r);
      return;
    }
    const latLng = [lat, lng];
    latLngs.push(latLng);

    L.marker(latLng).addTo(map)
      .bindPopup(`Lat: ${lat.toFixed(4)}, Lon: ${lng.toFixed(4)}<br>Density: ${r["Bird Density"]}`);
  });

  if (latLngs.length > 1) {
    const polyline = L.polyline(latLngs, { color: 'dodgerblue', weight: 3 }).addTo(map);
    map.fitBounds(polyline.getBounds().pad(0.1));
  } else if (latLngs.length === 1) {
    map.setView(latLngs[0], 10);
  }

  return map;
}



function createPatternsMap(containerId, results) {
    const mapContainer = document.getElementById(containerId);
    mapContainer.innerHTML = '';
    const map = initializeMap(containerId, [results[0].Latitude, results[0].Longitude]);
    const heatData = [];
    const markerLatLngs = [];

    results.forEach(r => {
        const lat = parseFloat(r.Latitude);
        const lon = parseFloat(r.Longitude);
        if (isNaN(lat) || isNaN(lon)) {
            console.warn("Invalid Lat/Lng in results for patterns map:", r);
            return;
        }
        const densityCode = r["Bird Density Code"];
        let intensity = 0.33;
        if (densityCode === 1) intensity = 0.66;
        else if (densityCode === 2) intensity = 1.0;
        heatData.push([lat, lon, intensity]);
        markerLatLngs.push([lat, lon]);
        L.marker([lat, lon]).addTo(map)
            .bindPopup(
                `Lat: ${lat.toFixed(4)}, Lon: ${lon.toFixed(4)}<br>
                Temp: ${r.Temperature}°C<br>
                Wind: ${r["Wind Speed"]} m/s ${r.Direction}<br>
                Arrival: ${r["Time of Arrival"]}<br>
                Dep. First: ${r["Departure First Flock"]}<br>
                Dep. Last: ${r["Departure Last Flock"]}<br>
                Density: ${r["Bird Density"]} (Code: ${densityCode})`
            );
    });

    if (heatData.length > 0) {
        L.heatLayer(heatData, {
            radius: 35,
            blur: 25,
            maxZoom: 14,
            gradient: { 0.3: 'blue', 0.6: 'lime', 1: 'red' }
        }).addTo(map);

        if (markerLatLngs.length > 0) {
            map.fitBounds(L.latLngBounds(markerLatLngs).pad(0.1));
        }
    }

    return map;
}

function focusMap(lat, lng, predictionIndex) {
    console.log(`Focusing on lat: ${lat}, lng: ${lng}, Prediction: ${predictionIndex}`); // Debugging line
    if (migrationMapInstance) {
        migrationMapInstance.setView([lat, lng], 10); // Zoom into the location
    }

    // Display additional info Interpreted_Data the prediction on the map
    let mapDetails = `
        <h4>Prediction ${predictionIndex}</h4>
        <p>Latitude: ${lat.toFixed(4)}<br>Longitude: ${lng.toFixed(4)}</p>
    `;
    document.getElementById('map-content-details').innerHTML = mapDetails;
}

function displayForecastResults(results) {
    // Next Arrivals Table
    let arrivalTable = `<table><thead><tr><th>Prediction</th><th>Arrival</th><th>First Departure</th><th>Last Departure</th></thead><tbody>`;
    results.forEach((r, index) => {
        const lat = parseFloat(r.Latitude);
        const lon = parseFloat(r.Longitude);
        if (!isNaN(lat) && !isNaN(lon)) {
            arrivalTable += `<tr class="clickable-row" data-lat="${lat}" data-lon="${lon}" data-prediction="${index + 1}">
                <td>Prediction ${index + 1}</td>
                <td>${r["Time of Arrival"]}</td>
                <td>${r["Departure First Flock"]}</td>
                <td>${r["Departure Last Flock"]}</td>
            </tr>`;
        }
    });
    arrivalTable += `</tbody></table>`;
    document.getElementById('arrival-content').innerHTML = arrivalTable;

    // Environmental Conditions Table
    let environmentTable = `<table><thead><tr><th>Prediction</th><th>Location</th><th>Temp (°C)</th><th>Wind (m/s)</th><th>Direction</th><th>Bird Density</th></tr></thead><tbody>`;
    results.forEach((r, index) => {
        const lat = parseFloat(r.Latitude);
        const lon = parseFloat(r.Longitude);
        if (!isNaN(lat) && !isNaN(lon)) {
            environmentTable += `<tr class="clickable-row" data-lat="${lat}" data-lon="${lon}" data-prediction="${index + 1}">
                <td>Prediction ${index + 1}</td>
                <td>${lat.toFixed(4)}, ${lon.toFixed(4)}</td>
                <td>${r.Temperature}</td>
                <td>${r["Wind Speed"]}</td>
                <td>${r.Direction}</td>
                <td>${r["Bird Density"]}</td>
            </tr>`;
        }
    });
    environmentTable += `</tbody></table>`;
    document.getElementById('environment-content').innerHTML = environmentTable;
    document.getElementById('weather-content').innerHTML = environmentTable;

    // Add event listeners to table rows
    const clickableRows = document.querySelectorAll('.clickable-row');
    clickableRows.forEach(row => {
        row.addEventListener('click', function() {
            const lat = parseFloat(this.getAttribute('data-lat'));
            const lon = parseFloat(this.getAttribute('data-lon'));
            const predictionIndex = this.getAttribute('data-prediction');
            focusMap(lat, lon, predictionIndex);
        });
    });
}

function showSection(id) {
    // Remove 'active' from all sections
    document.querySelectorAll(".section").forEach(section => section.classList.remove("active"));
    
    if (id) {
        // Activate the requested section
        document.getElementById(id).classList.add("active");
    }

    // Remove 'active' from all sidebar buttons
    document.querySelectorAll(".sidebar nav button").forEach(button => button.classList.remove("active"));

    if (id) {
        // Activate the nav button that corresponds to the section
        const btn = document.querySelector(`.sidebar nav button[onclick*="${id}"]`);
        if (btn) btn.classList.add("active");
    }

    // Hide dashboard if the selected section is NOT dashboard
    const dashboardSection = document.querySelector('.dashboard-section');
    if (dashboardSection) {
        if (id !== 'dashboard') {
            dashboardSection.style.display = 'none';
        } else {
            dashboardSection.style.display = '';
        }
    }
}

let routesMap;  // For the custom #map in the extra section
const mapInstances = {};
// === Main tab switcher ===
function showSubContent(sectionId, subContentIdToShow, btn) {
    const section = document.getElementById(sectionId);
    const subContents = section.querySelectorAll('.sub-content');
    subContents.forEach(sc => {
        sc.classList.remove('active');
        sc.style.display = 'none';
    });
    const targetSubContent = document.getElementById(`${sectionId}-${subContentIdToShow}`);
    if (targetSubContent) {
        targetSubContent.classList.add('active');
        targetSubContent.style.display = 'block';
    }

    if (btn) {
        const tabButtons = section.querySelectorAll('.tab-buttons button');
        tabButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
    } else {
        const tabButtons = section.querySelectorAll('.tab-buttons button');
        tabButtons.forEach(tbtn => {
            if (tbtn.getAttribute('onclick') && tbtn.getAttribute('onclick').includes(`'${subContentIdToShow}'`)) {
                tbtn.classList.add('active');
            } else {
                tbtn.classList.remove('active');
            }
        });
    }

    setTimeout(() => {
        if (subContentIdToShow === 'map') {
            if (!mapInstances['migration-map-container'] && lastForecastResults?.length) {
                createRouteMap('migration-map-container', lastForecastResults);
            }
            if (mapInstances['migration-map-container']) {
                mapInstances['migration-map-container'].invalidateSize();
            }
        }

        if (subContentIdToShow === 'routes') {
            if (!mapInstances['routes-map-container'] && lastForecastResults?.length) {
                createRouteMap('routes-map-container', lastForecastResults);
            }
            if (mapInstances['routes-map-container']) {
                mapInstances['routes-map-container'].invalidateSize();
            }

            // Mirror the content
            const original = document.getElementById('analytics-routes');
            const mirrorContainer = document.getElementById('mirrored-routes-content');
            if (original && mirrorContainer) {
                mirrorContainer.innerHTML = ''; // clear old content
                mirrorContainer.innerHTML = original.innerHTML;

                // Find cloned map container and rename its ID to avoid duplicate ID problem
                const clonedMapContainer = mirrorContainer.querySelector('#routes-map-container');
                if (clonedMapContainer) {
                    clonedMapContainer.id = 'mirrored-routes-map-container';
                    clonedMapContainer.style.height = '400px';
                    clonedMapContainer.style.width = '100%';
                }

                // Create map on the mirrored container ID
                if (lastForecastResults?.length) {
                    createRouteMap('mirrored-routes-map-container', lastForecastResults);
                    // Invalidate size after small delay
                    setTimeout(() => {
                        if (mapInstances['mirrored-routes-map-container']) {
                            mapInstances['mirrored-routes-map-container'].invalidateSize();
                        }
                    }, 100);
                }
            }
        }

        if (subContentIdToShow === 'patterns') {
            if (!mapInstances['patterns-map-container'] && lastForecastResults?.length) {
                createPatternsMap('patterns-map-container', lastForecastResults);
            }
            if (mapInstances['patterns-map-container']) {
                mapInstances['patterns-map-container'].invalidateSize();
            }
        }
    }, 100);
}




// === Robust initRoutesMap for custom <div id="map"> ===
function initRoutesMap() {
    const mapContainer = document.getElementById('map');

    if (!mapContainer) {
        console.warn('Map container #map not found.');
        return;
    }

    waitForMapVisibility(mapContainer, () => {
        if (!routesMap) {
            routesMap = L.map('map').setView([13.75, 121.05], 10);

            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '&copy; OpenStreetMap contributors'
            }).addTo(routesMap);

            L.marker([13.75, 121.05])
                .addTo(routesMap)
                .bindPopup('Batangas City')
                .openPopup();
        } else {
            routesMap.invalidateSize();
        }
    });
}

// === Utility: Wait until map container is visible before initializing Leaflet ===
function waitForMapVisibility(container, callback, maxTries = 20) {
    let tries = 0;

    function checkVisibility() {
        const isVisible = container.offsetWidth > 0 && container.offsetHeight > 0;

        if (isVisible) {
            callback();
        } else if (tries < maxTries) {
            tries++;
            requestAnimationFrame(checkVisibility);
        } else {
            console.error('Map container is still not visible after waiting.');
        }
    }

    checkVisibility();
}


// Training WebSocket functions
function connectTrainingSocket() {
    trainingSocket = new WebSocket(`ws://127.0.0.1:8001/ws/training-status`);

    
    trainingSocket.onopen = function(e) {
        console.log("Training WebSocket connected");
    };
    
    trainingSocket.onmessage = function(event) {
        const data = JSON.parse(event.data);
        
        if (data.active) {
            // Show the training modal if not already shown
            openModal('training-modal');
            document.getElementById('close-training-btn').style.display = 'none';
            
            // Update progress bar
            const percent = data.total > 0 ? (data.progress / data.total) * 100 : 0;
            document.getElementById('training-progress').style.width = `${percent}%`;
            
            // Update label
            document.getElementById('training-label').textContent = 
                `${data.step} (${data.progress}/${data.total})`;
            
            // Add to log if it's a new step
            if (trainingLog.length === 0 || trainingLog[trainingLog.length-1] !== data.step) {
                trainingLog.push(data.step);
                const logElement = document.getElementById('training-log');
                logElement.innerHTML += `<div>${new Date().toLocaleTimeString()}: ${data.step}</div>`;
                logElement.scrollTop = logElement.scrollHeight;
            }
        } else {
            // Training completed or not active
            if (document.getElementById('training-modal').classList.contains('active')) {
                document.getElementById('training-label').textContent = "Training completed!";
                document.getElementById('training-progress').style.width = "100%";
                document.getElementById('close-training-btn').style.display = 'block';
                
                // Add completion to log
                const logElement = document.getElementById('training-log');
                logElement.innerHTML += `<div style="color:green;font-weight:bold;">${new Date().toLocaleTimeString()}: Training process completed!</div>`;
                logElement.scrollTop = logElement.scrollHeight;
            }
        }
    };
    
    trainingSocket.onclose = function(event) {
        console.log("Training WebSocket disconnected");
    };
    
    trainingSocket.onerror = function(error) {
        console.error("Training WebSocket error:", error);
    };
}

// Tab handling
function setupTabs() {
    const tabs = document.querySelectorAll('.upload-tab');
    const tabContents = document.querySelectorAll('.upload-tab-content');
    
    // Check if URL has tab parameter
    const urlParams = new URLSearchParams(window.location.search);
    const activeTab = urlParams.get('tab') || 'uploads';
    
    // Activate the correct tab
    document.querySelector(`.upload-tab[data-tab="${activeTab}"]`).classList.add('active');
    document.getElementById(`${activeTab}-tab`).classList.add('active');
    
    // Add click handlers
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            tab.classList.add('active');
            const tabName = tab.getAttribute('data-tab');
            document.getElementById(`${tabName}-tab`).classList.add('active');
            
            // Update URL without reloading
            const url = new URL(window.location);
            url.searchParams.set('tab', tabName);
            window.history.pushState({}, '', url);
        });
    });
}

// Modal functions
function openModal(id) {
    document.getElementById(id).classList.add('active');
}

function closeModal(id) {
    document.getElementById(id).classList.remove('active');
}

function validateUpload() {
    const file = document.getElementById('xlsx-file').value;
    if (!file.endsWith('.xlsx')) {
        alert('Please select an XLSX file');
        return false;
    }
    
    // Clear training log
    trainingLog = [];
    document.getElementById('training-log').innerHTML = '';
    
    // Reset progress
    document.getElementById('training-progress').style.width = '0%';
    document.getElementById('training-label').textContent = 'Starting...';
    document.getElementById('close-training-btn').style.display = 'none';
    
    // Disable button to prevent multiple submissions
    document.getElementById('upload-btn').disabled = true;
    document.getElementById('upload-btn').innerHTML = 'Uploading...';
    
    // Show training modal
    openModal('training-modal');
    
    // Connect or reconnect the WebSocket if needed
    if (!trainingSocket || trainingSocket.readyState !== WebSocket.OPEN) {
        connectTrainingSocket();
    }
    
    return true;
}

// Load uploads
async function loadUploads() {
    try {
        const response = await fetch(`${API_BASE_URL}/uploads`);
        const uploads = await response.json();
        
        if (uploads.length === 0) {
            document.getElementById('uploads-list').innerHTML = `
                <tr>
                    <td colspan="7" class="loading">No uploads yet</td>
                </tr>
            `;
            return;
        }
        
        const html = uploads.map(upload => `
            <tr>
                <td>${upload.file_name}</td>
                <td>${formatSize(upload.file_size)}</td>
                <td>${upload.columns}</td>
                <td>${upload.rows}</td>
                <td>${formatDate(upload.uploaded_at)}</td>
                <td>${upload.version}</td>
                <td class="actions">
                    <button class="btn btn-secondary" onclick="downloadUpload(${upload.id})">Download</button>
                    <button class="btn btn-secondary" onclick="showRenameModal('upload', ${upload.id}, '${upload.file_name}')">Rename</button>
                    <button class="btn btn-danger" onclick="deleteUpload(${upload.id})">Delete</button>
                </td>
            </tr>
        `).join('');
        
        document.getElementById('uploads-list').innerHTML = html;
    } catch (error) {
        console.error('Error loading uploads:', error);
    }
}

// Load models
async function loadModels() {
    try {
        const response = await fetch(`${API_BASE_URL}/models`);
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Make sure we're dealing with an array
        const models = Array.isArray(data) ? data : [];
        
        if (models.length === 0) {
            document.getElementById('models-list').innerHTML = `
                <tr>
                    <td colspan="7" class="loading">No models yet</td>
                </tr>
            `;
            document.getElementById('validation-list').innerHTML = `
                <tr>
                    <td colspan="5" class="loading">No models yet</td>
                </tr>
            `;
            return;
        }
        
        // Models tab
        const modelsHtml = models.map(model => `
            <tr>
                <td>${model.name || 'Unnamed Model'}</td>
                <td>${formatSize(model.file_size || 0)}</td>
                <td>${model.version || 1}</td>
                <td>${formatDate(model.created_at || new Date())}</td>

                <td class="actions">
                    <button class="btn btn-secondary" onclick="downloadModel(${model.id})">Download</button>
                    <button class="btn btn-secondary" onclick="showRenameModal('model', ${model.id}, '${model.name || 'Unnamed Model'}')">Rename</button>
                    <button class="btn btn-danger" onclick="deleteModel(${model.id})">Delete</button>
                </td>
            </tr>
        `).join('');
        
        // Validation tab
        const validationHtml = models.map(model => `
            <tr>
                <td>${model.name || 'Unnamed Model'}</td>
                <td>${formatDate(model.created_at || new Date())}</td>
                <td class="actions">
                    <button class="btn btn-primary" onclick="showModelDetails(${model.id})">Info</button>
                </td>
            </tr>
        `).join('');
        
        document.getElementById('models-list').innerHTML = modelsHtml;
        document.getElementById('validation-list').innerHTML = validationHtml;
    } catch (error) {
        console.error('Error loading models:', error);
        document.getElementById('models-list').innerHTML = `
            <tr>
                <td colspan="7" class="loading">Error loading models: ${error.message}</td>
            </tr>
        `;
        document.getElementById('validation-list').innerHTML = `
            <tr>
                <td colspan="5" class="loading">Error loading models: ${error.message}</td>
            </tr>
        `;
    }
}

// Download, rename, delete actions
function downloadUpload(id) {
    window.location.href = `${API_BASE_URL}/download/upload/${id}`;
}

function downloadModel(id) {
    window.location.href = `${API_BASE_URL}/download/model/${id}`;
}

function showRenameModal(type, id, currentName) {
    document.getElementById('rename-id').value = id;
    document.getElementById('rename-type').value = type;
    document.getElementById('rename-name').value = currentName;
    openModal('rename-modal');
}

async function deleteUpload(id) {
    if (!confirm('Are you sure you want to delete this upload? All associated models will also be deleted.')) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/delete/upload/${id}`, {
            method: 'DELETE'
        });
        const result = await response.json();
        
        if (result.success) {
            loadUploads();
            loadModels();
        }
    } catch (error) {
        console.error('Error deleting upload:', error);
    }
}

async function deleteModel(id) {
    if (!confirm('Are you sure you want to delete this model?')) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/delete/model/${id}`, {
            method: 'DELETE'
        });
        const result = await response.json();
        
        if (result.success) {
            loadModels();
        }
    } catch (error) {
        console.error('Error deleting model:', error);
    }
}

// Submit rename form
function setupRenameForm() {
    const form = document.getElementById('rename-form');
    if (form) {
        form.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const id = document.getElementById('rename-id').value;
            const type = document.getElementById('rename-type').value;
            const name = document.getElementById('rename-name').value;
            
            try {
                const formData = new FormData();
                formData.append('name', name);
                
                const response = await fetch(`${API_BASE_URL}/rename/${type}/${id}`, {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();
                
                if (result.success) {
                    closeModal('rename-modal');
                    if (type === 'upload') {
                        loadUploads();
                    } else {
                        loadModels();
                    }
                }
            } catch (error) {
                console.error('Error renaming item:', error);
            }
        });
    }
}

// Show model details in modal
async function showModelDetails(id) {
    try {
        // Open modal and show loading
        openModal('model-modal');
        document.getElementById('model-modal-content').innerHTML = '<div class="loading">Loading model details...</div>';
        
        const response = await fetch(`${API_BASE_URL}/model/${id}`);
        const model = await response.json();
        
        document.getElementById('model-modal-title').textContent = `Model: ${model.name}`;
        
        // Create metrics cards
        let metricsHtml = '<div class="metrics-grid">';
        for (const [feature, metrics] of Object.entries(model.metrics)) {
            metricsHtml += `
                <div class="metric-card">
                    <div class="metric-name">${feature}</div>
                    <div class="metric-value">MSE: ${metrics.MSE.toFixed(4)}</div>
                    <div class="metric-value">RMSE: ${metrics.RMSE.toFixed(4)}</div>
                    <div class="metric-value">MAE: ${metrics.MAE.toFixed(4)}</div>
                </div>
            `;
        }
        metricsHtml += '</div>';
        
        // Check if we have last_results for this model (from current session)
        let plotsHtml = '<div class="plots">';
        plotsHtml += '<p>Plots are available for newly trained models only.</p>';
        
        // If this is the last trained model and we have results
        if (window.lastResults && window.lastResults.model_id === id) {
            plotsHtml = `
                <div class="plots">
                    <h3>Correlation Matrix</h3>
                    <img src="data:image/png;base64,${window.lastResults.corr_plot}">
                    
                    <h3>Actual vs Predicted</h3>
                    <img src="data:image/png;base64,${window.lastResults.avp_plot}">
                    
                    <h3>Training Loss</h3>
                    <img src="data:image/png;base64,${window.lastResults.loss_plot}">
                </div>
            `;
        }
        
        const html = `
            <h3>Model Information</h3>
            <table>
                <tr>
                    <th>Model Name</th>
                    <td>${model.name}</td>
                </tr>
                <tr>
                    <th>Created</th>
                    <td>${formatDate(model.created_at)}</td>
                </tr>
                <tr>
                    <th>Size</th>
                    <td>${formatSize(model.file_size)}</td>
                </tr>
                <tr>
                    <th>Version</th>
                    <td>${model.version}</td>
                </tr>
                <tr>
                    <th>Input Dataset</th>
                    <td>${model.upload_name}</td>
                </tr>
            </table>
            
            <h3>Performance Metrics</h3>
            ${metricsHtml}
            
            <h3>Visualization</h3>
            ${plotsHtml}
        `;
        
        document.getElementById('model-modal-content').innerHTML = html;
    } catch (error) {
        console.error('Error loading model details:', error);
        document.getElementById('model-modal-content').innerHTML = '<div class="loading">Error loading model details</div>';
    }
}

// Helper functions
function formatSize(bytes) {
    if (bytes < 1024) return bytes + ' bytes';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
}

function formatDate(dateString) {
    try {
        const date = new Date(dateString);
        if (isNaN(date.getTime())) {
            return 'Invalid Date';
        }
        return date.toLocaleString();
    } catch (error) {
        return 'Invalid Date';
    }
}

// Check for training results
async function checkForResults() {
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('result') === '1') {
        try {
            const response = await fetch(`${API_BASE_URL}/last_results`);
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            
            const results = await response.json();
            
            if (results.error) {
                document.getElementById('results-container').innerHTML = `
                    <div class="upload-card">
                        <div class="alert alert-danger">${results.error}</div>
                    </div>
                `;
                return;
            }
            
            // Save last results for model details modal
            window.lastResults = results;
            
            // Ensure metrics is an array
            const metrics = Array.isArray(results.metrics) ? results.metrics : [];
            
            // Create metrics table with defensive coding
            const metricsTable = `
                <table>
                    <thead>
                        <tr>
                            <th>Feature</th>
                            <th>MAE</th>
                            <th>RMSE</th>
                            <th>MSE</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${metrics.map(m => `
                            <tr>
                                <td>${m.feature || 'Unknown'}</td>
                                <td>${(m.MAE || 0).toFixed(4)}</td>
                                <td>${(m.RMSE || 0).toFixed(4)}</td>
                                <td>${(m.MSE || 0).toFixed(4)}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            `;
            
            const html = `
                <div class="upload-card">
                    <h2>Latest Training Results</h2>
                    <div class="badge badge-success">New Model Created</div>
                    
                    <h3>Test Metrics</h3>
                    ${metricsTable}
                    
                    <div class="plots">
                        ${results.corr_plot ? `
                            <h3>Correlation Matrix</h3>
                            <img src="data:image/png;base64,${results.corr_plot}">
                        ` : ''}
                        
                        ${results.avp_plot ? `
                            <h3>Actual vs Predicted</h3>
                            <img src="data:image/png;base64,${results.avp_plot}">
                        ` : ''}
                        
                        ${results.loss_plot ? `
                            <h3>Training Loss</h3>
                            <img src="data:image/png;base64,${results.loss_plot}">
                        ` : ''}
                    </div>
                </div>
            `;
            
            document.getElementById('results-container').innerHTML = html;
        } catch (error) {
            console.error('Error loading results:', error);
            document.getElementById('results-container').innerHTML = `
                <div class="card">
                    <div class="alert alert-danger">Error loading results: ${error.message}</div>
                </div>
            `;
        }
    }
}

// Initialize all functionality
window.onload = () => {
    // Initialize forecast functionality
    loadAssets();
    
    // Setup tabs and other UI elements
    if (document.querySelector('.tab-buttons button.active')) {
        showSubContent('analytics', 'arrival', document.querySelector('.tab-buttons button.active'));
    }
    
    // Initialize training system
    setupTabs();
    loadUploads();
    loadModels();
    checkForResults();
    connectTrainingSocket();
    setupRenameForm();
    
    // Add necessary event listeners
    document.addEventListener('click', function(event) {
        if (event.target.matches('.close-modal') || event.target.matches('.modal-bg')) {
            const modalId = event.target.closest('.modal').id;
            closeModal(modalId);
        }
    });
};


// Tab handling
function setupTabs() {
    const tabs = document.querySelectorAll('.upload-tab');
    const tabContents = document.querySelectorAll('.upload-tab-content');
    
    // Check if URL has tab parameter
    const urlParams = new URLSearchParams(window.location.search);
    const activeTab = urlParams.get('tab') || 'uploads';
    
    // Activate the correct tab
    document.querySelector(`.upload-tab[data-tab="${activeTab}"]`).classList.add('active');
    document.getElementById(`${activeTab}-tab`).classList.add('active');
    
    // Add click handlers
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            tab.classList.add('active');
            const tabName = tab.getAttribute('data-tab');
            document.getElementById(`${tabName}-tab`).classList.add('active');
            
            // Update URL without reloading
            const url = new URL(window.location);
            url.searchParams.set('tab', tabName);
            window.history.pushState({}, '', url);
        });
    });
}

// Load uploads
async function loadUploads() {
    try {
        const response = await fetch(`${API_BASE_URL_2}/uploads`);
        const uploads = await response.json();
        
        if (uploads.length === 0) {
            document.getElementById('uploads-list').innerHTML = `
                <tr>
                    <td colspan="7" class="loading">No uploads yet</td>
                </tr>
            `;
            return;
        }
        
        const html = uploads.map(upload => `
            <tr>
                <td>${upload.file_name}</td>
                <td>${formatSize(upload.file_size)}</td>
                <td>${upload.columns}</td>
                <td>${upload.rows}</td>
                <td>${formatDate(upload.uploaded_at)}</td>
                <td>${upload.version}</td>
                <td class="actions">
                    <button class="btn btn-secondary" onclick="downloadUpload(${upload.id})">Download</button>
                    <button class="btn btn-secondary" onclick="showRenameModal('upload', ${upload.id}, '${upload.file_name}')">Rename</button>
                    <button class="btn btn-danger" onclick="deleteUpload(${upload.id})">Delete</button>
                </td>
            </tr>
        `).join('');
        
        document.getElementById('uploads-list').innerHTML = html;
    } catch (error) {
        console.error('Error loading uploads:', error);
    }
}

async function loadModels() {
    try {
        const response = await fetch(`${API_BASE_URL_2}/models`);
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Make sure we're dealing with an array
        const models = Array.isArray(data) ? data : [];
        
        if (models.length === 0) {
            document.getElementById('models-list').innerHTML = `
                <tr>
                    <td colspan="7" class="loading">No models yet</td>
                </tr>
            `;
            document.getElementById('validation-list').innerHTML = `
                <tr>
                    <td colspan="5" class="loading">No models yet</td>
                </tr>
            `;
            return;
        }
        
        // Models tab
        const modelsHtml = models.map(model => `
            <tr>
                <td>${model.name || 'Unnamed Model'}</td>
                <td>${formatSize(model.file_size || 0)}</td>
                <td>${model.version || 1}</td>
                <td>${formatDate(model.created_at || new Date())}</td>

                <td class="actions">
                    <button class="btn btn-secondary" onclick="downloadModel(${model.id})">Download</button>
                    <button class="btn btn-secondary" onclick="showRenameModal('model', ${model.id}, '${model.name || 'Unnamed Model'}')">Rename</button>
                    <button class="btn btn-danger" onclick="deleteModel(${model.id})">Delete</button>
                </td>
            </tr>
        `).join('');
        
        // Validation tab
        const validationHtml = models.map(model => `
            <tr>
                <td>${model.name || 'Unnamed Model'}</td>
                <td>${formatDate(model.created_at || new Date())}</td>

                <td class="actions">
                    <button class="btn btn-primary" onclick="showModelDetails(${model.id})">Info</button>
                </td>
            </tr>
        `).join('');
        
        document.getElementById('models-list').innerHTML = modelsHtml;
        document.getElementById('validation-list').innerHTML = validationHtml;
    } catch (error) {
        console.error('Error loading models:', error);
        document.getElementById('models-list').innerHTML = `
            <tr>
                <td colspan="7" class="loading">Error loading models: ${error.message}</td>
            </tr>
        `;
        document.getElementById('validation-list').innerHTML = `
            <tr>
                <td colspan="5" class="loading">Error loading models: ${error.message}</td>
            </tr>
        `;
    }
}

// Helper functions
function formatSize(bytes) {
    if (bytes < 1024) return bytes + ' bytes';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
}

function formatDate(dateString) {
    try {
        const date = new Date(dateString);
        if (isNaN(date.getTime())) {
            return 'Invalid Date';
        }
        return date.toLocaleString();
    } catch (error) {
        return 'Invalid Date';
    }
}

async function showModelDetails(id) {
    try {
        // Open modal and show loading
        openModal('model-modal');
        document.getElementById('model-modal-content').innerHTML = '<div class="loading">Loading model details...</div>';
        
        const response = await fetch(`${API_BASE_URL_2}/model/${id}`);
        const model = await response.json();
        
        document.getElementById('model-modal-title').textContent = `Model: ${model.name}`;
        
        // Create metrics cards
        let metricsHtml = '<div class="metrics-grid">';
        for (const [feature, metrics] of Object.entries(model.metrics)) {
            metricsHtml += `
                <div class="metric-card">
                    <div class="metric-name">${feature}</div>
                    <div class="metric-value">MSE: ${metrics.MSE.toFixed(4)}</div>
                    <div class="metric-value">RMSE: ${metrics.RMSE.toFixed(4)}</div>
                    <div class="metric-value">MAE: ${metrics.MAE.toFixed(4)}</div>
                </div>
            `;
        }
        metricsHtml += '</div>';
        
        // Check if we have last_results for this model (from current session)
        let plotsHtml = '<div class="plots">';
        plotsHtml += '<p>Plots are available for newly trained models only.</p>';
        
        // If this is the last trained model and we have results
        if (window.lastResults && window.lastResults.model_id === id) {
            plotsHtml = `
                <div class="plots">
                    <h3>Correlation Matrix</h3>
                    <img src="data:image/png;base64,${window.lastResults.corr_plot}">
                    
                    <h3>Actual vs Predicted</h3>
                    <img src="data:image/png;base64,${window.lastResults.avp_plot}">
                    
                    <h3>Training Loss</h3>
                    <img src="data:image/png;base64,${window.lastResults.loss_plot}">
                </div>
            `;
        }
        
        const html = `
            <h3>Model Information</h3>
            <table>
                <tr>
                    <th>Model Name</th>
                    <td>${model.name}</td>
                </tr>
                <tr>
                    <th>Created</th>
                    <td>${formatDate(model.created_at)}</td>
                </tr>
                <tr>
                    <th>Size</th>
                    <td>${formatSize(model.file_size)}</td>
                </tr>
                <tr>
                    <th>Version</th>
                    <td>${model.version}</td>
                </tr>
                <tr>
                    <th>Input Dataset</th>
                    <td>${model.upload_name}</td>
                </tr>
            </table>
            
            <h3>Performance Metrics</h3>
            ${metricsHtml}
            
        `;
        
        document.getElementById('model-modal-content').innerHTML = html;
    } catch (error) {
        console.error('Error loading model details:', error);
        document.getElementById('model-modal-content').innerHTML = '<div class="loading">Error loading model details</div>';
    }
}

// Modal functions
function openModal(id) {
    document.getElementById(id).classList.add('active');
}

function closeModal(id) {
    document.getElementById(id).classList.remove('active');
}

async function deleteUpload(id) {
    if (!confirm('Are you sure you want to delete this upload? All associated models will also be deleted.')) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL_2}/delete/upload/${id}`, {
            method: 'DELETE'
        });
        const result = await response.json();
        
        if (result.success) {
            loadUploads();
            loadModels();
        }
    } catch (error) {
        console.error('Error deleting upload:', error);
    }
}

async function deleteModel(id) {
    if (!confirm('Are you sure you want to delete this model?')) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL_2}/delete/model/${id}`, {
            method: 'DELETE'
        });
        const result = await response.json();
        
        if (result.success) {
            loadModels();
        }
    } catch (error) {
        console.error('Error deleting model:', error);
    }
}


function connectTrainingSocket() {
    trainingSocket = new WebSocket(`${API_BASE_URL_2.replace('http', 'ws')}/ws/training-status`);

    trainingSocket.onopen = () => console.log("Training WebSocket connected");

    trainingSocket.onmessage = event => {
        const data = JSON.parse(event.data);

        if (data.active) {
            openModal('training-modal');
            document.getElementById('close-training-btn').style.display = 'none';

            const percent = data.total > 0 ? (data.progress / data.total) * 100 : 0;
            document.getElementById('training-progress').style.width = `${percent}%`;
            document.getElementById('training-label').textContent = `${data.step} (${data.progress}/${data.total})`;

            if (trainingLog.length === 0 || trainingLog[trainingLog.length - 1] !== data.step) {
                trainingLog.push(data.step);
                const logElement = document.getElementById('training-log');
                logElement.innerHTML += `<div>${new Date().toLocaleTimeString()}: ${data.step}</div>`;
                logElement.scrollTop = logElement.scrollHeight;
            }
        } else {
            if (document.getElementById('training-modal').classList.contains('active')) {
                document.getElementById('training-label').textContent = "Training completed!";
                document.getElementById('training-progress').style.width = "100%";
                document.getElementById('close-training-btn').style.display = 'block';

                const logElement = document.getElementById('training-log');
                logElement.innerHTML += `<div style="color:green;font-weight:bold;">${new Date().toLocaleTimeString()}: Training process completed!</div>`;
                logElement.scrollTop = logElement.scrollHeight;

                loadTrainingResults();
            }
        }
    };

    trainingSocket.onclose = () => console.log("Training WebSocket disconnected");
    trainingSocket.onerror = error => console.error("Training WebSocket error:", error);
}


// ================================
// UI Enhancements
// ================================

function showLoader() {
    const loader = document.getElementById('loader');
    if (loader) loader.style.display = 'block';
}

function hideLoader() {
    const loader = document.getElementById('loader');
    if (loader) loader.style.display = 'none';
}

function showToast(message, isError = false) {
    const toast = document.createElement('div');
    toast.className = `toast ${isError ? 'toast-error' : 'toast-success'}`;
    toast.innerText = message;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
}

// ================================
// Rename Handlers
// ================================

async function renameItem(type, id, newName) {
    try {
        const formData = new FormData();
        formData.append('name', newName);

        const response = await fetch(`${API_BASE_URL_2}/rename/${type}/${id}`, {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            showToast('Rename successful');
            if (type === 'upload') loadUploads();
            else loadModels();
        } else {
            showToast('Rename failed', true);
        }
    } catch (error) {
        showToast('Error renaming item', true);
        console.error(error);
    }
}

function showRenameModal(type, id, currentName) {
    const modal = document.getElementById('rename-modal');
    const renameInput = document.getElementById('rename-name');
    const renameType = document.getElementById('rename-type');
    const renameId = document.getElementById('rename-id');
    renameInput.value = currentName;
    renameType.value = type;
    renameId.value = id;
    openModal('rename-modal');
}

// ================================
// Initialization
// ================================
document.addEventListener('DOMContentLoaded', () => {
    setupTabs();
    connectTrainingSocket();
    loadUploads();
    loadModels();

    const renameForm = document.getElementById('rename-form');
    renameForm.addEventListener('submit', function (e) {
        e.preventDefault();
        const id = document.getElementById('rename-id').value;
        const type = document.getElementById('rename-type').value;
        const newName = document.getElementById('rename-name').value;
        renameItem(type, id, newName);
        closeModal('rename-modal');
    });
});

src="https://unpkg.com/leaflet@1.9.3/dist/leaflet.js"
  crossorigin=""
  const topImages = [
    {
      path: "Images/L_B/C.Tirona & P.Zamora.png",
      title: "C.Tirona & P.Zamora",
      backContent: `
        <div class="flip-card-back styled-back">
          <div class="card-title">C.Tirona & P.Zamora</div>
          <div class="arrow-row">
            <div class="arrow-col">
              <div class="arrow-symbol">&#8593;</div>
              <div class="arrow-date">18<br><small>Jan</small></div>
            </div>
            <div class="arrow-col">
              <div class="arrow-symbol">&#8595;</div>
              <div class="arrow-date">24<br><small>April</small></div>
            </div>
          </div>
        </div>`
    },
    {
      path: "Images/L_B/C.Tirona.png",
      title: "C.Tirona",
      backContent: `
        <div class="flip-card-back styled-back">
          <div class="card-title">C.Tirona</div>
          <div class="arrow-row">
            <div class="arrow-col">
              <div class="arrow-symbol">&#8593;</div>
              <div class="arrow-date">18<br><small>Jan</small></div>
            </div>
            <div class="arrow-col">
              <div class="arrow-symbol">&#8595;</div>
              <div class="arrow-date">24<br><small>April</small></div>
            </div>
          </div>
        </div>`
    },
    {
      path: "Images/L_B/D.Silang and C. Tirona Intersection.png",
      title: "D.Silang & C.Tirona Intersection",
      backContent: `
        <div class="flip-card-back styled-back">
          <div class="card-title">D.Silang & C.Tirona Intersection</div>
          <div class="arrow-row">
            <div class="arrow-col">
              <div class="arrow-symbol">&#8593;</div>
              <div class="arrow-date">18<br><small>Jan</small></div>
            </div>
            <div class="arrow-col">
              <div class="arrow-symbol">&#8595;</div>
              <div class="arrow-date">24<br><small>April</small></div>
            </div>
          </div>
        </div>`
    },
    {
      path: "Images/L_B/D.Silang.png",
      title: "D.Silang",
      backContent: `
        <div class="flip-card-back styled-back">
          <div class="card-title">D.Silang</div>
          <div class="arrow-row">
            <div class="arrow-col">
              <div class="arrow-symbol">&#8593;</div>
              <div class="arrow-date">18<br><small>Jan</small></div>
            </div>
            <div class="arrow-col">
              <div class="arrow-symbol">&#8595;</div>
              <div class="arrow-date">24<br><small>April</small></div>
            </div>
          </div>
        </div>`
    },
    {
      path: "Images/L_B/Evangelista.png",
      title: "Evangelista",
      backContent: `
        <div class="flip-card-back styled-back">
          <div class="card-title">Evangelista</div>
          <div class="arrow-row">
            <div class="arrow-col">
              <div class="arrow-symbol">&#8593;</div>
              <div class="arrow-date">18<br><small>Jan</small></div>
            </div>
            <div class="arrow-col">
              <div class="arrow-symbol">&#8595;</div>
              <div class="arrow-date">24<br><small>April</small></div>
            </div>
          </div>
        </div>`
    },
    {
      path: "Images/L_B/P. Burgos and C. Tirona Intersection.png",
      title: "P. Burgos & C. Tirona Intersection",
      backContent: `
        <div class="flip-card-back styled-back">
          <div class="card-title">P. Burgos & C. Tirona Intersection</div>
          <div class="arrow-row">
            <div class="arrow-col">
              <div class="arrow-symbol">&#8593;</div>
              <div class="arrow-date">18<br><small>Jan</small></div>
            </div>
            <div class="arrow-col">
              <div class="arrow-symbol">&#8595;</div>
              <div class="arrow-date">24<br><small>April</small></div>
            </div>
          </div>
        </div>`
    },
    {
      path: "Images/L_B/P.Burgos & D.Silang.png",
      title: "P.Burgos & D.Silang",
      backContent: `
        <div class="flip-card-back styled-back">
          <div class="card-title">P.Burgos & D.Silang</div>
          <div class="arrow-row">
            <div class="arrow-col">
              <div class="arrow-symbol">&#8593;</div>
              <div class="arrow-date">18<br><small>Jan</small></div>
            </div>
            <div class="arrow-col">
              <div class="arrow-symbol">&#8595;</div>
              <div class="arrow-date">24<br><small>April</small></div>
            </div>
          </div>
        </div>`
    },
    {
      path: "Images/L_B/P.Burgos 711 to PSBank.png",
      title: "P.Burgos 711 to PSBank",
      backContent: `
        <div class="flip-card-back styled-back">
          <div class="card-title">P.Burgos 711 to PSBank</div>
          <div class="arrow-row">
            <div class="arrow-col">
              <div class="arrow-symbol">&#8593;</div>
              <div class="arrow-date">18<br><small>Jan</small></div>
            </div>
            <div class="arrow-col">
              <div class="arrow-symbol">&#8595;</div>
              <div class="arrow-date">24<br><small>April</small></div>
            </div>
          </div>
        </div>`
    },
    {
      path: "Images/L_B/P.Burgos and Evangelista Intersection.png",
      title: "P.Burgos & Evangelista Intersection",
      backContent: `
        <div class="flip-card-back styled-back">
          <div class="card-title">P.Burgos & Evangelista Intersection</div>
          <div class="arrow-row">
            <div class="arrow-col">
              <div class="arrow-symbol">&#8593;</div>
              <div class="arrow-date">18<br><small>Jan</small></div>
            </div>
            <div class="arrow-col">
              <div class="arrow-symbol">&#8595;</div>
              <div class="arrow-date">24<br><small>April</small></div>
            </div>
          </div>
        </div>`
    },
    {
      path: "Images/L_B/P.Burgos Max to PNB.png",
      title: "P.Burgos Max to PNB",
      backContent: `
        <div class="flip-card-back styled-back">
          <div class="card-title">P.Burgos Max to PNB</div>
          <div class="arrow-row">
            <div class="arrow-col">
              <div class="arrow-symbol">&#8593;</div>
              <div class="arrow-date">18<br><small>Jan</small></div>
            </div>
            <div class="arrow-col">
              <div class="arrow-symbol">&#8595;</div>
              <div class="arrow-date">24<br><small>April</small></div>
            </div>
          </div>
        </div>`
    },
    {
      path: "Images/L_B/P.burgos to AICS.png",
      title: "P.burgos to AICS",
      backContent: `
        <div class="flip-card-back styled-back">
          <div class="card-title">P.burgos to AICS</div>
          <div class="arrow-row">
            <div class="arrow-col">
              <div class="arrow-symbol">&#8593;</div>
              <div class="arrow-date">18<br><small>Jan</small></div>
            </div>
            <div class="arrow-col">
              <div class="arrow-symbol">&#8595;</div>
              <div class="arrow-date">24<br><small>April</small></div>
            </div>
          </div>
        </div>`
    },
    {
      path: "Images/L_B/P.Burgos To andoks.png",
      title: "P.Burgos To Andoks",
      backContent: `
        <div class="flip-card-back styled-back">
          <div class="card-title">P.Burgos To Andoks</div>
          <div class="arrow-row">
            <div class="arrow-col">
              <div class="arrow-symbol">&#8593;</div>
              <div class="arrow-date">18<br><small>Jan</small></div>
            </div>
            <div class="arrow-col">
              <div class="arrow-symbol">&#8595;</div>
              <div class="arrow-date">24<br><small>April</small></div>
            </div>
          </div>
        </div>`
    },
    {
      path: "Images/L_B/P.Zamora & Evangelista Interaction.png",
      title: "P.Zamora & Evangelista Interaction",
      backContent: `
        <div class="flip-card-back styled-back">
          <div class="card-title">P.Zamora & Evangelista Interaction</div>
          <div class="arrow-row">
            <div class="arrow-col">
              <div class="arrow-symbol">&#8593;</div>
              <div class="arrow-date">18<br><small>Jan</small></div>
            </div>
            <div class="arrow-col">
              <div class="arrow-symbol">&#8595;</div>
              <div class="arrow-date">24<br><small>April</small></div>
            </div>
          </div>
        </div>`
    },
    {
      path: "Images/L_B/P.Zamora.png",
      title: "P.Zamora",
      backContent: `
        <div class="flip-card-back styled-back">
          <div class="card-title">P.Zamora</div>
          <div class="arrow-row">
            <div class="arrow-col">
              <div class="arrow-symbol">&#8595;</div>
              <div class="arrow-date">{DOWN_DATE}<br><small>{DOWN_MONTH}</small></div>
            </div>
            <div class="arrow-col">
              <div class="arrow-symbol">&#8593;</div>
              <div class="arrow-date">{UP_DATE}<br><small>{UP_MONTH}</small></div>
            </div>
          </div>
        </div>`
    }
  ];

  const bottomImages = [
    {
      path: "Images/L_N/C. Tirona and P. Zamora Intersection.png",
      title: "C. Tirona & P. Zamora Intersection",
      backContent: `
        <div class="flip-card-back bottom-back">
          <div class="card-title">C. Tirona & P. Zamora Intersection</div>
        </div>`
    },
    {
      path: "Images/L_N/C.Tirona_Lights and Sound.png",
      title: "C.Tirona Lights and Sound",
      backContent: `
        <div class="flip-card-back bottom-back">
          <div class="card-title">C.Tirona Lights and Sound</div>
        </div>`
    },
    {
      path: "Images/L_N/D.Silang and C.Tirona.png",
      title: "D.Silang & C.Tirona",
      backContent: `
        <div class="flip-card-back bottom-back">
          <div class="card-title">D.Silang & C.Tirona</div>
        </div>`
    },
    {
      path: "Images/L_N/D.Silang.png",
      title: "D.Silang",
      backContent: `
        <div class="flip-card-back bottom-back">
          <div class="card-title">D.Silang</div>
        </div>`
    },
    {
      path: "Images/L_N/Evangelista.png",
      title: "Evangelista",
      backContent: `
        <div class="flip-card-back bottom-back">
          <div class="card-title">Evangelista</div>
        </div>`
    },
    {
      path: "Images/L_N/P. Burgos 7_11 to PS Bank.png",
      title: "P. Burgos 7_11 to PS Bank",
      backContent: `
        <div class="flip-card-back bottom-back">
          <div class="card-title">P. Burgos 7_11 to PS Bank</div>
        </div>`
    },
    {
      path: "Images/L_N/P. Burgos and C. Tirona Intersection.png",
      title: "P. Burgos & C. Tirona Intersection",
      backContent: `
        <div class="flip-card-back bottom-back">
          <div class="card-title">P. Burgos & C. Tirona Intersection</div>
        </div>`
    },
    {
      path: "Images/L_N/P. Burgos and D. Silang Intersection.png",
      title: "P. Burgos & D. Silang Intersection",
      backContent: `
        <div class="flip-card-back bottom-back">
          <div class="card-title">P. Burgos & D. Silang Intersection</div>
        </div>`
    },
    {
      path: "Images/L_N/P. Burgos and Evangelista Intersection.png",
      title: "P. Burgos & Evangelista Intersection",
      backContent: `
        <div class="flip-card-back bottom-back">
          <div class="card-title">P. Burgos & Evangelista Intersection</div>
        </div>`
    },
  ];


topImages.forEach(({ path, title, backContent }) => {
  const card = createPopupCardUnique(path, title, backContent);
  document.getElementById("top-scroll").appendChild(card);
});

bottomImages.forEach(({ path, title, backContent }) => {
  const card = createPopupCardUnique(path, title, backContent);
  document.getElementById("bottom-scroll").appendChild(card);
});



/// Scroll functions
function scrollRightNegative(containerId) {
  const container = document.getElementById(containerId);
  container.scrollBy({ left: -200, behavior: 'smooth' });
}

function scrollRight(containerId) {
  const container = document.getElementById(containerId);
  container.scrollBy({ left: 200, behavior: 'smooth' });
}

// Initialize Leaflet map
const map = L.map("map").setView([13.75, 121.02], 12);

L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 18,
  attribution: "© OpenStreetMap contributors",
}).addTo(map);

// Function to create cards that open the custom popup modal
function createPopupCardUnique(imagePath, title, backContent) {
  const card = document.createElement("div");
  card.classList.add("flip-card");

  card.innerHTML = `
    <img src="${imagePath}" alt="${title}">
    <div class="hover-overlay">${title}</div>
  `;

  card.addEventListener("click", () => {
    const modalOverlay = document.getElementById("custom-popup-overlay");
    const modalContent = document.getElementById("custom-popup-content");

    // Clear previous content and add close button
    modalContent.innerHTML = `<span class="custom-popup-close" id="custom-popup-close">&times;</span>`;

    // Create and append enlarged image
    const enlargedImg = document.createElement("img");
    enlargedImg.src = imagePath;
    enlargedImg.alt = title;
    enlargedImg.style.width = "100%";
    enlargedImg.style.borderRadius = "12px";

    // Create and append styled back content container
    const backDiv = document.createElement("div");
    backDiv.classList.add("custom-popup-back-content");
    backDiv.innerHTML = backContent;

    modalContent.appendChild(enlargedImg);
    modalContent.appendChild(backDiv);

    // Show the modal with animation
    modalOverlay.classList.add("show");

    // Close modal event listeners
    const closeBtn = document.getElementById("custom-popup-close");
    closeBtn.onclick = () => {
      modalOverlay.classList.remove("show");
    };

    modalOverlay.onclick = (event) => {
      if (event.target === modalOverlay) {
        modalOverlay.classList.remove("show");
      }
    };
  });

  return card;
}

// Populate top and bottom scroll containers with cards
topImages.forEach(({ path, title, backContent }) => {
  const card = createPopupCardUnique(path, title, backContent);
  document.getElementById("top-scroll").appendChild(card);
});

bottomImages.forEach(({ path, title, backContent }) => {
  const card = createPopupCardUnique(path, title, backContent);
  document.getElementById("bottom-scroll").appendChild(card);
});
