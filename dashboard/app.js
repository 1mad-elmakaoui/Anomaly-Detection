// Dashboard Application for Information-Theoretic Anomaly Detection
// This is a frontend demo - in production, connect to Python backend API

// Sample data generator
function generateSampleData(n = 500) {
    const data = [];
    const timestamps = [];
    const baseDate = new Date('2023-01-01');
    
    for (let i = 0; i < n; i++) {
        // Generate realistic price movement with trends and volatility
        const trend = 100 + i * 0.1;
        const noise = (Math.random() - 0.5) * 5;
        const seasonal = 3 * Math.sin(i * 0.05);
        
        let value = trend + noise + seasonal;
        
        // Inject some anomalies
        if (Math.random() < 0.05) {
            value += (Math.random() - 0.5) * 30; // Spike anomalies
        }
        
        data.push(value);
        
        const date = new Date(baseDate);
        date.setDate(date.getDate() + i);
        timestamps.push(date.toISOString().split('T')[0]);
    }
    
    return { data, timestamps };
}

// Global state
let currentData = null;
let currentTimestamps = null;
let detectionResults = null;

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    updateThresholdDisplay();
});

function initializeEventListeners() {
    // Data source selection
    document.getElementById('data-source').addEventListener('change', handleDataSourceChange);
    
    // Threshold slider
    document.getElementById('threshold').addEventListener('input', updateThresholdDisplay);
    
    // Analyze button
    document.getElementById('analyze-btn').addEventListener('click', runAnalysis);
    
    // Export buttons
    document.getElementById('export-json').addEventListener('click', () => exportResults('json'));
    document.getElementById('export-csv').addEventListener('click', () => exportResults('csv'));
    document.getElementById('export-report').addEventListener('click', () => exportResults('report'));
}

function handleDataSourceChange(e) {
    const source = e.target.value;
    
    document.getElementById('ticker-group').style.display = source === 'yahoo' ? 'block' : 'none';
    document.getElementById('upload-group').style.display = source === 'upload' ? 'block' : 'none';
    
    if (source === 'sample') {
        const { data, timestamps } = generateSampleData();
        currentData = data;
        currentTimestamps = timestamps;
    }
}

function updateThresholdDisplay() {
    const threshold = document.getElementById('threshold').value;
    document.getElementById('threshold-value').textContent = threshold;
}

// Analysis Functions
async function runAnalysis() {
    showLoading(true);
    
    try {
        // Simulate processing delay
        await new Promise(resolve => setTimeout(resolve, 1500));
        
        // Ensure we have data
        if (!currentData) {
            const { data, timestamps } = generateSampleData();
            currentData = data;
            currentTimestamps = timestamps;
        }
        
        // Get parameters
        const method = document.getElementById('detector-method').value;
        const windowSize = parseInt(document.getElementById('window-size').value);
        const threshold = parseFloat(document.getElementById('threshold').value);
        
        // Run detection (simulated)
        detectionResults = simulateDetection(currentData, method, windowSize, threshold);
        
        // Update all visualizations
        updateVisualizations();
        updateStatistics();
        updateAnomalyList();
        updateMetrics();
        
        // Enable export buttons
        document.getElementById('export-json').disabled = false;
        document.getElementById('export-csv').disabled = false;
        document.getElementById('export-report').disabled = false;
        
    } catch (error) {
        console.error('Analysis error:', error);
        alert('Error during analysis. Please check console for details.');
    } finally {
        showLoading(false);
    }
}

function simulateDetection(data, method, windowSize, threshold) {
    // Simulate anomaly detection results
    const anomalies = [];
    const scores = [];
    const entropies = [];
    
    // Calculate rolling statistics
    for (let i = windowSize; i < data.length; i++) {
        const window = data.slice(i - windowSize, i);
        
        // Calculate simple entropy (simulation)
        const entropy = calculateSimpleEntropy(window);
        entropies.push(entropy);
        
        // Detect anomalies based on threshold
        const mean = window.reduce((a, b) => a + b, 0) / window.length;
        const std = Math.sqrt(window.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / window.length);
        const zscore = Math.abs((data[i] - mean) / std);
        
        const score = Math.min(zscore / 5, 1); // Normalize to [0, 1]
        scores.push(score);
        
        if (score > threshold) {
            const severity = score > 0.9 ? 'severe' : score > 0.7 ? 'moderate' : 'mild';
            anomalies.push({
                index: i,
                timestamp: currentTimestamps[i],
                value: data[i],
                score: score,
                severity: severity
            });
        }
    }
    
    return {
        anomalies,
        allScores: scores,
        entropies,
        method
    };
}

function calculateSimpleEntropy(data) {
    // Simple histogram-based entropy calculation
    const bins = 10;
    const min = Math.min(...data);
    const max = Math.max(...data);
    const binWidth = (max - min) / bins;
    
    const histogram = new Array(bins).fill(0);
    
    data.forEach(val => {
        const binIndex = Math.min(Math.floor((val - min) / binWidth), bins - 1);
        histogram[binIndex]++;
    });
    
    // Calculate entropy
    let entropy = 0;
    const total = data.length;
    
    histogram.forEach(count => {
        if (count > 0) {
            const p = count / total;
            entropy -= p * Math.log2(p);
        }
    });
    
    return entropy;
}

// Visualization Functions
function updateVisualizations() {
    plotTimeSeries();
    plotEntropyEvolution();
}

function plotTimeSeries() {
    const container = document.getElementById('timeseries-chart');
    
    // Normal data trace
    const normalTrace = {
        x: currentTimestamps,
        y: currentData,
        mode: 'lines',
        name: 'Time Series',
        line: {
            color: '#6366f1',
            width: 2
        }
    };
    
    // Anomaly points trace
    const anomalyIndices = detectionResults.anomalies.map(a => a.index);
    const anomalyValues = detectionResults.anomalies.map(a => a.value);
    const anomalyTimestamps = detectionResults.anomalies.map(a => a.timestamp);
    const anomalyColors = detectionResults.anomalies.map(a => {
        if (a.severity === 'severe') return '#ef4444';
        if (a.severity === 'moderate') return '#f59e0b';
        return '#10b981';
    });
    
    const anomalyTrace = {
        x: anomalyTimestamps,
        y: anomalyValues,
        mode: 'markers',
        name: 'Anomalies',
        marker: {
            size: 12,
            color: anomalyColors,
            line: {
                color: '#ffffff',
                width: 2
            }
        }
    };
    
    const layout = {
        title: '',
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: '#1a1f3a',
        font: { color: '#f8fafc' },
        xaxis: {
            title: 'Date',
            gridcolor: '#2a2f4a',
            showgrid: true
        },
        yaxis: {
            title: 'Value',
            gridcolor: '#2a2f4a',
            showgrid: true
        },
        hovermode: 'closest',
        showlegend: true,
        legend: {
            x: 0,
            y: 1,
            bgcolor: 'rgba(26, 31, 58, 0.8)',
            bordercolor: 'rgba(99, 102, 241, 0.3)',
            borderwidth: 1
        }
    };
    
    const config = { responsive: true, displayModeBar: false };
    
    Plotly.newPlot(container, [normalTrace, anomalyTrace], layout, config);
}

function plotEntropyEvolution() {
    const container = document.getElementById('entropy-chart');
    const windowSize = parseInt(document.getElementById('window-size').value);
    
    // Create timestamps for entropy values (starting from windowSize)
    const entropyTimestamps = currentTimestamps.slice(windowSize);
    
    const trace = {
        x: entropyTimestamps,
        y: detectionResults.entropies,
        mode: 'lines',
        name: 'Shannon Entropy',
        line: {
            color: '#8b5cf6',
            width: 2
        },
        fill: 'tozeroy',
        fillcolor: 'rgba(139, 92, 246, 0.1)'
    };
    
    const layout = {
        title: '',
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: '#1a1f3a',
        font: { color: '#f8fafc' },
        xaxis: {
            title: 'Date',
            gridcolor: '#2a2f4a'
        },
        yaxis: {
            title: 'Entropy',
            gridcolor: '#2a2f4a'
        },
        hovermode: 'closest'
    };
    
    const config = { responsive: true, displayModeBar: false };
    
    Plotly.newPlot(container, [trace], layout, config);
}

// Statistics Updates
function updateStatistics() {
    const anomalies = detectionResults.anomalies;
    
    // Count by severity
    const severityCounts = {
        severe: 0,
        moderate: 0,
        mild: 0
    };
    
    anomalies.forEach(a => {
        severityCounts[a.severity]++;
    });
    
    // Update DOM
    document.getElementById('total-anomalies').textContent = anomalies.length;
    document.getElementById('severe-count').textContent = severityCounts.severe;
    document.getElementById('moderate-count').textContent = severityCounts.moderate;
    document.getElementById('mild-count').textContent = severityCounts.mild;
    
    const detectionRate = ((anomalies.length / currentData.length) * 100).toFixed(2);
    document.getElementById('detection-rate').textContent = `${detectionRate}%`;
}

function updateAnomalyList() {
    const container = document.getElementById('anomaly-list');
    container.innerHTML = '';
    
    if (detectionResults.anomalies.length === 0) {
        container.innerHTML = `
            <div class="placeholder">
                <span class="placeholder-icon">✓</span>
                <p>No anomalies detected</p>
            </div>
        `;
        return;
    }
    
    detectionResults.anomalies.slice(0, 20).forEach(anomaly => {
        const item = document.createElement('div');
        item.className = `anomaly-item ${anomaly.severity}`;
        item.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <strong>${anomaly.timestamp}</strong>
                    <div style="font-size: 0.9rem; color: #94a3b8; margin-top: 4px;">
                        Value: ${anomaly.value.toFixed(2)} | Score: ${anomaly.score.toFixed(3)}
                    </div>
                </div>
                <span class="severity-badge" style="
                    padding: 4px 12px;
                    border-radius: 12px;
                    font-size: 0.8rem;
                    font-weight: 600;
                    background: ${anomaly.severity === 'severe' ? '#ef4444' : anomaly.severity === 'moderate' ? '#f59e0b' : '#10b981'};
                    color: white;
                ">
                    ${anomaly.severity.toUpperCase()}
                </span>
            </div>
        `;
        container.appendChild(item);
    });
    
    if (detectionResults.anomalies.length > 20) {
        const more = document.createElement('div');
        more.style.cssText = 'text-align: center; padding: 1rem; color: #94a3b8;';
        more.textContent = `... and ${detectionResults.anomalies.length - 20} more`;
        container.appendChild(more);
    }
}

function updateMetrics() {
    // Calculate average metrics
    const avgEntropy = detectionResults.entropies.reduce((a, b) => a + b, 0) / detectionResults.entropies.length;
    
    // Simulate other metrics (in production, these would come from backend)
    document.getElementById('shannon-entropy').textContent = avgEntropy.toFixed(3);
    document.getElementById('kl-divergence').textContent = (Math.random() * 0.5).toFixed(3);
    document.getElementById('js-divergence').textContent = (Math.random() * 0.3).toFixed(3);
    document.getElementById('transfer-entropy').textContent = (Math.random() * 0.2).toFixed(3);
    document.getElementById('mutual-info').textContent = (Math.random() * 0.4).toFixed(3);
}

// Export Functions
function exportResults(format) {
    if (!detectionResults) {
        alert('No results to export. Please run analysis first.');
        return;
    }
    
    if (format === 'json') {
        const json = JSON.stringify(detectionResults, null, 2);
        downloadFile('anomaly_results.json', json, 'application/json');
    } else if (format === 'csv') {
        const csv = convertToCSV(detectionResults.anomalies);
        downloadFile('anomaly_results.csv', csv, 'text/csv');
    } else if (format === 'report') {
        const html = generateReport();
        downloadFile('anomaly_report.html', html, 'text/html');
    }
}

function convertToCSV(anomalies) {
    const headers = ['Timestamp', 'Value', 'Score', 'Severity'];
    const rows = anomalies.map(a => [a.timestamp, a.value, a.score, a.severity]);
    
    return [
        headers.join(','),
        ...rows.map(row => row.join(','))
    ].join('\n');
}

function generateReport() {
    return `
<!DOCTYPE html>
<html>
<head>
    <title>Anomaly Detection Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .report { background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        h1 { color: #333; }
        .metric { display: inline-block; margin: 20px 30px 20px 0; }
        .metric-label { color: #666; font-size: 14px; }
        .metric-value { font-size: 32px; font-weight: bold; color: #6366f1; }
        table { width: 100%; border-collapse: collapse; margin-top: 30px; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #f8f9fa; font-weight: 600; }
    </style>
</head>
<body>
    <div class="report">
        <h1>Anomaly Detection Report</h1>
        <p>Generated: ${new Date().toLocaleString()}</p>
        <p>Method: ${detectionResults.method}</p>
        
        <div style="margin: 30px 0;">
            <div class="metric">
                <div class="metric-label">Total Anomalies</div>
                <div class="metric-value">${detectionResults.anomalies.length}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Detection Rate</div>
                <div class="metric-value">${((detectionResults.anomalies.length / currentData.length) * 100).toFixed(1)}%</div>
            </div>
        </div>
        
        <h2>Detected Anomalies</h2>
        <table>
            <thead>
                <tr>
                    <th>Timestamp</th>
                    <th>Value</th>
                    <th>Score</th>
                    <th>Severity</th>
                </tr>
            </thead>
            <tbody>
                ${detectionResults.anomalies.map(a => `
                    <tr>
                        <td>${a.timestamp}</td>
                        <td>${a.value.toFixed(2)}</td>
                        <td>${a.score.toFixed(3)}</td>
                        <td>${a.severity}</td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    </div>
</body>
</html>
    `;
}

function downloadFile(filename, content, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

// Utility Functions
function showLoading(show) {
    document.getElementById('loading-overlay').style.display = show ? 'flex' : 'none';
}
