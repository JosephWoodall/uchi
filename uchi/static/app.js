const chatWindow = document.getElementById('chat-window');
const inputForm = document.getElementById('input-form');
const promptInput = document.getElementById('prompt-input');
const modeBtns = document.querySelectorAll('.mode-btn');

let currentMode = 'stream';

const parametersPanel = document.getElementById('parameters-panel');
const tempSlider = document.getElementById('temp-slider');
const tempVal = document.getElementById('temp-val');
const creativitySlider = document.getElementById('creativity-slider');
const creativityVal = document.getElementById('creativity-val');

// Adjust textarea height automatically
promptInput.addEventListener('input', function() {
    this.style.height = '56px';
    this.style.height = (this.scrollHeight) + 'px';
});

tempSlider.addEventListener('input', (e) => tempVal.textContent = e.target.value);
creativitySlider.addEventListener('input', (e) => creativityVal.textContent = e.target.value);

// Handle mode switching
modeBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        modeBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentMode = btn.dataset.mode;
        
        // Show/hide parameter sliders only in Predict mode
        if (currentMode === 'predict') {
            parametersPanel.style.display = 'flex';
        } else {
            parametersPanel.style.display = 'none';
        }
        
        // Update placeholder based on mode
        if (currentMode === 'stream') promptInput.placeholder = 'Enter sequence tokens to train...';
        if (currentMode === 'query') promptInput.placeholder = 'Query the associative memory...';
        if (currentMode === 'predict') promptInput.placeholder = 'Provide context to predict future tokens...';
        
        promptInput.focus();
    });
});

// Enter to submit (Shift+Enter for newline)
promptInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        inputForm.dispatchEvent(new Event('submit'));
    }
});

function appendMessage(role, content, isHtml = false) {
    const div = document.createElement('div');
    div.className = `message ${role}-message`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    if (isHtml) {
        contentDiv.innerHTML = content;
    } else {
        contentDiv.textContent = content;
    }
    
    div.appendChild(contentDiv);
    chatWindow.appendChild(div);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    
    return contentDiv;
}

function showLoading() {
    const div = document.createElement('div');
    div.className = 'message ai-message';
    div.id = 'loading-indicator';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = '<div class="loader"><span></span><span></span><span></span></div>';
    
    div.appendChild(contentDiv);
    chatWindow.appendChild(div);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

function removeLoading() {
    const loader = document.getElementById('loading-indicator');
    if (loader) loader.remove();
}

inputForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = promptInput.value.trim();
    if (!text) return;
    
    promptInput.value = '';
    promptInput.style.height = '56px';
    
    appendMessage('user', `[${currentMode.toUpperCase()}] ${text}`);
    showLoading();
    
    const tokens = text.split(/\s+/);
    
    try {
        let response;
        let result;
        
        if (currentMode === 'stream') {
            response = await fetch('/stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ tokens })
            });
            if (response.ok) {
                appendMessage('system', `[+] Ingested ${tokens.length} tokens into the geometric trie.`);
            }
        } 
        else if (currentMode === 'query') {
            response = await fetch('/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ tokens })
            });
            result = await response.json();
            if (response.ok) {
                appendMessage('ai', result.answer || '[Unknown Context]');
            }
        }
        else if (currentMode === 'predict') {
            response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    context: tokens, 
                    steps: 10,
                    temperature: parseFloat(tempSlider.value),
                    creativity: parseFloat(creativitySlider.value)
                })
            });
            result = await response.json();
            if (response.ok) {
                const predStr = result.prediction.join(' ');
                appendMessage('ai', `<span class="highlight">Prediction:</span> ${predStr}`, true);
            }
        }
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
    } catch (error) {
        appendMessage('system', `[-] Error: ${error.message}`);
    } finally {
        removeLoading();
    }
});
