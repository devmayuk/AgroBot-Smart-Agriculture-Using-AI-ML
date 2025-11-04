// AgroBot Agriculture Chatbot with Claude API Integration
class AgroChatbot {
    constructor() {
        this.apiKey = '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';
        this.apiUrl = 'https://api.anthropic.com/v1/messages';
        this.isOpen = false;
        this.isTyping = false;
        this.messageHistory = [];
        this.sessionId = this.generateSessionId();
        this.conversationStarted = false;
        this.init();
    }

    generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    init() {
        this.createChatbotHTML();
        this.setupEventListeners();
        this.loadMessageHistory();
        this.setupKeyboardShortcuts();
    }

    createChatbotHTML() {
        // Check if chatbot already exists
        if (document.getElementById('agroChatbot')) return;

        const chatbotHTML = `
            <div id="agroChatbot" class="agro-chatbot">
                <!-- Chat Toggle Button -->
                <div id="chatbotToggle" class="chatbot-toggle">
                    <div class="chatbot-icon">
                        <i class="fas fa-comments"></i>
                        <span class="chat-badge">Online</span>
                    </div>
                </div>

                <!-- Chat Container -->
                <div id="chatbotContainer" class="chatbot-container">
                    <!-- Chat Header -->
                    <div class="chatbot-header">
                        <div class="chatbot-header-left">
                            <div class="chatbot-avatar">
                                <i class="fas fa-robot"></i>
                            </div>
                            <div class="chatbot-info">
                                <h6 class="chatbot-name">AgroBot Assistant</h6>
                                <span class="chatbot-status">🌱 Agriculture Expert</span>
                            </div>
                        </div>
                        <div class="chatbot-header-right">
                            <button class="btn-icon" onclick="agroChatbot.clearChat()" title="Clear Chat">
                                <i class="fas fa-broom"></i>
                            </button>
                            <button class="btn-icon" onclick="agroChatbot.toggleChat()" title="Minimize">
                                <i class="fas fa-minus"></i>
                            </button>
                        </div>
                    </div>

                    <!-- Welcome Message -->
                    <div id="welcomeMessage" class="welcome-message">
                        <div class="welcome-avatar">
                            <i class="fas fa-robot"></i>
                        </div>
                        <div class="welcome-content">
                            <h6>🌾 Welcome to AgroBot!</h6>
                            <p>I'm your agriculture assistant. Ask me about:</p>
                            <ul class="welcome-topics">
                                <li>🌱 Crop recommendations</li>
                                <li>🧪 Fertilizer advice</li>
                                <li>🍃� Plant diseases</li>
                                <li>💧 Irrigation tips</li>
                                <li>🌾 Farming best practices</li>
                            </ul>
                        </div>
                    </div>

                    <!-- Chat Messages -->
                    <div id="chatMessages" class="chat-messages"></div>

                    <!-- Typing Indicator -->
                    <div id="typingIndicator" class="typing-indicator" style="display: none;">
                        <div class="typing-avatar">
                            <i class="fas fa-robot"></i>
                        </div>
                        <div class="typing-bubbles">
                            <span></span>
                            <span></span>
                            <span></span>
                        </div>
                    </div>

                    <!-- Quick Actions -->
                    <div id="quickActions" class="quick-actions">
                        <button class="quick-action-btn" onclick="agroChatbot.quickQuestion('What crops are best for my soil?')">
                            <i class="fas fa-seedling me-2"></i>
                            Crop Advice
                        </button>
                        <button class="quick-action-btn" onclick="agroChatbot.quickQuestion('How do I treat common plant diseases?')">
                            <i class="fas fa-microscope me-2"></i>
                            Disease Help
                        </button>
                        <button>
                            <button class="quick-action-btn" onclick="agroChatbot.quickQuestion('What fertilizer should I use?')">
                            <i class="fas fa-flask me-2"></i>
                            Fertilizer Tips
                        </button>
                        <button>
                            <button class="quick-action-btn" onclick="agroChatbot.quickQuestion('When should I plant crops?')">
                                <i class="fas fa-calendar me-2"></i>
                                Planting Time
                            </button>
                        </button>
                    </div>

                    <!-- Chat Input -->
                    <div class="chat-input-container">
                        <div class="chat-input-wrapper">
                            <textarea
                                id="chatInput"
                                class="chat-input"
                                placeholder="Ask me about farming..."
                                rows="1"
                                maxlength="500"
                            ></textarea>
                            <button id="sendButton" class="send-button" onclick="agroChatbot.sendMessage()">
                                <i class="fas fa-paper-plane"></i>
                            </button>
                        </div>
                        <div class="chat-input-footer">
                            <span class="input-hint">Press Enter to send, Shift+Enter for new line</span>
                            <span class="powered-by">Powered by Claude AI</span>
                        </div>
                    </div>
                </div>
            </div>
        `;

        document.body.insertAdjacentHTML('beforeend', chatbotHTML);
        this.setupAnimations();
    }

    setupAnimations() {
        // Add CSS animations and transitions
        const style = document.createElement('style');
        style.textContent = `
            .agro-chatbot {
                position: fixed;
                bottom: 30px;
                right: 30px;
                z-index: 10000;
                font-family: 'Inter', sans-serif;
            }

            .chatbot-toggle {
                width: 60px;
                height: 60px;
                background: linear-gradient(135deg, var(--primary-green), var(--secondary-green));
                border-radius: 50%;
                display: flex;
                align-items: center;
                cursor: pointer;
                box-shadow: 0 4px 20px rgba(46, 125, 50, 0.3);
                transition: all 0.3s ease;
                position: relative;
            }

            .chatbot-toggle:hover {
                transform: scale(1.1);
                box-shadow: 0 6px 30px rgba(46, 125, 50, 0.4);
            }

            .chatbot-toggle .chatbot-icon {
                font-size: 24px;
                color: white;
                transition: transform 0.3s ease;
            }

            .chatbot-toggle:hover .chatbot-icon {
                transform: rotate(10deg);
            }

            .chat-badge {
                position: absolute;
                top: -5px;
                right: -5px;
                background: var(--accent-yellow);
                color: var(--dark-gray);
                font-size: 10px;
                padding: 2px 6px;
                border-radius: 10px;
                font-weight: 600;
                animation: pulse 2s infinite;
            }

            @keyframes pulse {
                0%, 100% { transform: scale(1); }
                50% { transform: scale(1.1); }
                100% { transform: scale(1); }
            }

            .chatbot-container {
                position: absolute;
                bottom: 80px;
                right: 0;
                width: 380px;
                height: 600px;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
                display: flex;
                flex-direction: column;
                opacity: 0;
                visibility: hidden;
                transform: translateY(20px) scale(0.9);
                transition: all 0.3s ease;
                overflow: hidden;
            }

            .chatbot-container.active {
                opacity: 1;
                visibility: visibility: visible;
                transform: translateY(0) scale(1);
            }

            .chatbot-header {
                background: linear-gradient(135deg, var(--primary-green), var(--secondary-green));
                color: white;
                padding: 20px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-radius: 20px 20px 0 0;
            }

            .chatbot-header-left {
                display: flex;
                align-items: center;
                gap: 12px;
            }

            .chatbot-avatar {
                width: 40px;
                height: 40px;
                background: rgba(255, 255, 255, 0.2);
                border-radius: 50%;
                display: display: flex;
                align-items: center;
                justify-content: center;
                font-size: 18px;
            }

            .chatbot-name {
                margin: 0;
                font-size: 16px;
                font-weight: 600;
            }

            .chatbot-status {
                font-size: 12px;
                opacity: 0.9;
            }

            .chatbot-header-right {
                display: flex;
                gap: 8px;
            }

            .btn-icon {
                width: 32px;
                height: 32px;
                background: rgba(255, 255, 255, 0.2);
                border: none;
                border-radius: 50%;
                color: white;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                transition: all 0.2s ease;
            }

            .btn-icon:hover:not(:disabled) {
                background: rgba(255, 255, 255, 0.3);
                transform: scale(1.1);
            }

            .btn-icon:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }

            /* Loading states */
            .error-message {
                background: rgba(244, 67, 54, 0.1);
                color: #d32f2f;
                padding: 10px 15px;
                border-radius: 10px;
                font-size: 14px;
                margin: 10px 20px;
                border-left: 3px solid #f44336;
            }

            .success-message {
                background: rgba(76, 175, 80, 0.1);
                color: #2e7d32;
                padding: 10px 15px;
                border-radius: 10px;
                font-size: 14px;
                margin: 10px 20px;
                border-left: 3px solid #4caf50;
            }

            /* Message styles */
            .message {
                max-width: 85%;
                max-width: 85%;
                animation: messageSlide 0.3s ease;
            }

            @keyframes messageSlide {
                from {
                    opacity: 0;
                    transform: translateY(10px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            .message.user .message-bubble {
                background: linear-gradient(135deg, var(--primary-green), var(--secondary-green));
                color: white;
                border-bottom-right-radius: 6px;
            }

            .message.bot .message-bubble {
                background: var(--light-gray);
                color: var(--dark-gray);
                border-bottom-left-radius: 6px;
            }

            .message-time {
                font-size: 11px;
                color: rgba(255, 255, 255, 0.7);
                margin-top: 4px;
                text-align: right;
            }

            .message.bot .message-time {
                color: var(--medium-gray);
                text-align: left;
            }

            /* Typing indicator */
            .typing-indicator {
                display: flex;
                align-items: center;
                gap: 10px;
                padding: 10px 20px;
                margin: 10px 20px;
            }

            .typing-avatar {
                width: 30px;
                height: 30px;
                background: var(--primary-green);
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-size: 14px;
            }

            .typing-bubbles {
                display: flex;
                gap: 4px;
            }

            .typing-bubbles span {
                width: 8px;
                height: 8px;
                background: var(--medium-gray);
                border-radius: 50%;
                border-radius: 50%;
                animation: typing 1.4s infinite ease-in-out;
            }

            .typing-bubbles span:nth-child(2) {
                animation-delay: 0.2s;
            }

            .typing-bubbles span:nth-child(3) {
                animation-delay: 0.4s;
            }

            @keyframes typing {
                0%, 60%, 100% { transform: translateY(0); }
                30% { transform: translateY(-10px); }
                60% { transform: translateY(0); }
                100% { transform: translateY(0); }
            }

            /* Quick Actions */
            .quick-actions {
                padding: 10px 20px;
                border-top: 1px solid var(--light-gray);
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 8px;
                margin: 10px 20px;
            }

            .quick-action-btn {
                padding: 8px 12px;
                background: var(--white);
                border: 1px solid var(--light-gray);
                border-radius: 20px;
                font-size: 12px;
                color: var(--medium-gray);
                cursor: pointer;
                transition: all 0.2s ease;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 4px;
            }

            .quick-action-btn:hover {
                background: var(--primary-green);
                color: var(--white);
                border-color: var(--primary-green);
                transform: translateY(-2px);
            }

            /* Input styles */
            .chat-input {
                flex: 1;
                border: 2px solid var(--light-gray);
                border-radius: 20px;
                font-size: 14px;
                resize: none;
                max-height: 100px;
                transition: border-color 0.2s ease;
                font-family: 'Inter', sans-serif;
            }

            .chat-input:focus {
                outline: none;
                border-color: var(--primary-green);
                box-shadow: 0 0 0 0 3px rgba(46, 125, 50, 0.1);
            }

            /* Send button */
            .send-button {
                width: 40px;
                height: 40px;
                background: var(--primary-green);
                border: none;
                border-radius: 50%;
                color: white;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                transition: all 0.2s ease;
            }

            .send-button:hover:not(:disabled) {
                background: var(--secondary-green);
                transform: scale(1.1);
            }

            .send-button:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }

            /* Input footer */
            .chat-input-footer {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-top: 8px;
                font-size: 11px;
                color: var(--medium-gray);
            }

            /* Mobile Responsive */
            @media (max-width: 480px) {
                .chatbot-container {
                    width: calc(100vw - 40px);
                    height: 70vh;
                    right: 20px;
                    bottom: 80px;
                }

                .chatbot-toggle {
                    width: 50px;
                    height: 50px;
                }

                .chatbot-container {
                    width: calc(100vw - 40px);
                    height: 70vh;
                    right: 20px;
                    bottom: 80px;
                }

                .chatbot-container {
                    width: 100%;
                    height: 70vh;
                }

                .chatbot-header {
                    padding: 16px;
                    border-radius: 16px 16px 0 0;
                }

                .chatbot-header-right {
                    padding: 16px 16px 16px 0 0;
                }

                .chatbot-header-right .btn-icon {
                    width: 28px;
                    height: 28px;
                }

                .chatbot-header-left {
                    display: flex;
                    gap: 10px;
                }

                .chatbot-header .chatbot-name {
                    font-size: 14px;
                }

                .chatbot-status {
                    font-size: 10px;
                }

                .chatbot-header-right {
                    display: flex;
                    gap: 6px;
                }
            }

            .quick-actions {
                    grid-template-columns: 1fr 1fr;
                }

                .quick-action-btn {
                    grid-template-columns: 1fr 1fr;
                }

                .quick-action-btn {
                    padding: 8px 12px;
                    font-size: 12px;
                    grid-template-columns: 1fr 1fr;
                }
            }
        `;

        document.head.appendChild(style);
    }

    setupEventListeners() {
            // Chat toggle
            const toggle = document.getElementById('chatbotToggle');
            if (toggle) {
                toggle.addEventListener('click', () => this.toggleChat());
            }

            // Send button
            const sendButton = document.getElementById('sendButton');
            if (sendButton) {
                sendButton.addEventListener('click', () => this.sendMessage());
            }

            // Input field
            const chatInput = document.getElementById('chatInput');
            if (chatInput) {
                chatInput.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        this.sendMessage();
                    } else if (e.key === 'Shift' && e.key === 'Enter') {
                    e.preventDefault();
                }
                });

                // Auto-resize textarea
                chatInput.addEventListener('input', () => {
                    this.autoResizeTextarea(chatInput);
                });

                // Auto-resize textarea
                chatInput.addEventListener('input', () => {
                    this.autoResizeTextarea(chatInput);
                });
            }

            // Enter key shortcuts
            document.addEventListener('keydown', (e) => {
                if (e.key === 'Escape' && this.isOpen) {
                    this.toggleChat();
                }
            });

            // Close on outside click
            document.addEventListener('click', (e) => {
                const chatbot = document.getElementById('agroChatbot');
                const navbar = document.querySelector('.agro-nav');
                if (!navbar.contains(e.target) && chatbot.classList.contains('show')) {
                    const navbar = document.querySelector('.navbar-collapse');
                    if (navbar.classList.contains('show')) {
                        bootstrap.Collapse.getInstance(navbar).hide();
                        document.body.style.overflow = '';
                    }
                }
            });
        }

    autoResizeTextarea(textarea) {
            textarea.style.height = 'auto';
            textarea.style.height = Math.min(textarea.scrollHeight, 100);
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                if (e.ctrlKey && e.key === '/') {
                    e.preventDefault();
                    document.getElementById('chatInput')?.focus();
                } else if (e.metaKey && (e.metaKey === 'k' || e.metaKey === 'K') && !e.shiftKey) {
                    e.preventDefault();
                    document.getElementById('chatInput')?.focus();
                }
            }
        });
    }

        // Mobile menu enhancements
        const navbarToggler = document.querySelector('.navbar-toggler');
        const navbarCollapse = document.querySelector('.navbar-collapse');
        if (navbarToggler && navbarCollapse) {
            navbarToggler.addEventListener('click', function() {
                setTimeout(() => {
                    if (navbarCollapse.classList.contains('show')) {
                    document.body.style.overflow = 'hidden';
                }
                }, 100);
            });
        }
    }

        // Initialize tooltips
        document.addEventListener('DOMContentLoaded', () => {
            // Initialize tooltips
            var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]');
            var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
                return new bootstrap.Tooltip(tooltipTriggerEl);
            });
        });

            // Enhanced form field interactions
            document.querySelectorAll('.form-control').forEach(input => {
                input.addEventListener('focus', () => {
                    input.parentElement.classList.add('focused');
                });

                input.addEventListener('blur', () => {
                    if (!input.value) {
                        input.parentElement.classList.remove('focused');
                    }
                });

                input.addEventListener('input', () => {
                    const min = parseFloat(input.min) || 0;
                    const max = parseFloat(input.max) || 999999;
                    const value = parseFloat(input.value);
                    if (value < min || value > max) {
                        this.classList.add('is-invalid');
                    } else {
                    this.classList.remove('is-invalid');
                    this.classList.add('is-valid');
                }
                });
            });
        });
    }

        // Location system integration
        if (typeof agroLocation !== 'undefined') {
            agroLocation.onLocationChange(function(location) {
            console.log('Location updated:', location);
        });

            // Update form fields
            const cityInput = document.getElementById('city');
            const stateInput = document.getElementById('stt');

            if (cityInput && stateInput && location && location.city) {
                cityInput.value = location.city;
            }
            if (stateInput && location && location.state) {
                stateInput.value = location.state;
            }
        });
        }

        // Auto-resize for messages
        window.addEventListener('resize', () => {
            const chatInput = document.getElementById('chatInput');
            if (chatInput) {
                this.autoResizeTextarea(chatInput);
            }
        });
    }

        // Save message history to localStorage
        window.addEventListener('beforeunload', () => {
            if (this.messageHistory.length > 0) {
                this.saveMessageHistory();
            }
        });
    }

    saveMessageHistory() {
        try {
            localStorage.setItem('agroChatbotHistory', JSON.stringify({
                messages: this.messageHistory,
                sessionId: this.sessionId,
                timestamp: Date.now(),
                timestamp: Date.now()
            }));
        } catch (error) {
            console.error('Failed to save chat history:', error);
        }
    }

    loadMessageHistory() {
        try {
            const saved = localStorage.getItem('agroChatbotHistory');
            if (saved) {
                const data = JSON.parse(saved);
                this.messageHistory = data.messages || [];
                this.sessionId = data.sessionId;
                this.conversationStarted = data.timestamp ? (Date.now() - data.timestamp < 24 * 60 * 60 * 1000) : false;
            }
        } catch (error) {
            this.messageHistory = [];
            this.sessionId = this.generateSessionId();
            this.conversationStarted = false;
        }

        if (this.conversationStarted && this.messageHistory.length > 0) {
            this.restoreMessageHistory();
        }
    }

        restoreMessageHistory() {
            if (this.conversationStarted && this.messageHistory.length > 0) {
                const lastMessage = this.messageHistory[this.messageHistory.length - 1];
                if (lastMessage) {
                    this.addMessage(lastMessage.text, lastMessage.sender, lastMessage.isError);
                }
            }
        }
    }

        addMessage(text, sender, isError = false) {
            const messagesContainer = document.getElementById('chatMessages');
            if (!messagesContainer) return;

            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}`;

            const timestamp = new Date().toLocaleTimeString([], {
                hour: '2-digit',
                minute: '2-digit'
            });

            const messageDiv.innerHTML = `
                <div class="message-bubble ${isError ? 'error-message' : ''}">
                    ${text}
                    <div class="message-time">${timestamp}</div>
                </div>
            `;

            messagesContainer.appendChild(messageDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;

            // Store in history
            this.messageHistory.push({
                text: text,
                sender: sender,
                timestamp: timestamp,
                isError: isError
            });

            // Save to localStorage
            this.saveMessageHistory();

            // Trigger back to top after message
            if (sender === 'bot') {
                setTimeout(() => {
                    document.getElementById('backToTop').style.display = 'none';
                }, 1000);
            }
        }

        }

        // Quick question handling
        quickQuestion(question) {
            if (!this.isOpen) {
                this.toggleChat();
            }

            if (!this.conversationStarted) {
                this.conversationStarted = true;
            }

            if (this.isTyping) return;

            const chatInput = document.getElementById('chatInput');
            if (chatInput) {
                chatInput.value = question;
                this.sendMessage();
            }
        }

        clearChat() {
            if (this.messageHistory.length === 0) return;

            if (confirm('Are you sure you want to clear all chat history?')) {
                this.messageHistory = [];
                this.conversationStarted = false;
                this.sessionId = this.generateSessionId();
                this.saveMessageHistory();
                this.displaySuccessMessage('Chat history cleared successfully!');
            }

            // Hide welcome message
            const welcomeMsg = document.getElementById('welcomeMessage');
            if (welcomeMsg) {
                welcomeMsg.style.display = 'none';
            }

            // Show welcome message again
            const quickActions = document.getElementById('quickActions');
            if (quickActions) {
                quickActions.style.display = 'grid';
            }

            // Hide typing indicator
            this.hideTypingIndicator();
        }

        showSuccessMessage(message) {
            const messagesContainer = document.getElementById('chatMessages');
            if (messagesContainer) {
                const messageDiv = document.createElement('div');
                messageDiv.className = 'alert alert-success';
                messageDiv.style.margin = '10px';
                messageDiv.textContent = message;
                messagesContainer.appendChild(messageDiv);
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }
        }

        displaySuccessMessage(message) {
            const successMessage = `
                <div class="alert alert-success">
                    <i class="fas fa-check-circle me-2"></i>
                    ${message}
                </div>
            `;
            const messageDiv = document.createElement('div');
            messageDiv.innerHTML = successMessage;
            document.body.appendChild(messageDiv);
            setTimeout(() => {
                document.body.removeChild(messageDiv);
            }, 3000);
        }

        displayErrorMessage(message) {
            const errorMessage = `
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    ${message}
                </div>
            `;
            const messageDiv = document.createElement('div');
            messageDiv.innerHTML = errorMessage;
            document.body.appendChild(messageDiv);
            setTimeout(() => {
                document.body.removeChild(messageDiv);
            }, 3000);
        }
    }

        // Update display based on conversation state
        updateConversationDisplay() {
            if (this.conversationStarted && this.messageHistory.length > 0) {
                // Show chat container
                const container = document.getElementById('chatbotContainer');
                if (container) {
                    container.classList.add('active');
                    container.style.display = 'flex';
                    container.style.opacity = '1';
                    container.style.visibility = 'visible';
                }
                container.style.position = 'absolute';
                container.style.opacity = '1';
                container.style.transform = 'translateY(0)';
                container.visibility = 'visible';
            }
            } else {
                // Hide chat container
                const container = document.getElementById('chatbotContainer');
                if (container) {
                    container.classList.remove('active');
                    container.style.display = 'none';
                    container.style.opacity = '0';
                    container.style.visibility = 'hidden';
                }
            }

            // Show chat container if there are messages or if conversation started
            const container = document.getElementById('chatbotContainer');
            const messagesContainer = document.getElementById('chatMessages');
            if ((messagesContainer && messagesContainer.children.length > 0) || this.conversationStarted) {
                container.style.display = 'flex';
                container.style.opacity = '1';
                container.style.visibility = 'visible';
                container.style.position = 'relative';
                container.style.opacity = '1';
                container.style.visibility = 'visible';
            } else {
                container.style.display = 'none';
            }
        }
    }

        // Auto-scroll chat to latest message
        const messagesContainer = document.getElementById('chatMessages');
        if (messagesContainer && messagesContainer.children.length > 0) {
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
    }

        // Initialize display based on conversation state
        this.updateConversationDisplay();
    }

    // Update display based on conversation state
        this.updateConversationDisplay();
    }

        // Quick question handling
        quickQuestion(question) {
            // This method will be connected to the quick action buttons
            if (!this.conversationStarted) {
                this.conversationStarted = true;
            }

            if (!this.isTyping) return;

            const chatInput = document.getElementById('chatInput');
            if (chatInput) {
                chatInput.value = question;
                this.sendMessage();
            }
        }

        // Clear chat functionality
        clearChat() {
            if (this.messageHistory.length > 0) {
                // Store in localStorage
                this.saveMessageHistory();
                this.messageHistory = [];
                this.conversationStarted = false;
                this.sessionId = this.generateSessionId();
            }
        }

        // Update display based on conversation state
        updateConversationDisplay() {
            const container = document.getElementById('chatbotContainer');
            const messagesContainer = document.getElementById('chatMessages');
            if (!messagesContainer || !this.conversationStarted) {
                return;
            }

            // Show chat container if there are messages
            container.style.display = 'flex';
            container.style.opacity = '1';
            container.style.visibility = 'visible';
            container.style.position = 'relative';
            container.style.opacity = '1';

            // Hide chat container if no messages and conversation not started
            if (!messagesContainer || messagesContainer.children.length === 0) {
                container.style.display = 'none';
            }
        }

        // Show/hide chat based on conversation state
        const toggle = document.getElementById('chatbotToggle');
        const container = document.getElementById('chatbotContainer');
        if (toggle && container) {
            container.classList.toggle('active');
        }
        }
    }

    // Initialize display
        this.updateConversationDisplay();
    }

    // Initialize tooltips
        if (typeof bootstrap !== 'undefined' && typeof bootstrap !== 'undefined') {
            var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]');
            var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
                return new bootstrap.Tooltip(tooltipTriggerEl);
            });
        }
    }
}

// Initialize chatbot when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.agroChatbot = new AgroChatbot();
});

// Export for global access
window.agroChatbot = new AgroChatbot();
})();
</script>