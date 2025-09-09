// WebSocket Management for Enhanced Trading Platform
// This file handles WebSocket connections and real-time updates

class WebSocketManager {
    constructor() {
        this.socket = null;
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        
        this.init();
    }
    
    init() {
        this.connect();
    }
    
    connect() {
        try {
            // Use Socket.IO if available, otherwise fallback to WebSocket
            if (typeof io !== 'undefined') {
                this.connectSocketIO();
            } else {
                this.connectWebSocket();
            }
        } catch (error) {
            console.error('WebSocket connection failed:', error);
            this.handleConnectionError();
        }
    }
    
    connectSocketIO() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}`;
        
        this.socket = io(wsUrl, {
            transports: ['websocket', 'polling'],
            timeout: 20000,
            reconnection: true,
            reconnectionDelay: this.reconnectDelay,
            reconnectionAttempts: this.maxReconnectAttempts
        });
        
        this.setupSocketIOEvents();
    }
    
    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        this.socket = new WebSocket(wsUrl);
        this.setupWebSocketEvents();
    }
    
    setupSocketIOEvents() {
        this.socket.on('connect', () => {
            console.log('Socket.IO connected');
            this.isConnected = true;
            this.reconnectAttempts = 0;
            this.onConnect();
        });
        
        this.socket.on('disconnect', () => {
            console.log('Socket.IO disconnected');
            this.isConnected = false;
            this.onDisconnect();
        });
        
        this.socket.on('reconnect', () => {
            console.log('Socket.IO reconnected');
            this.onReconnect();
        });
        
        this.socket.on('connect_error', (error) => {
            console.error('Socket.IO connection error:', error);
            this.handleConnectionError();
        });
    }
    
    setupWebSocketEvents() {
        this.socket.onopen = () => {
            console.log('WebSocket connected');
            this.isConnected = true;
            this.reconnectAttempts = 0;
            this.onConnect();
        };
        
        this.socket.onclose = () => {
            console.log('WebSocket disconnected');
            this.isConnected = false;
            this.onDisconnect();
            this.attemptReconnect();
        };
        
        this.socket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.handleConnectionError();
        };
        
        this.socket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleMessage(data);
            } catch (error) {
                console.error('Failed to parse WebSocket message:', error);
            }
        };
    }
    
    onConnect() {
        // Update connection status in UI
        this.updateConnectionStatus('connected');
        
        // Subscribe to updates
        this.subscribe();
        
        // Trigger custom event
        window.dispatchEvent(new CustomEvent('websocket-connected'));
    }
    
    onDisconnect() {
        // Update connection status in UI
        this.updateConnectionStatus('disconnected');
        
        // Trigger custom event
        window.dispatchEvent(new CustomEvent('websocket-disconnected'));
    }
    
    onReconnect() {
        // Re-subscribe to updates
        this.subscribe();
        
        // Trigger custom event
        window.dispatchEvent(new CustomEvent('websocket-reconnected'));
    }
    
    handleConnectionError() {
        this.isConnected = false;
        this.updateConnectionStatus('error');
        
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.attemptReconnect();
        } else {
            console.error('Max reconnection attempts reached');
            this.updateConnectionStatus('failed');
        }
    }
    
    attemptReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            return;
        }
        
        this.reconnectAttempts++;
        console.log(`Attempting to reconnect... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
        
        setTimeout(() => {
            this.connect();
        }, this.reconnectDelay * this.reconnectAttempts);
    }
    
    subscribe() {
        if (!this.isConnected) return;
        
        const currentPage = this.getCurrentPage();
        const subscriptionData = {
            page: currentPage,
            updates: ['system_status', 'scan_results', 'market_data', 'account_update']
        };
        
        if (this.socket && typeof this.socket.emit === 'function') {
            // Socket.IO
            this.socket.emit('subscribe', subscriptionData);
        } else if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            // WebSocket
            this.socket.send(JSON.stringify({
                type: 'subscribe',
                data: subscriptionData
            }));
        }
    }
    
    handleMessage(data) {
        switch (data.type) {
            case 'system_status':
                window.dispatchEvent(new CustomEvent('system-status-update', { detail: data.data }));
                break;
            case 'scan_results':
                window.dispatchEvent(new CustomEvent('scan-results-update', { detail: data.data }));
                break;
            case 'market_data':
                window.dispatchEvent(new CustomEvent('market-data-update', { detail: data.data }));
                break;
            case 'account_update':
                window.dispatchEvent(new CustomEvent('account-update', { detail: data.data }));
                break;
            case 'golden_opportunity':
                window.dispatchEvent(new CustomEvent('golden-opportunity', { detail: data.data }));
                break;
            default:
                console.log('Unknown message type:', data.type);
        }
    }
    
    updateConnectionStatus(status) {
        // Update WebSocket connection indicator
        const wsStatus = document.querySelector('#ws-status');
        if (wsStatus) {
            wsStatus.className = `status-indicator status-${status}`;
            wsStatus.innerHTML = `
                <div class="status-dot"></div>
                <span>WebSocket ${status.charAt(0).toUpperCase() + status.slice(1)}</span>
            `;
        }
        
        // Update general connection status
        const connectionElements = document.querySelectorAll('.connection-status');
        connectionElements.forEach(element => {
            element.className = `connection-status ${status}`;
            element.textContent = status.charAt(0).toUpperCase() + status.slice(1);
        });
    }
    
    getCurrentPage() {
        const path = window.location.pathname;
        if (path === '/' || path === '/dashboard') return 'dashboard';
        if (path.includes('scanner')) return 'scanner';
        if (path.includes('trade-builder')) return 'trade_builder';
        if (path.includes('execution')) return 'execution';
        if (path.includes('alerts')) return 'alerts';
        return 'dashboard';
    }
    
    send(data) {
        if (!this.isConnected) {
            console.warn('WebSocket not connected, cannot send data');
            return false;
        }
        
        try {
            if (this.socket && typeof this.socket.emit === 'function') {
                // Socket.IO
                this.socket.emit(data.type, data.data);
            } else if (this.socket && this.socket.readyState === WebSocket.OPEN) {
                // WebSocket
                this.socket.send(JSON.stringify(data));
            }
            return true;
        } catch (error) {
            console.error('Failed to send WebSocket message:', error);
            return false;
        }
    }
    
    disconnect() {
        if (this.socket) {
            if (typeof this.socket.disconnect === 'function') {
                // Socket.IO
                this.socket.disconnect();
            } else {
                // WebSocket
                this.socket.close();
            }
        }
        this.isConnected = false;
    }
}

// Initialize WebSocket manager when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.wsManager = new WebSocketManager();
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = WebSocketManager;
}

