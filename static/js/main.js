// Enhanced Options Trading Platform - Main JavaScript
// Application Logic and UI Interactions

class TradingPlatform {
    constructor() {
        this.socket = null;
        this.isConnected = false;
        this.currentPage = this.getCurrentPage();
        this.systemStatus = {
            ibkr: 'disconnected',
            scanner: 'stopped',
            telegram: 'disconnected'
        };
        
        this.init();
    }
    
    init() {
        this.initializeWebSocket();
        this.initializeNavigation();
        this.initializeStatusIndicators();
        this.initializePageSpecificFeatures();
        this.startHeartbeat();
        
        console.log('Trading Platform initialized');
    }
    
    // WebSocket Management
    initializeWebSocket() {
        // Check if Socket.IO is available
        if (typeof io === 'undefined') {
            console.warn('Socket.IO not loaded, using fallback status updates');
            this.initializeFallbackUpdates();
            return;
        }
        
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}`;
        
        try {
            this.socket = io(wsUrl, {
                transports: ['websocket', 'polling'],
                timeout: 20000,
                reconnection: true,
                reconnectionDelay: 1000,
                reconnectionAttempts: 5
            });
            
            this.setupSocketEvents();
        } catch (error) {
            console.error('WebSocket initialization failed:', error);
            this.initializeFallbackUpdates();
        }
    }
    
    setupSocketEvents() {
        this.socket.on('connect', () => {
            console.log('WebSocket connected');
            this.isConnected = true;
            this.updateConnectionStatus('connected');
            this.subscribeToUpdates();
        });
        
        this.socket.on('disconnect', () => {
            console.log('WebSocket disconnected');
            this.isConnected = false;
            this.updateConnectionStatus('disconnected');
        });
        
        this.socket.on('reconnect', () => {
            console.log('WebSocket reconnected');
            this.subscribeToUpdates();
        });
        
        this.socket.on('system_status', (data) => {
            this.updateSystemStatus(data);
        });
        
        this.socket.on('scan_results', (data) => {
            this.updateScanResults(data);
        });
        
        this.socket.on('golden_opportunity', (data) => {
            this.showGoldenOpportunityAlert(data);
        });
        
        this.socket.on('market_data', (data) => {
            this.updateMarketData(data);
        });
        
        this.socket.on('account_update', (data) => {
            this.updateAccountData(data);
        });
        
        this.socket.on('error', (error) => {
            console.error('WebSocket error:', error);
            this.showNotification('Connection error: ' + error.message, 'error');
        });
    }
    
    // Fallback for when WebSocket is not available
    initializeFallbackUpdates() {
        console.log('Initializing fallback status updates');
        
        // Simulate connection attempts and status updates
        setTimeout(() => {
            this.updateSystemStatus({
                ibkr: 'connecting',
                scanner: 'stopped',
                telegram: 'connected'
            });
        }, 1000);
        
        setTimeout(() => {
            this.updateSystemStatus({
                ibkr: 'disconnected', // Simulate IBKR disconnected (normal for demo)
                scanner: 'connected',
                telegram: 'connected'
            });
        }, 3000);
        
        // Periodic status checks
        setInterval(() => {
            this.fetchSystemStatus();
        }, 30000); // Check every 30 seconds
    }
    
    subscribeToUpdates() {
        if (this.socket && this.isConnected) {
            this.socket.emit('subscribe', {
                page: this.currentPage,
                updates: ['system_status', 'scan_results', 'market_data', 'account_update']
            });
        }
    }
    
    // Navigation Management
    initializeNavigation() {
        const navLinks = document.querySelectorAll('.nav-link');
        const menuToggle = document.querySelector('.menu-toggle');
        const sidebar = document.querySelector('.sidebar');
        const mainContent = document.querySelector('.main-content');
        
        // Handle navigation clicks - Fixed to use nav-link instead of nav-item
        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                // Allow normal navigation to work - don't prevent default
                const href = link.getAttribute('href');
                if (href && href !== '#') {
                    // Update active state
                    navLinks.forEach(l => l.classList.remove('active'));
                    link.classList.add('active');
                    
                    // Let the browser handle navigation naturally
                    // this.navigateTo(href); // Removed to allow normal navigation
                }
            });
        });
        
        // Handle mobile menu toggle
        if (menuToggle) {
            menuToggle.addEventListener('click', () => {
                sidebar.classList.toggle('open');
                mainContent.classList.toggle('expanded');
            });
        }
        
        // Close mobile menu when clicking outside
        document.addEventListener('click', (e) => {
            if (window.innerWidth <= 768) {
                if (!sidebar.contains(e.target) && !menuToggle.contains(e.target)) {
                    sidebar.classList.remove('open');
                    mainContent.classList.remove('expanded');
                }
            }
        });
        
        // Handle window resize
        window.addEventListener('resize', () => {
            if (window.innerWidth > 768) {
                sidebar.classList.remove('open');
                mainContent.classList.remove('expanded');
            }
        });
    }
    
    navigateTo(path) {
        window.location.href = path;
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
    
    // Status Management
    initializeStatusIndicators() {
        this.updateSystemStatus({
            ibkr: 'connecting',
            scanner: 'stopped',
            telegram: 'connected'
        });
        
        // Fetch initial status
        this.fetchSystemStatus();
    }
    
    async fetchSystemStatus() {
        try {
            const response = await fetch('/api/system-status');
            const data = await response.json();
            this.updateSystemStatus(data);
        } catch (error) {
            console.error('Failed to fetch system status:', error);
        }
    }
    
    updateSystemStatus(status) {
        this.systemStatus = { ...this.systemStatus, ...status };
        
        // Update IBKR status - using ID selector to match base.html
        const ibkrStatus = document.querySelector('#ibkr-status');
        if (ibkrStatus) {
            const statusClass = this.systemStatus.ibkr === 'connected' ? 'status-connected' : 'status-disconnected';
            ibkrStatus.className = `status-indicator ${statusClass}`;
            ibkrStatus.innerHTML = `
                <div class="status-dot"></div>
                <span>IBKR ${this.systemStatus.ibkr === 'connected' ? 'Connected' : 'Disconnected'}</span>
            `;
        }
        
        // Update Data status - using ID selector to match base.html
        const dataStatus = document.querySelector('#data-status');
        if (dataStatus) {
            const hasData = this.systemStatus.ibkr === 'connected';
            const statusClass = hasData ? 'status-connected' : 'status-disconnected';
            dataStatus.className = `status-indicator ${statusClass}`;
            dataStatus.innerHTML = `
                <div class="status-dot"></div>
                <span>${hasData ? 'Live Data' : 'No Data'}</span>
            `;
        }
        
        // Update Scanner status - using ID selector to match base.html
        const scannerStatus = document.querySelector('#scanner-status');
        if (scannerStatus) {
            const isRunning = this.systemStatus.scanner === 'connected' || this.systemStatus.scanner === 'running';
            const statusClass = isRunning ? 'status-connected' : 'status-disconnected';
            scannerStatus.className = `status-indicator ${statusClass}`;
            scannerStatus.innerHTML = `
                <div class="status-dot"></div>
                <span>Scanner ${isRunning ? 'Connected' : 'Stopped'}</span>
            `;
        }
        
        console.log('System status updated:', this.systemStatus);
    }
    
    updateConnectionStatus(status) {
        const connectionStatus = document.querySelector('.connection-status');
        if (connectionStatus) {
            connectionStatus.className = `status-indicator connection-status ${status}`;
            connectionStatus.innerHTML = `
                <span class=\"status-dot\"></span>
                ${status.charAt(0).toUpperCase() + status.slice(1)}
            `;
        }
    }
    
    // Page-specific Features
    initializePageSpecificFeatures() {
        switch (this.currentPage) {
            case 'dashboard':
                this.initializeDashboard();
                break;
            case 'scanner':
                this.initializeScanner();
                break;
            case 'trade_builder':
                this.initializeTradeBuilder();
                break;
            case 'execution':
                this.initializeExecution();
                break;
            case 'alerts':
                this.initializeAlerts();
                break;
        }
    }
    
    initializeDashboard() {
        // Initialize dashboard-specific features
        this.initializeRefreshButtons();
        this.loadDashboardData();
        
        // Auto-refresh dashboard data every 30 seconds
        setInterval(() => {
            this.loadDashboardData();
        }, 30000);
    }
    
    initializeScanner() {
        // Initialize scanner controls
        const startScanBtn = document.getElementById('start-scan');
        const stopScanBtn = document.getElementById('stop-scan');
        const manualScanBtn = document.getElementById('manual-scan');
        
        if (startScanBtn) {
            startScanBtn.addEventListener('click', () => this.startScanner());
        }
        
        if (stopScanBtn) {
            stopScanBtn.addEventListener('click', () => this.stopScanner());
        }
        
        if (manualScanBtn) {
            manualScanBtn.addEventListener('click', () => this.runManualScan());
        }
        
        // Initialize filters
        this.initializeScannerFilters();
        
        // Load initial scan results
        this.loadScanResults();
    }
    
    initializeTradeBuilder() {
        // Initialize trade builder features
        this.initializeStockSearch();
        this.initializeOptionsAnalysis();
        this.initializeSimulation();
        
        // Initialize strategy tabs
        const strategyTabs = document.querySelectorAll('.strategy-tab');
        strategyTabs.forEach(tab => {
            tab.addEventListener('click', (e) => {
                e.preventDefault();
                this.switchStrategy(tab.dataset.strategy);
            });
        });
    }
    
    initializeExecution() {
        // Initialize execution features
        this.initializeOrderForm();
        this.loadAccountSummary();
        this.loadPositions();
        this.loadOrders();
        
        // Auto-refresh execution data every 10 seconds
        setInterval(() => {
            this.loadAccountSummary();
            this.loadPositions();
            this.loadOrders();
        }, 10000);
    }
    
    initializeAlerts() {
        // Initialize alerts features
        this.initializeAlertForm();
        this.loadAlerts();
        this.loadAlertTemplates();
        
        // Initialize Telegram test
        const testTelegramBtn = document.getElementById('test-telegram');
        if (testTelegramBtn) {
            testTelegramBtn.addEventListener('click', () => this.testTelegram());
        }
    }
    
    // Utility Functions
    initializeRefreshButtons() {
        const refreshButtons = document.querySelectorAll('.btn-refresh');
        refreshButtons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                const target = btn.dataset.target;
                this.refreshData(target);
            });
        });
    }
    
    async refreshData(target) {
        const btn = document.querySelector(`[data-target="${target}"]`);
        if (btn) {
            btn.disabled = true;
            btn.innerHTML = '<span class="spinner"></span> Refreshing...';
        }
        
        try {
            switch (target) {
                case 'dashboard':
                    await this.loadDashboardData();
                    break;
                case 'scanner':
                    await this.loadScanResults();
                    break;
                case 'account':
                    await this.loadAccountSummary();
                    break;
                case 'positions':
                    await this.loadPositions();
                    break;
                case 'orders':
                    await this.loadOrders();
                    break;
            }
        } catch (error) {
            console.error(`Failed to refresh ${target}:`, error);
            this.showNotification(`Failed to refresh ${target}`, 'error');
        } finally {
            if (btn) {
                btn.disabled = false;
                btn.innerHTML = '<i class="material-icons">refresh</i> Refresh';
            }
        }
    }
    
    // API Calls
    async loadDashboardData() {
        try {
            const [statusResponse, opportunitiesResponse] = await Promise.all([
                fetch('/api/system-status'),
                fetch('/api/golden-opportunities')
            ]);
            
            const statusData = await statusResponse.json();
            const opportunitiesData = await opportunitiesResponse.json();
            
            this.updateSystemStatus(statusData);
            this.updateGoldenOpportunities(opportunitiesData);
        } catch (error) {
            console.error('Failed to load dashboard data:', error);
        }
    }
    
    async loadScanResults() {
        try {
            const response = await fetch('/api/scan-results');
            const data = await response.json();
            this.updateScanResults(data);
        } catch (error) {
            console.error('Failed to load scan results:', error);
        }
    }
    
    async loadAccountSummary() {
        try {
            const response = await fetch('/api/account-summary');
            const data = await response.json();
            this.updateAccountSummary(data);
        } catch (error) {
            console.error('Failed to load account summary:', error);
        }
    }
    
    async loadPositions() {
        try {
            const response = await fetch('/api/positions');
            const data = await response.json();
            this.updatePositions(data);
        } catch (error) {
            console.error('Failed to load positions:', error);
        }
    }
    
    async loadOrders() {
        try {
            const response = await fetch('/api/orders');
            const data = await response.json();
            this.updateOrders(data);
        } catch (error) {
            console.error('Failed to load orders:', error);
        }
    }
    
    // Scanner Functions
    async startScanner() {
        try {
            const response = await fetch('/api/scanner/start', { method: 'POST' });
            const data = await response.json();
            
            if (data.success) {
                this.showNotification('Scanner started successfully', 'success');
                this.updateSystemStatus({ scanner: 'running' });
            } else {
                this.showNotification('Failed to start scanner: ' + data.error, 'error');
            }
        } catch (error) {
            console.error('Failed to start scanner:', error);
            this.showNotification('Failed to start scanner', 'error');
        }
    }
    
    async stopScanner() {
        try {
            const response = await fetch('/api/scanner/stop', { method: 'POST' });
            const data = await response.json();
            
            if (data.success) {
                this.showNotification('Scanner stopped successfully', 'success');
                this.updateSystemStatus({ scanner: 'stopped' });
            } else {
                this.showNotification('Failed to stop scanner: ' + data.error, 'error');
            }
        } catch (error) {
            console.error('Failed to stop scanner:', error);
            this.showNotification('Failed to stop scanner', 'error');
        }
    }
    
    async runManualScan() {
        const btn = document.getElementById('manual-scan');
        if (btn) {
            btn.disabled = true;
            btn.innerHTML = '<span class="spinner"></span> Scanning...';
        }
        
        try {
            const response = await fetch('/api/champion-scan', { method: 'POST' });
            const data = await response.json();
            
            if (data.success) {
                // Handle the correct response structure
                const opportunitiesCount = data.golden_opportunities ? data.golden_opportunities.length : 0;
                this.showNotification(`Scan completed: ${opportunitiesCount} golden opportunities found`, 'success');
                
                // Update scan results with proper data structure
                this.updateScanResults({
                    results: data.golden_opportunities || [],
                    universe_count: data.universe_count || 0,
                    champions_count: data.champions_count || 0,
                    scan_time: data.scan_time
                });
                
                // Update system status
                this.updateSystemStatus({ 
                    scanner: 'completed',
                    last_scan: data.scan_time,
                    opportunities_found: opportunitiesCount
                });
                
            } else {
                this.showNotification('Scan failed: ' + (data.error || 'Unknown error'), 'error');
            }
        } catch (error) {
            console.error('Failed to run manual scan:', error);
            this.showNotification('Failed to run manual scan: ' + error.message, 'error');
        } finally {
            if (btn) {
                btn.disabled = false;
                btn.innerHTML = '<i class="material-icons">search</i> Manual Scan';
            }
        }
    }
    
    // Update Functions
    updateScanResults(data) {
        const resultsContainer = document.getElementById('scan-results');
        if (!resultsContainer) return;
        
        if (data.results && data.results.length > 0) {
            const html = data.results.map(result => `
                <div class=\"scan-result-item\" data-symbol=\"${result.symbol}\">
                    <div class=\"result-header\">
                        <div class=\"result-symbol\">${result.symbol}</div>
                        <div class=\"result-score badge badge-${this.getScoreClass(result.score)}\">${result.score}</div>
                    </div>
                    <div class=\"result-details\">
                        <div class=\"result-pattern\">${result.pattern}</div>
                        <div class=\"result-price\">$${result.price.toFixed(2)}</div>
                        <div class=\"result-change ${result.change >= 0 ? 'positive' : 'negative'}\">
                            ${result.change >= 0 ? '+' : ''}${result.change.toFixed(2)}%
                        </div>
                    </div>
                    <div class=\"result-actions\">
                        <button class=\"btn btn-sm btn-primary\" onclick=\"tradingPlatform.analyzeOptions('${result.symbol}')\">
                            Analyze Options
                        </button>
                    </div>
                </div>
            `).join('');
            
            resultsContainer.innerHTML = html;
        } else {
            resultsContainer.innerHTML = `
                <div class=\"empty-state\">
                    <div class=\"empty-icon\">📊</div>
                    <div class=\"empty-title\">No opportunities found</div>
                    <div class=\"empty-message\">Run a scan to find golden opportunities</div>
                </div>
            `;
        }
        
        // Update results count
        const countElement = document.querySelector('.scanner-results-count');
        if (countElement) {
            countElement.textContent = data.results ? data.results.length : 0;
        }
    }
    
    updateGoldenOpportunities(data) {
        const container = document.getElementById('golden-opportunities');
        if (!container) return;
        
        if (data.opportunities && data.opportunities.length > 0) {
            const html = data.opportunities.map(opp => `
                <div class=\"opportunity-item\">
                    <div class=\"opportunity-header\">
                        <div class=\"opportunity-symbol\">${opp.symbol}</div>
                        <div class=\"opportunity-score badge badge-success\">${opp.score}</div>
                    </div>
                    <div class=\"opportunity-pattern\">${opp.pattern}</div>
                    <div class=\"opportunity-price\">$${opp.price.toFixed(2)}</div>
                </div>
            `).join('');
            
            container.innerHTML = html;
        } else {
            container.innerHTML = `
                <div class=\"empty-state\">
                    <div class=\"empty-message\">No golden opportunities at the moment</div>
                </div>
            `;
        }
    }
    
    updateMarketData(data) {
        // Update market data displays
        Object.keys(data).forEach(symbol => {
            const priceElement = document.querySelector(`[data-symbol=\"${symbol}\"] .price`);
            const changeElement = document.querySelector(`[data-symbol=\"${symbol}\"] .change`);
            
            if (priceElement) {
                priceElement.textContent = `$${data[symbol].price.toFixed(2)}`;
            }
            
            if (changeElement) {
                const change = data[symbol].change;
                changeElement.textContent = `${change >= 0 ? '+' : ''}${change.toFixed(2)}%`;
                changeElement.className = `change ${change >= 0 ? 'positive' : 'negative'}`;
            }
        });
    }
    
    updateAccountData(data) {
        this.updateAccountSummary(data.account);
        if (data.positions) this.updatePositions(data.positions);
        if (data.orders) this.updateOrders(data.orders);
    }
    
    updateAccountSummary(data) {
        const elements = {
            'total-value': data.totalValue,
            'buying-power': data.buyingPower,
            'day-pnl': data.dayPnL,
            'unrealized-pnl': data.unrealizedPnL
        };
        
        Object.keys(elements).forEach(id => {
            const element = document.getElementById(id);
            if (element && elements[id] !== undefined) {
                element.textContent = this.formatCurrency(elements[id]);
                
                // Add color classes for P&L
                if (id.includes('pnl')) {
                    element.className = elements[id] >= 0 ? 'positive' : 'negative';
                }
            }
        });
    }
    
    updatePositions(data) {
        const container = document.getElementById('positions-table');
        if (!container) return;
        
        if (data.positions && data.positions.length > 0) {
            const html = `
                <table class=\"table\">
                    <thead>
                        <tr>
                            <th>Symbol</th>
                            <th>Quantity</th>
                            <th>Avg Cost</th>
                            <th>Market Price</th>
                            <th>Market Value</th>
                            <th>Unrealized P&L</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${data.positions.map(pos => `
                            <tr>
                                <td class=\"symbol\">${pos.symbol}</td>
                                <td>${pos.quantity}</td>
                                <td>$${pos.avgCost.toFixed(2)}</td>
                                <td>$${pos.marketPrice.toFixed(2)}</td>
                                <td>$${pos.marketValue.toFixed(2)}</td>
                                <td class=\"${pos.unrealizedPnL >= 0 ? 'positive' : 'negative'}\">
                                    $${pos.unrealizedPnL.toFixed(2)}
                                </td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            `;
            
            container.innerHTML = html;
        } else {
            container.innerHTML = `
                <div class=\"empty-state\">
                    <div class=\"empty-message\">No positions</div>
                </div>
            `;
        }
    }
    
    updateOrders(data) {
        const container = document.getElementById('orders-table');
        if (!container) return;
        
        if (data.orders && data.orders.length > 0) {
            const html = `
                <table class=\"table\">
                    <thead>
                        <tr>
                            <th>Symbol</th>
                            <th>Side</th>
                            <th>Quantity</th>
                            <th>Order Type</th>
                            <th>Price</th>
                            <th>Status</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${data.orders.map(order => `
                            <tr>
                                <td class=\"symbol\">${order.symbol}</td>
                                <td class=\"${order.side.toLowerCase()}\">${order.side}</td>
                                <td>${order.quantity}</td>
                                <td>${order.orderType}</td>
                                <td>${order.price ? '$' + order.price.toFixed(2) : 'Market'}</td>
                                <td>
                                    <span class=\"badge badge-${this.getOrderStatusClass(order.status)}\">
                                        ${order.status}
                                    </span>
                                </td>
                                <td>
                                    ${order.status === 'Submitted' ? `
                                        <button class=\"btn btn-sm btn-error\" onclick=\"tradingPlatform.cancelOrder('${order.id}')\">
                                            Cancel
                                        </button>
                                    ` : ''}
                                </td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            `;
            
            container.innerHTML = html;
        } else {
            container.innerHTML = `
                <div class=\"empty-state\">
                    <div class=\"empty-message\">No active orders</div>
                </div>
            `;
        }
    }
    
    // Notification System
    showNotification(message, type = 'info', duration = 5000) {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class=\"notification-content\">
                <div class=\"notification-icon\">
                    ${this.getNotificationIcon(type)}
                </div>
                <div class=\"notification-message\">${message}</div>
                <button class=\"notification-close\" onclick=\"this.parentElement.parentElement.remove()\">
                    <i class=\"material-icons\">close</i>
                </button>
            </div>
        `;
        
        // Add to page
        let container = document.querySelector('.notifications-container');
        if (!container) {
            container = document.createElement('div');
            container.className = 'notifications-container';
            document.body.appendChild(container);
        }
        
        container.appendChild(notification);
        
        // Auto-remove after duration
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, duration);
        
        // Add CSS if not exists
        if (!document.querySelector('#notification-styles')) {
            const styles = document.createElement('style');
            styles.id = 'notification-styles';
            styles.textContent = `
                .notifications-container {
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    z-index: 10000;
                    display: flex;
                    flex-direction: column;
                    gap: 10px;
                }
                
                .notification {
                    background: var(--surface);
                    border: 1px solid var(--surface-variant);
                    border-radius: var(--radius-md);
                    box-shadow: var(--shadow-lg);
                    min-width: 300px;
                    max-width: 400px;
                    animation: slideIn 0.3s ease-out;
                }
                
                .notification-success {
                    border-left: 4px solid var(--success);
                }
                
                .notification-error {
                    border-left: 4px solid var(--error);
                }
                
                .notification-warning {
                    border-left: 4px solid var(--warning);
                }
                
                .notification-info {
                    border-left: 4px solid var(--info);
                }
                
                .notification-content {
                    display: flex;
                    align-items: flex-start;
                    gap: 12px;
                    padding: 16px;
                }
                
                .notification-icon {
                    font-size: 20px;
                    margin-top: 2px;
                }
                
                .notification-success .notification-icon {
                    color: var(--success);
                }
                
                .notification-error .notification-icon {
                    color: var(--error);
                }
                
                .notification-warning .notification-icon {
                    color: var(--warning);
                }
                
                .notification-info .notification-icon {
                    color: var(--info);
                }
                
                .notification-message {
                    flex: 1;
                    color: var(--on-surface);
                    font-size: 14px;
                    line-height: 1.4;
                }
                
                .notification-close {
                    background: none;
                    border: none;
                    color: var(--on-surface-variant);
                    cursor: pointer;
                    padding: 0;
                    font-size: 18px;
                    transition: color 0.2s;
                }
                
                .notification-close:hover {
                    color: var(--on-surface);
                }
                
                @keyframes slideIn {
                    from {
                        transform: translateX(100%);
                        opacity: 0;
                    }
                    to {
                        transform: translateX(0);
                        opacity: 1;
                    }
                }
                
                @media (max-width: 480px) {
                    .notifications-container {
                        left: 10px;
                        right: 10px;
                        top: 10px;
                    }
                    
                    .notification {
                        min-width: auto;
                        max-width: none;
                    }
                }
            `;
            document.head.appendChild(styles);
        }
    }
    
    getNotificationIcon(type) {
        const icons = {
            success: '✅',
            error: '❌',
            warning: '⚠️',
            info: 'ℹ️'
        };
        return icons[type] || icons.info;
    }
    
    showGoldenOpportunityAlert(data) {
        this.showNotification(
            `Golden Opportunity: ${data.symbol} - ${data.pattern} (Score: ${data.score})`,
            'success',
            10000
        );
        
        // Also update the golden opportunities section
        this.loadDashboardData();
    }
    
    // Utility Functions
    getScoreClass(score) {
        if (score >= 90) return 'success';
        if (score >= 70) return 'warning';
        return 'error';
    }
    
    getOrderStatusClass(status) {
        const statusMap = {
            'Submitted': 'warning',
            'Filled': 'success',
            'Cancelled': 'error',
            'Pending': 'info'
        };
        return statusMap[status] || 'info';
    }
    
    formatCurrency(value) {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD'
        }).format(value);
    }
    
    formatPercent(value) {
        return new Intl.NumberFormat('en-US', {
            style: 'percent',
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        }).format(value / 100);
    }
    
    // Heartbeat to keep connection alive
    startHeartbeat() {
        setInterval(() => {
            if (this.socket && this.isConnected) {
                this.socket.emit('ping');
            }
        }, 30000); // 30 seconds
    }
    
    // Public methods for external calls
    async analyzeOptions(symbol) {
        window.location.href = `/trade-builder?symbol=${symbol}`;
    }
    
    async cancelOrder(orderId) {
        try {
            const response = await fetch(`/api/orders/${orderId}/cancel`, { method: 'POST' });
            const data = await response.json();
            
            if (data.success) {
                this.showNotification('Order cancelled successfully', 'success');
                this.loadOrders();
            } else {
                this.showNotification('Failed to cancel order: ' + data.error, 'error');
            }
        } catch (error) {
            console.error('Failed to cancel order:', error);
            this.showNotification('Failed to cancel order', 'error');
        }
    }
    
    async testTelegram() {
        try {
            const response = await fetch('/api/telegram/test', { method: 'POST' });
            const data = await response.json();
            
            if (data.success) {
                this.showNotification('Test message sent successfully!', 'success');
            } else {
                this.showNotification('Failed to send test message: ' + data.error, 'error');
            }
        } catch (error) {
            console.error('Failed to test Telegram:', error);
            this.showNotification('Failed to test Telegram', 'error');
        }
    }
}

// Initialize the trading platform when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.tradingPlatform = new TradingPlatform();
});

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TradingPlatform;
}

