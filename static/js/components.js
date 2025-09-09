// UI Components and Interactive Elements for Enhanced Trading Platform

class UIComponents {
    constructor() {
        this.init();
    }
    
    init() {
        this.initializeButtons();
        this.initializeModals();
        this.initializeTooltips();
        this.initializeTabs();
        this.initializeDropdowns();
        this.initializeNotifications();
        
        console.log('UI Components initialized');
    }
    
    // Button Management
    initializeButtons() {
        // Scan buttons
        const scanButtons = document.querySelectorAll('[data-action="scan"]');
        scanButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                e.preventDefault();
                this.handleScanAction(button);
            });
        });
        
        // Refresh buttons
        const refreshButtons = document.querySelectorAll('[data-action="refresh"]');
        refreshButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                e.preventDefault();
                this.handleRefreshAction(button);
            });
        });
        
        // Test connection buttons
        const testButtons = document.querySelectorAll('[data-action="test-connection"]');
        testButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                e.preventDefault();
                this.handleTestConnection(button);
            });
        });
        
        // Generic action buttons
        const actionButtons = document.querySelectorAll('[data-action]');
        actionButtons.forEach(button => {
            if (!button.hasAttribute('data-initialized')) {
                button.addEventListener('click', (e) => {
                    this.handleGenericAction(button, e);
                });
                button.setAttribute('data-initialized', 'true');
            }
        });
    }
    
    handleScanAction(button) {
        const scanType = button.getAttribute('data-scan-type') || 'manual';
        
        // Show loading state
        this.setButtonLoading(button, true);
        
        // Simulate scan action
        setTimeout(() => {
            this.setButtonLoading(button, false);
            this.showNotification('Scan completed successfully', 'success');
            
            // Trigger scan update event
            window.dispatchEvent(new CustomEvent('scan-completed', {
                detail: { type: scanType }
            }));
        }, 2000);
    }
    
    handleRefreshAction(button) {
        const refreshType = button.getAttribute('data-refresh-type') || 'general';
        
        // Show loading state
        this.setButtonLoading(button, true);
        
        // Simulate refresh action
        setTimeout(() => {
            this.setButtonLoading(button, false);
            this.showNotification('Data refreshed', 'info');
            
            // Trigger refresh event
            window.dispatchEvent(new CustomEvent('data-refreshed', {
                detail: { type: refreshType }
            }));
        }, 1000);
    }
    
    handleTestConnection(button) {
        const connectionType = button.getAttribute('data-connection-type') || 'general';
        
        // Show loading state
        this.setButtonLoading(button, true);
        
        // Simulate connection test
        setTimeout(() => {
            this.setButtonLoading(button, false);
            
            // Simulate random success/failure
            const success = Math.random() > 0.3;
            if (success) {
                this.showNotification('Connection test successful', 'success');
            } else {
                this.showNotification('Connection test failed', 'error');
            }
            
            // Trigger connection test event
            window.dispatchEvent(new CustomEvent('connection-tested', {
                detail: { type: connectionType, success }
            }));
        }, 1500);
    }
    
    handleGenericAction(button, event) {
        const action = button.getAttribute('data-action');
        const target = button.getAttribute('data-target');
        
        switch (action) {
            case 'toggle':
                this.handleToggleAction(button, target);
                break;
            case 'submit':
                this.handleSubmitAction(button, target);
                break;
            case 'clear':
                this.handleClearAction(button, target);
                break;
            case 'export':
                this.handleExportAction(button, target);
                break;
            default:
                console.log(`Unhandled action: ${action}`);
        }
    }
    
    handleToggleAction(button, target) {
        const targetElement = document.querySelector(target);
        if (targetElement) {
            targetElement.classList.toggle('hidden');
            targetElement.classList.toggle('visible');
            
            // Update button text
            const isVisible = targetElement.classList.contains('visible');
            const showText = button.getAttribute('data-show-text') || 'Show';
            const hideText = button.getAttribute('data-hide-text') || 'Hide';
            button.textContent = isVisible ? hideText : showText;
        }
    }
    
    handleSubmitAction(button, target) {
        const form = document.querySelector(target);
        if (form) {
            this.setButtonLoading(button, true);
            
            // Simulate form submission
            setTimeout(() => {
                this.setButtonLoading(button, false);
                this.showNotification('Form submitted successfully', 'success');
            }, 1000);
        }
    }
    
    handleClearAction(button, target) {
        const form = document.querySelector(target);
        if (form) {
            form.reset();
            this.showNotification('Form cleared', 'info');
        }
    }
    
    handleExportAction(button, target) {
        this.setButtonLoading(button, true);
        
        // Simulate export
        setTimeout(() => {
            this.setButtonLoading(button, false);
            this.showNotification('Export completed', 'success');
        }, 1500);
    }
    
    setButtonLoading(button, loading) {
        if (loading) {
            button.disabled = true;
            button.classList.add('loading');
            
            // Store original content
            button.setAttribute('data-original-content', button.innerHTML);
            
            // Show loading spinner
            button.innerHTML = '<span class="spinner"></span> Loading...';
        } else {
            button.disabled = false;
            button.classList.remove('loading');
            
            // Restore original content
            const originalContent = button.getAttribute('data-original-content');
            if (originalContent) {
                button.innerHTML = originalContent;
            }
        }
    }
    
    // Modal Management
    initializeModals() {
        const modalTriggers = document.querySelectorAll('[data-modal]');
        const modals = document.querySelectorAll('.modal');
        const modalCloses = document.querySelectorAll('.modal-close');
        
        modalTriggers.forEach(trigger => {
            trigger.addEventListener('click', (e) => {
                e.preventDefault();
                const modalId = trigger.getAttribute('data-modal');
                this.openModal(modalId);
            });
        });
        
        modalCloses.forEach(close => {
            close.addEventListener('click', (e) => {
                e.preventDefault();
                this.closeModal(close.closest('.modal'));
            });
        });
        
        // Close modal on backdrop click
        modals.forEach(modal => {
            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    this.closeModal(modal);
                }
            });
        });
        
        // Close modal on Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                const openModal = document.querySelector('.modal.open');
                if (openModal) {
                    this.closeModal(openModal);
                }
            }
        });
    }
    
    openModal(modalId) {
        const modal = document.getElementById(modalId);
        if (modal) {
            modal.classList.add('open');
            document.body.classList.add('modal-open');
        }
    }
    
    closeModal(modal) {
        if (modal) {
            modal.classList.remove('open');
            document.body.classList.remove('modal-open');
        }
    }
    
    // Tooltip Management
    initializeTooltips() {
        const tooltipElements = document.querySelectorAll('[data-tooltip]');
        
        tooltipElements.forEach(element => {
            element.addEventListener('mouseenter', (e) => {
                this.showTooltip(e.target);
            });
            
            element.addEventListener('mouseleave', (e) => {
                this.hideTooltip();
            });
        });
    }
    
    showTooltip(element) {
        const tooltipText = element.getAttribute('data-tooltip');
        const tooltip = document.createElement('div');
        tooltip.className = 'tooltip';
        tooltip.textContent = tooltipText;
        tooltip.id = 'active-tooltip';
        
        document.body.appendChild(tooltip);
        
        // Position tooltip
        const rect = element.getBoundingClientRect();
        tooltip.style.left = rect.left + (rect.width / 2) - (tooltip.offsetWidth / 2) + 'px';
        tooltip.style.top = rect.top - tooltip.offsetHeight - 10 + 'px';
    }
    
    hideTooltip() {
        const tooltip = document.getElementById('active-tooltip');
        if (tooltip) {
            tooltip.remove();
        }
    }
    
    // Tab Management
    initializeTabs() {
        const tabButtons = document.querySelectorAll('.tab-button');
        const tabContents = document.querySelectorAll('.tab-content');
        
        tabButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                e.preventDefault();
                const tabId = button.getAttribute('data-tab');
                this.switchTab(tabId, button);
            });
        });
    }
    
    switchTab(tabId, activeButton) {
        // Remove active class from all tabs
        document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
        
        // Add active class to selected tab
        activeButton.classList.add('active');
        const targetContent = document.getElementById(tabId);
        if (targetContent) {
            targetContent.classList.add('active');
        }
    }
    
    // Dropdown Management
    initializeDropdowns() {
        const dropdownTriggers = document.querySelectorAll('.dropdown-trigger');
        
        dropdownTriggers.forEach(trigger => {
            trigger.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                
                const dropdown = trigger.nextElementSibling;
                if (dropdown && dropdown.classList.contains('dropdown-menu')) {
                    this.toggleDropdown(dropdown);
                }
            });
        });
        
        // Close dropdowns when clicking outside
        document.addEventListener('click', () => {
            document.querySelectorAll('.dropdown-menu.open').forEach(dropdown => {
                dropdown.classList.remove('open');
            });
        });
    }
    
    toggleDropdown(dropdown) {
        // Close other dropdowns
        document.querySelectorAll('.dropdown-menu.open').forEach(other => {
            if (other !== dropdown) {
                other.classList.remove('open');
            }
        });
        
        // Toggle current dropdown
        dropdown.classList.toggle('open');
    }
    
    // Notification System
    initializeNotifications() {
        // Create notification container if it doesn't exist
        if (!document.getElementById('notification-container')) {
            const container = document.createElement('div');
            container.id = 'notification-container';
            container.className = 'notification-container';
            document.body.appendChild(container);
        }
    }
    
    showNotification(message, type = 'info', duration = 5000) {
        const container = document.getElementById('notification-container');
        if (!container) return;
        
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <span class="notification-message">${message}</span>
                <button class="notification-close">&times;</button>
            </div>
        `;
        
        // Add close functionality
        const closeBtn = notification.querySelector('.notification-close');
        closeBtn.addEventListener('click', () => {
            this.removeNotification(notification);
        });
        
        // Add to container
        container.appendChild(notification);
        
        // Auto-remove after duration
        if (duration > 0) {
            setTimeout(() => {
                this.removeNotification(notification);
            }, duration);
        }
        
        // Animate in
        setTimeout(() => {
            notification.classList.add('show');
        }, 100);
    }
    
    removeNotification(notification) {
        notification.classList.add('hide');
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }
    
    // Utility Methods
    showLoading(element) {
        if (element) {
            element.classList.add('loading');
        }
    }
    
    hideLoading(element) {
        if (element) {
            element.classList.remove('loading');
        }
    }
    
    updateProgress(progressBar, percentage) {
        if (progressBar) {
            const fill = progressBar.querySelector('.progress-fill');
            if (fill) {
                fill.style.width = `${percentage}%`;
            }
        }
    }
}

// Initialize components when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.uiComponents = new UIComponents();
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = UIComponents;
}

