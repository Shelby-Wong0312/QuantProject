#!/bin/bash
# Automated Deployment Script
# Cloud DevOps Agent

set -e

# Configuration
PROJECT_NAME="quant-trading-system"
DEPLOY_ENV=${1:-production}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/opt/backups"
PROJECT_DIR="/opt/quanttrading"
LOG_FILE="/var/log/deploy_${TIMESTAMP}.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a $LOG_FILE
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a $LOG_FILE
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a $LOG_FILE
}

# Pre-deployment checks
pre_deploy_checks() {
    log "Running pre-deployment checks..."
    
    # Check disk space
    DISK_USAGE=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')
    if [ $DISK_USAGE -gt 80 ]; then
        warning "Disk usage is high: ${DISK_USAGE}%"
    fi
    
    # Check if Docker is running
    if ! docker info > /dev/null 2>&1; then
        error "Docker is not running"
    fi
    
    # Check if docker-compose is installed
    if ! command -v docker-compose &> /dev/null; then
        error "docker-compose is not installed"
    fi
    
    log "Pre-deployment checks passed"
}

# Backup current deployment
backup_current() {
    log "Creating backup of current deployment..."
    
    mkdir -p $BACKUP_DIR
    
    # Backup database
    if [ -f "$PROJECT_DIR/data/quant_trading.db" ]; then
        cp "$PROJECT_DIR/data/quant_trading.db" "$BACKUP_DIR/quant_trading_${TIMESTAMP}.db"
        log "Database backed up"
    fi
    
    # Backup config files
    if [ -d "$PROJECT_DIR/config" ]; then
        tar -czf "$BACKUP_DIR/config_${TIMESTAMP}.tar.gz" -C "$PROJECT_DIR" config/
        log "Configuration backed up"
    fi
    
    # Backup logs
    if [ -d "$PROJECT_DIR/logs" ]; then
        tar -czf "$BACKUP_DIR/logs_${TIMESTAMP}.tar.gz" -C "$PROJECT_DIR" logs/
        log "Logs backed up"
    fi
}

# Pull latest code
pull_latest() {
    log "Pulling latest code..."
    
    cd $PROJECT_DIR
    
    # Stash any local changes
    git stash
    
    # Pull latest from main branch
    git checkout main
    git pull origin main
    
    log "Code updated to latest version"
}

# Build and deploy
deploy() {
    log "Starting deployment..."
    
    cd $PROJECT_DIR
    
    # Load environment variables
    if [ -f ".env.$DEPLOY_ENV" ]; then
        export $(cat .env.$DEPLOY_ENV | xargs)
        log "Environment variables loaded for $DEPLOY_ENV"
    fi
    
    # Build images
    log "Building Docker images..."
    docker-compose build --no-cache
    
    # Stop current containers
    log "Stopping current containers..."
    docker-compose down
    
    # Start new containers
    log "Starting new containers..."
    docker-compose up -d
    
    # Wait for services to be healthy
    log "Waiting for services to be healthy..."
    sleep 30
    
    # Run database migrations if needed
    if [ -f "scripts/migrate.py" ]; then
        log "Running database migrations..."
        docker-compose exec trading-system python scripts/migrate.py
    fi
    
    log "Deployment completed"
}

# Health checks
health_check() {
    log "Running health checks..."
    
    # Check if main service is running
    if docker-compose ps | grep -q "trading-system.*Up"; then
        log "Trading system is running"
    else
        error "Trading system is not running"
    fi
    
    # Check API health endpoint
    HEALTH_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)
    if [ $HEALTH_STATUS -eq 200 ]; then
        log "Health check passed"
    else
        error "Health check failed with status: $HEALTH_STATUS"
    fi
    
    # Check database connection
    docker-compose exec trading-system python -c "
import sqlite3
conn = sqlite3.connect('/app/data/quant_trading.db')
print('Database connection successful')
conn.close()
    " || error "Database connection failed"
    
    # Check monitoring
    if curl -s http://localhost:9090/-/healthy > /dev/null; then
        log "Prometheus is healthy"
    else
        warning "Prometheus health check failed"
    fi
    
    if curl -s http://localhost:3000/api/health > /dev/null; then
        log "Grafana is healthy"
    else
        warning "Grafana health check failed"
    fi
}

# Rollback function
rollback() {
    error "Deployment failed, rolling back..."
    
    cd $PROJECT_DIR
    
    # Stop current containers
    docker-compose down
    
    # Restore from backup
    if [ -f "$BACKUP_DIR/quant_trading_${TIMESTAMP}.db" ]; then
        cp "$BACKUP_DIR/quant_trading_${TIMESTAMP}.db" "$PROJECT_DIR/data/quant_trading.db"
        log "Database restored from backup"
    fi
    
    # Restart with previous version
    git checkout HEAD~1
    docker-compose up -d
    
    log "Rollback completed"
}

# Cleanup old backups
cleanup_backups() {
    log "Cleaning up old backups..."
    
    # Keep only last 7 days of backups
    find $BACKUP_DIR -type f -mtime +7 -delete
    
    log "Old backups cleaned"
}

# Send notification
send_notification() {
    local status=$1
    local message=$2
    
    # Send to Slack webhook if configured
    if [ ! -z "$SLACK_WEBHOOK" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"Deployment $status: $message\"}" \
            $SLACK_WEBHOOK
    fi
    
    # Log to file
    echo "[$status] $message" >> /var/log/deployments.log
}

# Main deployment flow
main() {
    log "========================================="
    log "Starting deployment for $DEPLOY_ENV environment"
    log "========================================="
    
    # Set trap for rollback on error
    trap rollback ERR
    
    # Run deployment steps
    pre_deploy_checks
    backup_current
    pull_latest
    deploy
    health_check
    cleanup_backups
    
    # Remove trap after successful deployment
    trap - ERR
    
    log "========================================="
    log "Deployment completed successfully!"
    log "========================================="
    
    send_notification "SUCCESS" "Deployment to $DEPLOY_ENV completed at $TIMESTAMP"
}

# Run main function
main "$@"