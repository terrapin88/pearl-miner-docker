#!/bin/bash
# ============================================================
# Block Watcher — monitors pearl-gateway/pearld output for block signals
# and fires a webhook (Discord/Telegram) with machine attribution.
# 
# Also persists ALL block events to /app/blocks.log for historical review.
# ============================================================

WEBHOOK_URL="${BLOCK_WEBHOOK_URL:-}"
MACHINE_ID="${VASTAI_MACHINE_ID:-unknown}"
INSTANCE_ID="${VASTAI_INSTANCE_ID:-unknown}"
GPU_TYPE="${GPU_TYPE:-unknown}"
GPU_COUNT="${GPU_COUNT:-1}"
LOG_FILE="/app/blocks.log"

# Identify this machine
HOSTNAME_TAG=$(hostname 2>/dev/null || echo "unknown")

echo "👁️  Block Watcher started"
echo "   Machine: $MACHINE_ID | Instance: $INSTANCE_ID | GPU: ${GPU_COUNT}x ${GPU_TYPE}"
echo "   Webhook: ${WEBHOOK_URL:+configured}${WEBHOOK_URL:-NOT SET}"
echo "   Log file: $LOG_FILE"

# Function to send webhook notification
send_notification() {
    local message="$1"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    # Always log locally
    echo "[$timestamp] $message" >> "$LOG_FILE"
    echo "🎉 BLOCK FOUND — $message"
    
    # Send to webhook if configured
    if [ -n "$WEBHOOK_URL" ]; then
        # Detect webhook type by URL pattern
        if echo "$WEBHOOK_URL" | grep -q "discord"; then
            # Discord webhook format
            curl -s -X POST "$WEBHOOK_URL" \
                -H "Content-Type: application/json" \
                -d "{
                    \"content\": \"⛏️ **BLOCK FOUND!**\",
                    \"embeds\": [{
                        \"title\": \"Pearl Block Mined!\",
                        \"color\": 65280,
                        \"fields\": [
                            {\"name\": \"Machine\", \"value\": \"$MACHINE_ID\", \"inline\": true},
                            {\"name\": \"Instance\", \"value\": \"$INSTANCE_ID\", \"inline\": true},
                            {\"name\": \"GPU\", \"value\": \"${GPU_COUNT}x ${GPU_TYPE}\", \"inline\": true},
                            {\"name\": \"Hostname\", \"value\": \"$HOSTNAME_TAG\", \"inline\": true},
                            {\"name\": \"Time (UTC)\", \"value\": \"$timestamp\", \"inline\": true}
                        ],
                        \"description\": \"$message\"
                    }]
                }" 2>/dev/null && echo "   ✅ Discord webhook sent" || echo "   ⚠️  Discord webhook failed"
        else
            # Generic webhook (Telegram bot, custom endpoint, etc.)
            curl -s -X POST "$WEBHOOK_URL" \
                -H "Content-Type: application/json" \
                -d "{
                    \"event\": \"block_found\",
                    \"machine_id\": \"$MACHINE_ID\",
                    \"instance_id\": \"$INSTANCE_ID\",
                    \"gpu_type\": \"$GPU_TYPE\",
                    \"gpu_count\": $GPU_COUNT,
                    \"hostname\": \"$HOSTNAME_TAG\",
                    \"timestamp\": \"$timestamp\",
                    \"message\": \"$message\"
                }" 2>/dev/null && echo "   ✅ Webhook sent" || echo "   ⚠️  Webhook failed"
        fi
    fi
}

# Monitor: read from a named pipe that entrypoint tees into
# The entrypoint will pipe all output through here
WATCH_PIPE="/tmp/pearl_output"

if [ -p "$WATCH_PIPE" ]; then
    # Pipe mode — read from named pipe
    while IFS= read -r line; do
        if echo "$line" | grep -qi "Block found"; then
            send_notification "Block signal: $line"
        fi
    done < "$WATCH_PIPE"
else
    # Fallback: tail the pearld/gateway logs if available
    echo "   (No pipe detected, will monitor via process stdout redirect)"
    # In this mode, the entrypoint must redirect output to a file we can tail
    PEARL_LOG="/app/pearl_combined.log"
    
    # Wait for log file to appear
    for i in $(seq 1 120); do
        [ -f "$PEARL_LOG" ] && break
        sleep 1
    done
    
    if [ -f "$PEARL_LOG" ]; then
        tail -F "$PEARL_LOG" 2>/dev/null | while IFS= read -r line; do
            if echo "$line" | grep -qi "Block found"; then
                send_notification "Block signal: $line"
            fi
        done
    else
        echo "⚠️  No log source found. Block watcher idle."
        sleep infinity
    fi
fi
