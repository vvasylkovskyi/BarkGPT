#!/bin/bash

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Error: .env file not found!"
    exit 1
fi

# Read .env file and create JSON string
JSON_STRING="{"
FIRST=true

while IFS='=' read -r key value; do
    # Skip empty lines and comments
    [[ -z "$key" || "$key" =~ ^# ]] && continue

    # Remove quotes from value
    value=$(echo "$value" | sed 's/^"//;s/"$//')

    # Add comma if not first item
    if [ "$FIRST" = true ]; then
        FIRST=false
    else
        JSON_STRING="$JSON_STRING,"
    fi

    # Add key-value pair to JSON
    JSON_STRING="$JSON_STRING\"$key\": \"$value\""
done < .env

JSON_STRING="$JSON_STRING}"

# Check if secret exists
if aws secretsmanager describe-secret --secret-id "my_app/v1/credentials" >/dev/null 2>&1; then
    # Update existing secret
    aws secretsmanager update-secret \
        --secret-id "my_app/v1/credentials" \
        --secret-string "$JSON_STRING"
    echo "Secret updated successfully!"
else
    # Create new secret
    aws secretsmanager create-secret \
        --name "my_app/v1/credentials" \
        --description "Application credentials" \
        --secret-string "$JSON_STRING"
    echo "Secret created successfully!"
fi