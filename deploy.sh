#!/bin/bash
set -e

PROJECT_ID="${PROJECT_ID:-your-project-id}"
REGION="${REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-self-improving-engine-api}"

echo "Building and deploying to Google Cloud..."
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Service: $SERVICE_NAME"

# Build and deploy using Cloud Build
gcloud builds submit --config=cloudbuild.yaml \
  --substitutions=_SERVICE_NAME=$SERVICE_NAME,_REGION=$REGION

echo "Deployment completed successfully!"
echo "Service URL: https://$SERVICE_NAME-$PROJECT_ID.a.run.app"

