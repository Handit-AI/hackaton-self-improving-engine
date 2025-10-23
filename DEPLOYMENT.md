# Google Cloud Deployment Guide

This guide explains how to deploy the FastAPI application to Google Cloud Run.

## Prerequisites

1. Google Cloud SDK installed and configured
2. A Google Cloud project with billing enabled
3. Cloud Run API enabled
4. Container Registry API enabled

## Environment Setup

1. Set your project ID:
```bash
export PROJECT_ID=your-project-id
gcloud config set project $PROJECT_ID
```

2. Enable required APIs:
```bash
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

## Database Setup

### Option 1: Cloud SQL (Recommended for Production)

1. Create a Cloud SQL instance:
```bash
gcloud sql instances create postgres-instance \
  --database-version=POSTGRES_15 \
  --tier=db-f1-micro \
  --region=us-central1
```

2. Create a database:
```bash
gcloud sql databases create dbname --instance=postgres-instance
```

3. Create a user:
```bash
gcloud sql users create dbuser --instance=postgres-instance --password=your-password
```

4. Get the connection name:
```bash
gcloud sql instances describe postgres-instance --format="value(connectionName)"
```

5. Update `cloudbuild.yaml` with the connection name:
```yaml
- '--add-cloudsql-instances'
- 'PROJECT_ID:REGION:INSTANCE_NAME'
```

### Option 2: Local PostgreSQL

For development, you can use a local PostgreSQL instance or a managed service.

## Configuration

1. Create a `.env` file for local development:
```bash
cp env.example .env
```

2. For Cloud Run, set environment variables in the deployment:
```bash
gcloud run services update self-improving-engine-api \
  --set-env-vars="DATABASE_URL=postgresql://user:password@host:5432/dbname"
```

## Deployment Methods

### Method 1: Using deploy.sh script

```bash
chmod +x deploy.sh
./deploy.sh
```

### Method 2: Using Cloud Build directly

```bash
gcloud builds submit --config=cloudbuild.yaml \
  --substitutions=_SERVICE_NAME=self-improving-engine-api,_REGION=us-central1
```

### Method 3: Manual deployment

1. Build the Docker image:
```bash
docker build -t gcr.io/$PROJECT_ID/self-improving-engine-api .
```

2. Push to Container Registry:
```bash
docker push gcr.io/$PROJECT_ID/self-improving-engine-api
```

3. Deploy to Cloud Run:
```bash
gcloud run deploy self-improving-engine-api \
  --image gcr.io/$PROJECT_ID/self-improving-engine-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080
```

## Update Environment Variables

```bash
gcloud run services update self-improving-engine-api \
  --region us-central1 \
  --set-env-vars="DATABASE_URL=postgresql://user:pass@host:5432/db,DEBUG=False"
```

## View Logs

```bash
gcloud run services logs read self-improving-engine-api --region us-central1
```

## Monitoring

Access the Cloud Run console:
```
https://console.cloud.google.com/run
```

## Troubleshooting

### Database connection issues

1. Check if Cloud SQL instance is running:
```bash
gcloud sql instances describe postgres-instance
```

2. Verify connection from Cloud Run:
```bash
gcloud run services describe self-improving-engine-api --region us-central1
```

### Build failures

1. Check Cloud Build logs:
```bash
gcloud builds list --limit=5
gcloud builds log <BUILD_ID>
```

### Container startup issues

1. Check container logs:
```bash
gcloud run services logs read self-improving-engine-api --region us-central1 --limit=50
```

## Security Best Practices

1. Use Secret Manager for sensitive data:
```bash
echo -n "your-secret-value" | gcloud secrets create database-url --data-file=-
```

2. Grant Cloud Run access to secrets:
```bash
gcloud secrets add-iam-policy-binding database-url \
  --member="serviceAccount:YOUR_PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

3. Update service to use secret:
```bash
gcloud run services update self-improving-engine-api \
  --update-secrets="DATABASE_URL=database-url:latest"
```

## Cost Optimization

- Use `min-instances: 0` for development to avoid charges when idle
- Set appropriate `max-instances` based on expected traffic
- Consider using Cloud SQL tier appropriate for your workload
- Monitor usage in Cloud Console

## Rollback

If needed, rollback to a previous revision:
```bash
gcloud run services update-traffic self-improving-engine-api \
  --region us-central1 \
  --to-revisions=REVISION_NAME=100
```

