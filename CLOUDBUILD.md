# Google Cloud Build Configuration Guide

This Cloud Build configuration deploys the FastAPI application to Google Cloud Run.

## Prerequisites

1. Google Cloud project with billing enabled
2. Cloud Build API enabled
3. Cloud Run API enabled
4. Artifact Registry API enabled

## Setup

### 1. Enable Required APIs

```bash
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com
```

### 2. Create Artifact Registry Repository

```bash
gcloud artifacts repositories create cloud-run-source-deploy \
  --repository-format=docker \
  --location=us-central1 \
  --description="Docker repository for Cloud Run deployments"
```

### 3. Create Secret Manager Secret for Database URL

```bash
echo -n "postgresql://user:password@host:5432/dbname" | \
  gcloud secrets create DATABASE_URL --data-file=-
```

Grant Cloud Run access to the secret:
```bash
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
gcloud secrets add-iam-policy-binding DATABASE_URL \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

## Deployment

### Automatic Deployment (Trigger)

1. Connect your repository to Cloud Build:
   - Go to Cloud Build > Triggers
   - Click "Create Trigger"
   - Connect your repository (GitHub, Bitbucket, etc.)
   - Set the configuration file path to `cloudbuild.yaml`
   - Set the substitution variables:
     - `_SERVICE_NAME`: self-improving-engine-api
     - `_DEPLOY_REGION`: us-central1
     - `_AR_HOSTNAME`: us-central1-docker.pkg.dev
   - Save and enable the trigger

2. Push to your repository:
   ```bash
   git push origin main
   ```

### Manual Deployment

```bash
gcloud builds submit --config=cloudbuild.yaml \
  --substitutions=_SERVICE_NAME=self-improving-engine-api,_DEPLOY_REGION=us-central1,_AR_HOSTNAME=us-central1-docker.pkg.dev
```

## Configuration

### Resource Allocation

The current configuration uses:
- **Memory**: 512Mi
- **CPU**: 1
- **Timeout**: 300 seconds
- **Concurrency**: 80
- **Min Instances**: 0 (scales to zero)
- **Max Instances**: 10

To adjust these values, modify the `args` section in the `Deploy` step.

### Environment Variables

The build config sets:
- `DEBUG=False` - Production mode
- `LOG_LEVEL=INFO` - Logging level
- `PORT=8080` - Application port

### Secrets

The `DATABASE_URL` is pulled from Secret Manager. To update it:
```bash
echo -n "new-database-url" | gcloud secrets versions add DATABASE_URL --data-file=-
```

## Monitoring

View logs:
```bash
gcloud run services logs read self-improving-engine-api --region us-central1
```

View service details:
```bash
gcloud run services describe self-improving-engine-api --region us-central1
```

## Troubleshooting

### Build Failures

1. Check build logs:
   ```bash
   gcloud builds list --limit=5
   gcloud builds log <BUILD_ID>
   ```

2. Test Docker build locally:
   ```bash
   docker build -t test-image .
   ```

### Deployment Failures

1. Check service logs:
   ```bash
   gcloud run services logs read self-improving-engine-api --region us-central1 --limit=50
   ```

2. Verify secrets are accessible:
   ```bash
   gcloud secrets versions access latest --secret=DATABASE_URL
   ```

### Database Connection Issues

1. Ensure Cloud SQL instance is running
2. Verify connection string format
3. Check network connectivity from Cloud Run to database

## Cost Optimization

- **Min instances = 0**: Service scales to zero when idle
- **Memory**: Start with 512Mi, increase if needed
- **CPU**: Scale based on actual usage
- **Timeout**: Set based on expected request duration

## Security

- Secrets are managed through Secret Manager
- Non-root user in container
- HTTPS enforced by Cloud Run
- IAM-based access control

