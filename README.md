# Self-Improving Engine API

A FastAPI application with PostgreSQL database support, following best practices.

## Features

- FastAPI with async/await support
- PostgreSQL database connection with SQLAlchemy
- Alembic for database migrations
- Environment-based configuration
- Health check endpoint
- Docker containerization
- Google Cloud Run deployment ready
- Proper logging and error handling
- Type hints throughout

## Prerequisites

- Python 3.9+
- PostgreSQL 12+
- pip

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd hackaton-self-improving-engine
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp env.example .env
# Edit .env with your database credentials
```

**Important**: Make sure to create a `.env` file with your database URL before running the application.

## Configuration

Edit the `.env` file with your database configuration:

```env
DATABASE_URL=postgresql://user:password@localhost:5432/dbname
DEBUG=True
LOG_LEVEL=INFO
HOST=0.0.0.0
PORT=8000
```

## Database Setup

1. Create a PostgreSQL database:
```bash
createdb dbname
```

2. Run migrations:
```bash
alembic upgrade head
```

## Running the Application

### Development Mode

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --reload
```

### Production Mode

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Docker

Build the Docker image:
```bash
docker build -t self-improving-engine-api .
```

Run the container:
```bash
docker run -p 8080:8080 --env-file .env self-improving-engine-api
```

## Google Cloud Deployment

### Cloud Build (Recommended)

For automatic deployments using Cloud Build triggers, see [CLOUDBUILD.md](./CLOUDBUILD.md).

Manual deployment:
```bash
gcloud builds submit --config=cloudbuild.yaml
```

### Alternative Deployment

For simple deployment without Cloud Build, see [DEPLOYMENT.md](./DEPLOYMENT.md).

Quick deployment:
```bash
chmod +x deploy.sh
./deploy.sh
```

## API Endpoints

### Root
- `GET /` - Welcome message

### Health Check
- `GET /health` - Health check endpoint with database connectivity status

## API Documentation

Once the application is running, you can access:
- Interactive API docs: `http://localhost:8000/docs`
- Alternative API docs: `http://localhost:8000/redoc`

## Database Migrations

### Create a new migration
```bash
alembic revision --autogenerate -m "description"
```

### Apply migrations
```bash
alembic upgrade head
```

### Rollback last migration
```bash
alembic downgrade -1
```

## Project Structure

```
.
├── alembic/           # Alembic migration files
│   ├── versions/      # Migration versions
│   ├── env.py         # Alembic environment config
│   └── script.py.mako # Migration template
├── main.py            # FastAPI application entry point
├── config.py          # Application configuration
├── database.py        # Database connection and session management
├── requirements.txt   # Python dependencies
├── alembic.ini        # Alembic configuration
├── env.example        # Environment variables template
├── Dockerfile         # Docker configuration
├── cloudbuild.yaml    # Google Cloud Build configuration
├── app.yaml           # App Engine configuration (optional)
├── deploy.sh          # Deployment script
├── startup.sh         # Container startup script
├── DEPLOYMENT.md      # Deployment documentation
├── CLOUDBUILD.md      # Cloud Build configuration guide
└── README.md          # This file
```

## Development

### Adding Models

1. Create your model in `database.py` or a separate `models.py` file:

```python
from sqlalchemy import Column, Integer, String
from database import Base

class YourModel(Base):
    __tablename__ = "your_table"
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
```

2. Create a migration:
```bash
alembic revision --autogenerate -m "add your_model table"
```

3. Apply the migration:
```bash
alembic upgrade head
```

### Adding Endpoints

Add your endpoints in `main.py` or create separate router files for better organization.

## License

MIT

