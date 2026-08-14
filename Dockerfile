# OKUMA CNC Predictive Maintenance -- backend image.
# Build from the backend/ directory: `docker build -t okuma-backend .`
FROM python:3.12-slim

WORKDIR /app

# Install deps first so this layer is cached across code-only changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code + its own copy of models/configs/data (see README.md: "backend/
# duplicates the relevant model/config/data files ... so it can be deployed
# on its own -- that's deliberate, not an accidental copy").
COPY . .

# maintenance_events.db is created here on first write (see backend_app.py) --
# mount this as a volume in production so events survive container restarts:
#   docker run -v okuma-maintenance-data:/app/data ...
VOLUME ["/app/data"]

EXPOSE 8000

# ALLOWED_ORIGINS and API_KEY are read from the environment at runtime, not
# baked in -- see backend_app.py's security-config comment block. Set them
# with `docker run -e API_KEY=... -e ALLOWED_ORIGINS=... ...` or in
# docker-compose.yml / your hosting platform's env var settings.
CMD ["uvicorn", "backend_app:app", "--host", "0.0.0.0", "--port", "8000"]
