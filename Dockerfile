# Use Python 3.11 base image
FROM python:3.11

# Set working directory inside the container
WORKDIR /app

# Copy the project files into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn torch torchvision requests pillow numpy pydantic

# Expose port 8000
EXPOSE 8000

# Command to run FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
