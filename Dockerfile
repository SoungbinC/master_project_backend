# Use Python 3.11 base image
FROM python:3.11

# Set working directory inside the container
WORKDIR /app

# Copy only requirements first (for caching)
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . /app

# Expose port 8000
EXPOSE 8000

# Command to run FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
