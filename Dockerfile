# Use a standard Python 3.11 slim image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install all your Python dependencies
# Make sure torch and sentence-transformers are in your requirements.txt!
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your project files (all .py files, etc.) into the container
COPY . .

# Make our startup script executable
RUN chmod +x ./start.sh

# Tell Hugging Face that your app will be on port 8501
EXPOSE 8501

# Run the startup script when the container starts
CMD ["./start.sh"]

