# Use a standard Python 3.11 slim image
FROM python:3.11-slim

# Set up a new user named "user" with user ID 1000 (required by HF Spaces)
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy the requirements file first to leverage Docker layer caching
# Use --chown=user to ensure proper permissions
COPY --chown=user requirements.txt .

# Install all your Python dependencies
# Make sure torch and sentence-transformers are in your requirements.txt!
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all your project files into the container with proper ownership
COPY --chown=user . .

# Make our startup script executable
RUN chmod +x ./start.sh

# Tell Hugging Face that your app will be on port 8501
EXPOSE 8501

# Run the startup script when the container starts
CMD ["./start.sh"]
