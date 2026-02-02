# Use Python 3.9 (Very stable for TensorFlow)
FROM python:3.9

# Set the working directory inside the container
WORKDIR /code

# Copy the requirements file first (for caching)
COPY ./requirements.txt /code/requirements.txt

# Install the libraries
# We add --no-cache-dir to keep the image small
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the application files
COPY . .

# Create the logs directory (optional but good for safety)
RUN mkdir -p /code/logs

# Command to run the app
# Hugging Face EXPECTS the app to run on port 7860
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app", "--timeout", "120"]