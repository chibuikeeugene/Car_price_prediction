# Use the official Python image as the base image
FROM python:3.9.12

# Create the user that will run the app
RUN adduser --disabled-password --gecos '' ml-api-user

COPY ./car-prices-api /opt/car-prices-api/
# # Make directory
# RUN mkdir /app

# Set the working directory in the container
WORKDIR /opt/car-prices-api

# Install the dependencies
RUN pip install --no-cache-dir -r /opt/car-prices-api/requirements.txt
RUN pip install --upgrade pip

# Copy the application code into the container
COPY . .

USER ml-api-user

# Expose the port on which the Flask app will run (change as needed)
EXPOSE 5000

# Set the command to run the Fast application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]

