# Use the official Python image as the base image
FROM python:3.9.12

ENV PYTHONPATH "${PYTHONPATH}:."

# Create the user that will run the app
RUN adduser --disabled-password --gecos '' ml-api-user

ADD ./car-prices-api /opt/car-prices-api/

ARG PIP_EXTRA_INDEX_URL

# Set the working directory in the container in which our commands will be run from
WORKDIR /opt/car-prices-api

# Install the dependencies
RUN pip install --no-cache-dir -r /opt/car-prices-api/requirements.txt
RUN pip install --upgrade pip

# Copy the application code into the container
COPY . .

USER ml-api-user

# Expose the port on which the Flask app will run (change as needed)
EXPOSE 8001

# Set the command to run the Fast application
CMD ["bash", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]

