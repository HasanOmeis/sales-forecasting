# start from a small official Linux image that already has Python 3.12
FROM python:3.12-slim

# install libgomp1, the system library LightGBM needs to run on Linux, then delete the apt cache to keep the image small
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# all following commands run inside /app in the container
WORKDIR /app

# copy only the API requirements first, so Docker can reuse this layer when only code changes
COPY requirements-api.txt .

# install the API libraries without keeping pip's download cache
RUN pip install --no-cache-dir -r requirements-api.txt

# copy the source code into /app/src
COPY src/ ./src/

# copy the saved model into /app/models
COPY models/ ./models/

# document that the app listens on port 8000
EXPOSE 8000

# start the API; 0.0.0.0 makes it reachable from outside the container
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]