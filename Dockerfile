FROM python:3.10-slim

WORKDIR /app

# This uid/gid may not exist on the host. Change if needed.
ARG UID=1100
ARG GID=1100
RUN groupadd -g "${GID}" python && \
    useradd --create-home --no-log-init -u "${UID}" -g "${GID}" python && \
    chown python:python /app
USER python
ENV PATH="${PATH}:/home/python/.local/bin"

# Copy the minimal files to create requirements.txt and install the core dependencies.
# With these options, non-dev dependencies are exported and hashes are verified by pip.
COPY --chown=python:python poetry.lock pyproject.toml ./
RUN mypip='pip3 --no-cache-dir --no-python-version-warning -qq' && \
    $mypip install poetry==1.4.2 && \
    poetry export --output requirements.txt && \
    $mypip uninstall --yes poetry && \
    $mypip install -r requirements.txt

COPY --chown=python:python ichthywhat/ ./ichthywhat/
COPY --chown=python:python resources/model.onnx ./resources/model.onnx

EXPOSE 8000
# Default number of workers.
ENV WEB_CONCURRENCY=1
# Set to 0.0.0.0 to allow access from the outside world.
ENV UVICORN_HOST="127.0.0.1"
# Override to restrict origins if needed.
ENV ICHTHYWHAT_API_ALLOW_ORIGINS="*"
CMD exec uvicorn --host "$UVICORN_HOST" ichthywhat.api:api
