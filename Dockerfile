FROM python:3.10-slim

WORKDIR /app

# This uid/gid may not exist on the host. Change if needed.
ARG UID=1100
ARG GID=1100
RUN groupadd -g "${GID}" python && \
    useradd --create-home --no-log-init -u "${UID}" -g "${GID}" python
USER python
ENV PATH="${PATH}:/home/python/.local/bin"

# Installing in two steps to avoid pulling in the massive GPU-based Torch.
# Not relying on the pyproject.toml file as this is meant to be the bare minimum to
# be able to load the model.
RUN pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
      torch==2.0.0 \
      torchvision==0.15.1 && \
    pip3 install --no-cache-dir \
      fastai==2.7.12 \
      fastapi==0.95.0 \
      python-multipart==0.0.6 \
      uvicorn==0.21.1

COPY --chown=python:python ichthywhat/ ./ichthywhat/
COPY --chown=python:python resources/model.pkl ./resources/model.pkl

EXPOSE 8000
# Default number of workers.
ENV WEB_CONCURRENCY=1
# Set to 0.0.0.0 to allow access from the outside world.
ENV UVICORN_HOST="127.0.0.1"
# Override to restrict origins if needed.
ENV ICHTHYWHAT_API_ALLOW_ORIGINS="*"
CMD exec uvicorn --host "$UVICORN_HOST" ichthywhat.api:api
