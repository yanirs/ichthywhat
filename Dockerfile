FROM python:3.10-slim

WORKDIR /app

# TODO: figure out what's a good setting for uid/gid
ARG UID=1000
ARG GID=1000
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
CMD ["uvicorn", "--host", "0.0.0.0", "ichthywhat.api:api"]
