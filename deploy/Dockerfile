FROM python:3.13-slim

EXPOSE 8000

RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 dengine

WORKDIR /opt/app/dengine

COPY pyproject.toml setup.py* requirements.txt* ./

RUN pip install --no-cache-dir .[serve]

COPY . .

RUN pip install --no-cache-dir --no-deps -e .[serve]

RUN chown -R dengine:dengine /opt/app
USER dengine

CMD ["dengine-serve"]
