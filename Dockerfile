FROM apache/spark-py as base
# python image + project as base
WORKDIR /spark_forecast
USER root
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get -yq install python3-venv
COPY pyproject.toml ./
COPY poetry.lock ./
ADD /fairy_forecast ./fairy_forecast
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install poetry \
    #&& poetry config virtualenvs.create false \
    && python3 -m venv .venv \
    && poetry install --no-dev --no-interaction --no-ansi
# Test image, added pyspark to run integration tests
FROM base as tester
COPY tests ./tests
RUN source .venv/bin/activate \
    && pip install pytest