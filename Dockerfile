FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc make && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir numpy pytest

# Build Phase 1 C binary
RUN cd 01-KMeans-Basic && gcc -ansi -Wall -Wextra -Werror -pedantic-errors -o lloyd lloyd_clustering.c -lm

CMD ["pytest", "tests/", "-v"]
