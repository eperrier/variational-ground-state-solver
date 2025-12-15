# Running with Docker

## Quick start

From the repository root:

```bash
docker compose up --build
```

Then open:

- `http://localhost:8501`

## Output persistence

The default `docker-compose.yml` mounts a host folder into the container:

- Host: `./outputs/`
- Container: `/workspace/1 - RESEARCH/PSITest`

So the application writes runs to `./outputs/runs/...` on your machine.

You can change the base output directory by changing the environment variable:

- `PSITEST_OUTPUT_DIR`

In Docker, this is set in `docker-compose.yml`.

## GPU mode

If you have an NVIDIA GPU and have installed the NVIDIA Container Toolkit, you can enable GPU access with the overlay file:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

The `docker-compose.gpu.yml` enables `gpus: all` for the container.

## Stopping

```bash
docker compose down
```

Your outputs remain in `./outputs/`.

## Notes on images and dependencies

The `Dockerfile` uses a CUDA runtime PyTorch image (`pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime`). It can still run on CPU-only machines, but GPU access requires the NVIDIA Container Toolkit.
