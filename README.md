# AI Project

Python monorepo containing a machine learning package and a FastAPI service intended to expose the model.

## Architecture

```text
AI_project/
├── ai_model/              # ML model package
│   ├── pyproject.toml
│   └── src/ai_model/
├── api/                   # FastAPI HTTP API
│   ├── pyproject.toml
│   └── src/api/
├── docker-compose.yml     # Extension point for local services
└── README.md
```

### Components

- **`ai_model`**: Python package using `scikit-learn`. Its `ai-model` entry point is currently a minimal example.
- **`api`**: Minimal FastAPI service with interactive OpenAPI documentation and a health endpoint.
- **`docker-compose.yml`**: Present but no services are configured yet.

## Prerequisites

- Python `3.14` or a version compatible with both `pyproject.toml` files.
- [uv](https://docs.astral.sh/uv/) to manage Python environments and dependencies.
- Docker and Docker Compose only if services are added to `docker-compose.yml`.

## Installation

Each component has its own virtual environment and dependency file. From the repository root:

```powershell
cd ai_model
uv sync

cd ..\api
uv sync
```

On macOS or Linux, use `cd ../api` instead of `cd ..\api`.

## Run the API

From the `api` directory:

```powershell
uv run uvicorn api:app --app-dir src --reload
```

The API is available at <http://127.0.0.1:8000>.

- Swagger documentation: <http://127.0.0.1:8000/docs>
- Health check: <http://127.0.0.1:8000/health>

Example response:

```json
{"status":"ok"}
```

The API uses strict Pydantic schemas: unknown fields are rejected and implicit type conversions are not accepted.

## Run the Model Package

From the `ai_model` directory:

```powershell
uv run ai-model
```

The current entry point prints a demo message. Inference logic can be added under `ai_model/src/ai_model/` and then integrated into the API.

## Testing and Quality

No automated tests or linting tools are currently defined. Before contributing, at minimum verify that both environments synchronize correctly:

```powershell
cd ai_model
uv sync

cd ..\api
uv sync
```

To test the API manually:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

## Development

The `ai_model` package development dependencies include JupyterLab:

```powershell
cd ai_model
uv run jupyter lab
```

The API dependencies include Uvicorn with its standard extras for the development server.

## Contributing

1. Create or activate the relevant component environment with `uv sync`.
2. Modify the code in the corresponding `src/` directory.
3. Verify that the component starts and that the API `/health` endpoint works.
4. Add the required tests or documentation for each new feature.
5. Keep dependencies in the component that uses them.

## License

No license is currently declared in the repository.
