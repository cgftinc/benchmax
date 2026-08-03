# Neon setup

this one-time setup prepares a new Neon project to store and search the example corpus. it configures the required Postgres extensions and corpus schema, creates separate database connections for data preparation and read-only search, and generates `.env.neon` for the commands in the main README.

the Neon API key is required only while running setup.

## 1. create a Neon project

sign in to the [Neon Console](https://console.neon.tech/) and create a new project for this example. the default project settings are sufficient.

after creating the project, copy its project ID from the Neon Console. a project ID looks similar to:

```text
flat-frost-16947914
```

## 2. create a Neon API key

follow Neon's [API key instructions](https://neon.com/docs/manage/api-keys) to create an API key with access to the new project.

if the project belongs to a Neon organization and project-scoped keys are available, use a key scoped to this project. otherwise, use a personal API key that can access it. copy the API key when Neon displays it.

## 3. run setup

from the example directory:

```bash
cd examples/neon_rag

export NEON_API_KEY="..."
export NEON_PROJECT_ID="..."

uv run python setup_neon.py
```

the setup command:

1. verifies access to the selected Neon project.
2. connects to its Postgres database.
3. enables the vector and text-search extensions.
4. creates the schema that will hold the corpus tables and indexes.
5. creates a database connection that can initialize and update the corpus.
6. creates a read-only database connection for retrieving passages during validation and training.
7. generates `.env.neon` with both database connections.

the setup command can be rerun against an existing project.

## generated database connections

setup generates two database URLs in `.env.neon`:

```dotenv
NEON_DATA_PREPARATION_DATABASE_URL="postgresql://..."
NEON_SEARCH_DATABASE_URL="postgresql://..."
```

### data preparation database URL

the data preparation URL can create and update document chunks, embeddings, and metadata in Neon. it is used while preparing or updating the corpus and is not provided to the training environment.

### search database URL

the search URL is read-only. `NeonRagEnv` uses it to retrieve passages during validation and training. it cannot add, update, or remove corpus content.

after setup, continue with [run end to end](README.md#run-end-to-end).
