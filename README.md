
# SkillSathi AI | SkillShikshya Guide

An AI-powered bilingual chatbot for SkillShikshya, designed to help students and professionals explore courses, career paths, practical skills, and training opportunities using verified website information.

The chatbot uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from the SkillShikshya website and generate answers with source references.

## Features

- **Bilingual chatbot:** Supports English, Nepali, and Romanized Nepali questions.
- **Website-grounded answers:** Generates responses using indexed SkillShikshya website content.
- **Course exploration:** Helps learners discover available courses and their details.
- **Course comparison:** Supports comparisons between relevant training programs.
- **Career guidance:** Answers questions about skills, career paths, and learning opportunities.
- **Multi-question handling:** Processes multiple questions in a single message.
- **Source citations:** Provides links to the website pages used to support answers.
- **Automatic website synchronization:** Periodically checks the website for changes.
- **Incremental indexing:** Reuses unchanged vectors and embeds updated content.
- **Knowledge activation:** Verifies new knowledge generations before activation.
- **Web-based chat interface:** Includes a custom SkillShikshya-themed frontend.

## Technology Stack

| Component | Technology |
| --- | --- |
| Programming language | Python 3.12+ |
| Backend | FastAPI |
| Frontend | HTML, CSS, JavaScript |
| Website crawler | Crawl4AI |
| Embedding model | BAAI/bge-small-en-v1.5 |
| Vector database | Qdrant |
| Answer generation | Google Gemini |
| Dependency management | uv |
| Testing | Python unittest |

## Project Structure

```text
skillsathi-ai/
├── app/
│   ├── agent/
│   │   ├── router.py
│   │   ├── executor.py
│   │   └── service.py
│   ├── api/
│   │   ├── server.py
│   │   ├── chat.html
│   │   └── widget.js
│   ├── ingestion/
│   │   ├── crawler.py
│   │   ├── auto_sync.py
│   │   ├── auto_sync_scheduler.py
│   │   ├── change_detection.py
│   │   └── vector_activation.py
│   └── rag/
│       └── engine.py
├── tests/
├── data/                  # Generated locally, not committed
├── pyproject.toml
├── uv.lock
├── .env.example
├── .gitignore
└── README.md
```

## Prerequisites

Before running the application, install or configure:

- Python 3.12 or newer
- uv package manager
- Google Gemini API key
- Crawl4AI browser dependencies
- Git

## Installation

### 1. Clone the repository

Once the project has been published to GitHub:

```bash
git clone https://github.com/<YOUR_USERNAME>/skillsathi-ai.git
cd skillsathi-ai
```

Replace `<YOUR_USERNAME>` with the actual GitHub account name.

### 2. Install dependencies

```bash
uv sync
```

### 3. Configure environment variables

Copy the example configuration:

```bash
cp .env.example .env
```

Open `.env` and insert your Gemini API key.

Never commit `.env` or share your API credentials publicly.

## Environment Variables

| Variable | Description |
| --- | --- |
| `GEMINI_API_KEY` | API key used for Gemini-generated answers |
| `GEMINI_MODEL` | Optional Gemini model configuration |
| `SKILLSHIKSHYA_AUTO_SYNC_ENABLED` | Enable or disable automatic website synchronization |
| `SKILLSHIKSHYA_AUTO_SYNC_INTERVAL_HOURS` | Refresh interval in hours |

## Running the Application

Start the FastAPI server:

```bash
uv run uvicorn app.api.server:app \
  --host 127.0.0.1 \
  --port 8000 \
  --workers 1
```

Open the chatbot in your browser:

http://127.0.0.1:8000

### Running with Automatic Website Synchronization

Enable the scheduler using:

```bash
SKILLSHIKSHYA_AUTO_SYNC_ENABLED=1 \
SKILLSHIKSHYA_AUTO_SYNC_INTERVAL_HOURS=6 \
uv run uvicorn app.api.server:app \
  --host 127.0.0.1 \
  --port 8000 \
  --workers 1
```

The scheduler checks the website periodically and prepares updated knowledge when eligible content changes.

**Important:** Automatic synchronization operates only while the application server is running.

The local embedded Qdrant database should be accessed by a single server worker.

## How the RAG Pipeline Works

The knowledge pipeline follows these stages:

1. **Website discovery:** Identify eligible public content pages from SkillShikshya.
2. **Crawling:** Extract website text and metadata using Crawl4AI.
3. **Processing:** Clean and organize content into searchable chunks.
4. **Embeddings:** Convert text chunks into vector representations using FastEmbed.
5. **Storage:** Store vectors and source metadata in Qdrant.
6. **Retrieval:** Find relevant evidence for a student question.
7. **Answer generation:** Generate a grounded answer using Google Gemini.
8. **Citations:** Display source references associated with the response.

## Automatic Knowledge Synchronization

The application includes an automatic website refresh workflow.

During a scheduled refresh, the system:

1. Discovers current eligible website pages.
2. Crawls and verifies the website snapshot.
3. Detects added or updated content.
4. Reuses vectors when content remains unchanged.
5. Creates embeddings for changed content.
6. Validates the new vector collection.
7. Activates the verified collection.

The workflow is designed to preserve previously active knowledge if a refresh fails before activation.



### Verification Status

Verified on October 31, 2026, using a clean GitHub clone.

- **Automated tests:** 266 passed.
- **Fresh installation:** Dependencies installed successfully with `uv sync --frozen`.
- **Live website crawl:** 38 approved public pages crawled successfully.
- **Knowledge activation:** 38 verified pages and 857 embedding vectors indexed.
- **Local Qdrant:** Initial knowledge collection activated successfully.
- **Security:** Unapproved website URLs are excluded from bootstrap.

Page and vector counts may change as SkillShikshya updates its website.

Before using the chatbot for the first time:

1. Install the project dependencies using `uv sync`.
2. Copy `.env.example` to `.env` and configure `GEMINI_API_KEY`.
3. Ensure the FastAPI server is stopped.
4. Run the initial knowledge bootstrap:

```bash
uv run python -m app.ingestion.bootstrap
```

The bootstrap process:

- Crawls eligible public SkillShikshya website pages.
- Processes and verifies the website snapshot.
- Generates text embeddings.
- Creates an initial Qdrant collection.
- Verifies indexed vectors and metadata.
- Activates the `skillshikshya_active` alias.
- Creates the initial knowledge activation receipt.

**Safety:** Run this command only for a fresh installation with an empty knowledge database. The bootstrap refuses to overwrite an existing initialized database.

After successful initialization, start the FastAPI server with automatic synchronization enabled.

**Verification status:** The bootstrap has passed unit tests and an isolated integration test using a real verified 40-page website snapshot and synthetic embeddings. Full fresh-clone execution with live crawling and real embeddings remains to be verified during release acceptance testing.


## Running Tests

Execute the full regression suite:

```bash
uv run python -m unittest discover -s tests -q
```

At the latest verified local checkpoint, all 262 automated tests passed.

The test suite covers routing, RAG functionality, website ingestion, change detection, incremental activation, scheduling, and multi-question handling.

## Security and Data Handling

The repository excludes the following local files and directories:

- `.env` and other local secret files
- `data/` and generated crawl snapshots
- Local Qdrant database files
- Python virtual environments
- Cache files and generated logs

API keys must be stored in environment variables and must not be hardcoded into source files.

Website content is treated as untrusted reference data. The chatbot is instructed not to follow commands embedded in retrieved website text.

## Limitations

- Answers depend on the information available in indexed website sources.
- The chatbot is not intended to provide unrestricted general web search.
- Marketing statements are not treated as independently verified rankings.
- Some information may become outdated between website refreshes.
- A fresh installation requires initial knowledge setup.
- The automatic scheduler runs only while its server process is active.
- Local embedded Qdrant is not intended for concurrent multi-worker deployment.
- Public production hosting requires additional deployment and security validation.

## Project Status

The local MVP has been implemented and functionally tested.

Completed functionality includes:

- Website-grounded RAG chatbot
- Bilingual question answering
- Course information and comparisons
- Multi-question routing
- Source citations
- Automated website synchronization
- Verified incremental knowledge activation
- Automated regression tests

Remaining release work includes first-time installation verification, deployment review, GitHub publication, and fresh-clone acceptance testing.

## Developer

**Ganesh Pokharel**

Project: SkillSathi AI | SkillShikshya Guide

Website content source: https://skillshikshya.com
