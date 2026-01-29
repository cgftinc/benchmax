"""Utility functions for search environment integration tests."""

import aiohttp
from typing import Dict, List, Any, Optional


async def create_account(
    base_url: str, email: str, password: str, name: str = "Test User"
) -> None:
    """Create a user account."""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/api/auth/sign-up/email",
            json={"email": email, "password": password, "name": name},
        ) as resp:
            if resp.status == 200:
                return

            error = await resp.json()
            # Account already exists is ok
            if "already exists" in str(error.get("error", "")).lower() or \
               "already exists" in str(error.get("message", "")).lower():
                return

            raise Exception(f"Failed to create account: {error}")


async def sign_in(base_url: str, email: str, password: str) -> str:
    """Sign in and return session cookie."""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/api/auth/sign-in/email",
            json={"email": email, "password": password},
        ) as resp:
            if resp.status != 200:
                error = await resp.json()
                raise Exception(f"Failed to sign in: {error}")

            set_cookie = resp.headers.get("set-cookie")
            if not set_cookie:
                raise Exception("No session cookie received")

            return set_cookie


async def create_api_key(base_url: str, session_cookie: str, name: str) -> tuple[str, str]:
    """Create an API key and return (key, key_id)."""
    async with aiohttp.ClientSession() as session:
        # Create API key
        async with session.post(
            f"{base_url}/api/api-keys",
            headers={"Cookie": session_cookie},
            json={"name": name},
        ) as resp:
            if resp.status not in (200, 201):
                error = await resp.json()
                raise Exception(f"Failed to create API key: {error}")

            data = await resp.json()
            api_key = data["key"]

        # Get API key ID
        async with session.get(
            f"{base_url}/api/api-keys",
            headers={"Cookie": session_cookie},
        ) as resp:
            if resp.status != 200:
                raise Exception("Failed to fetch API keys list")

            keys = await resp.json()
            test_key = next((k for k in keys if k["name"] == name), None)

            if not test_key:
                raise Exception("Could not find the created API key")

            return api_key, test_key["id"]


async def delete_api_key(base_url: str, session_cookie: str, key_id: str) -> None:
    """Delete an API key."""
    async with aiohttp.ClientSession() as session:
        async with session.delete(
            f"{base_url}/api/api-keys",
            headers={"Cookie": session_cookie},
            json={"id": key_id},
        ) as resp:
            if resp.status != 200:
                error = await resp.json()
                raise Exception(f"Failed to delete API key: {error}")


async def create_corpus(base_url: str, api_key: str, name: str) -> str:
    """Create a corpus and return its ID."""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/api/corpora",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"name": name},
        ) as resp:
            if resp.status not in (200, 201):
                error = await resp.json()
                raise Exception(f"Failed to create corpus: {error}")

            data = await resp.json()
            return data["id"]


async def upload_chunks(
    base_url: str,
    api_key: str,
    corpus_id: str,
    filename: str,
    chunks: List[Dict[str, Any]]
) -> int:
    """Upload chunks to a corpus and return count of inserted chunks."""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/api/corpora/{corpus_id}/chunks",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"filename": filename, "chunks": chunks},
        ) as resp:
            if resp.status not in (200, 201):
                error = await resp.json()
                raise Exception(f"Failed to upload chunks: {error}")

            data = await resp.json()
            return data.get("insertedCount", 0)


async def delete_corpus(base_url: str, api_key: str, corpus_id: str) -> None:
    """Delete a corpus."""
    async with aiohttp.ClientSession() as session:
        async with session.delete(
            f"{base_url}/api/corpora/{corpus_id}",
            headers={"Authorization": f"Bearer {api_key}"},
        ) as resp:
            if resp.status != 200:
                error = await resp.json()
                raise Exception(f"Failed to delete corpus: {error}")


def get_test_files() -> List[Dict[str, Any]]:
    """Return test data files for integration tests."""
    return [
        {
            "filename": "config.json",
            "chunks": [
                {
                    "content": "Database configuration: host=localhost, port=5432",
                    "metadata": {"type": "config", "page": 1},
                },
                {
                    "content": "API configuration: endpoint=/api/v1, timeout=3000",
                    "metadata": {"type": "config", "page": 2},
                },
            ],
        },
        {
            "filename": "settings.json",
            "chunks": [
                {
                    "content": "User preferences: theme=dark, language=en",
                    "metadata": {"type": "settings", "page": 1},
                },
                {
                    "content": "Feature flags: search_v2=true, beta_mode=false",
                    "metadata": {"type": "settings", "page": 2},
                },
            ],
        },
        {
            "filename": "README.md",
            "chunks": [
                {
                    "content": "Welcome to our application! This guide will help you get started.",
                    "metadata": {"type": "documentation", "section": "intro"},
                },
                {
                    "content": "Configuration instructions: Set up your database connection.",
                    "metadata": {"type": "documentation", "section": "setup"},
                },
                {
                    "content": "API documentation: Learn how to use our REST endpoints.",
                    "metadata": {"type": "documentation", "section": "api"},
                },
            ],
        },
        {
            "filename": "guide.txt",
            "chunks": [
                {
                    "content": "Getting started with PostgreSQL database setup.",
                    "metadata": {"type": "tutorial", "difficulty": "beginner"},
                },
                {
                    "content": "Advanced database optimization techniques.",
                    "metadata": {"type": "tutorial", "difficulty": "advanced"},
                },
            ],
        },
        {
            "filename": "app.config.ts",
            "chunks": [
                {
                    "content": "TypeScript configuration for the application.",
                    "metadata": {"type": "code", "language": "typescript"},
                },
                {
                    "content": "Environment variables and secrets management.",
                    "metadata": {"type": "code", "language": "typescript"},
                },
            ],
        },
    ]
