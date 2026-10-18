import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest import mock

from utils import generation_utils


FAKE_ORCA_KEY = "sk-orca-test-not-a-real-key"
PROVIDER_ENV_VARS = (
    "GOOGLE_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "ORCAROUTER_API_KEY",
)


class OrcaRouterProviderTests(unittest.TestCase):
    def test_to_orcarouter_model_id_mapping(self) -> None:
        self.assertEqual(
            generation_utils._to_orcarouter_model_id("deepseek/deepseek-v4-flash"),
            "deepseek/deepseek-v4-flash",
        )
        self.assertEqual(
            generation_utils._to_orcarouter_model_id("openai/gpt-5.5"),
            "openai/gpt-5.5",
        )
        self.assertEqual(
            generation_utils._to_orcarouter_model_id("gemini-3-pro-preview"),
            "google/gemini-3-pro-preview",
        )
        self.assertEqual(
            generation_utils._to_orcarouter_model_id("gemini-3.1-flash-image-preview"),
            "google/gemini-3.1-flash-image-preview",
        )
        self.assertEqual(
            generation_utils._to_orcarouter_model_id("gpt-5.5"),
            "gpt-5.5",
        )

    def test_reinitialize_clients_creates_orcarouter_client_only_when_key_is_set(self) -> None:
        saved_env = {name: os.environ.get(name) for name in PROVIDER_ENV_VARS}
        saved_clients = {
            "gemini_client": generation_utils.gemini_client,
            "anthropic_client": generation_utils.anthropic_client,
            "openai_client": generation_utils.openai_client,
            "openrouter_client": generation_utils.openrouter_client,
            "openrouter_api_key": generation_utils.openrouter_api_key,
            "orcarouter_client": generation_utils.orcarouter_client,
            "orcarouter_api_key": generation_utils.orcarouter_api_key,
        }
        try:
            for name in PROVIDER_ENV_VARS:
                os.environ.pop(name, None)
            with mock.patch.object(generation_utils, "model_config", {}):
                initialized = generation_utils.reinitialize_clients()
                self.assertNotIn("OrcaRouter", initialized)
                self.assertIsNone(generation_utils.orcarouter_client)
                self.assertEqual(generation_utils.orcarouter_api_key, "")

                os.environ["ORCAROUTER_API_KEY"] = FAKE_ORCA_KEY
                initialized = generation_utils.reinitialize_clients()
                self.assertEqual(initialized, ["OrcaRouter"])
                self.assertEqual(generation_utils.orcarouter_api_key, FAKE_ORCA_KEY)
                client = generation_utils.orcarouter_client
                self.assertIsNotNone(client)
                self.assertEqual(str(client.base_url).rstrip("/"), "https://api.orcarouter.ai/v1")
                self.assertEqual(client.api_key, FAKE_ORCA_KEY)
                self.assertIsNone(generation_utils.openrouter_client)
                self.assertIsNone(generation_utils.gemini_client)
                self.assertIsNone(generation_utils.openai_client)
                self.assertIsNone(generation_utils.anthropic_client)
        finally:
            for name, value in saved_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            for name, value in saved_clients.items():
                setattr(generation_utils, name, value)

    def test_call_model_routes_orcarouter_prefix(self) -> None:
        async def run() -> None:
            config = SimpleNamespace(
                system_instruction="system",
                temperature=0.2,
                candidate_count=1,
                max_output_tokens=128,
            )
            with (
                mock.patch.object(
                    generation_utils,
                    "call_orcarouter_with_retry_async",
                    mock.AsyncMock(return_value=["pong"]),
                ) as orca_mock,
                mock.patch.object(
                    generation_utils,
                    "call_openrouter_with_retry_async",
                    mock.AsyncMock(return_value=["openrouter"]),
                ) as openrouter_mock,
                mock.patch.object(
                    generation_utils,
                    "call_gemini_with_retry_async",
                    mock.AsyncMock(return_value=["gemini"]),
                ) as gemini_mock,
            ):
                result = await generation_utils.call_model_with_retry_async(
                    model_name="orcarouter/deepseek/deepseek-v4-flash",
                    contents=[{"type": "text", "text": "ping"}],
                    config=config,
                    max_attempts=1,
                    retry_delay=0,
                    error_context="unit-test",
                )

            self.assertEqual(result, ["pong"])
            orca_mock.assert_awaited_once()
            self.assertEqual(
                orca_mock.await_args.kwargs["model_name"],
                "deepseek/deepseek-v4-flash",
            )
            self.assertEqual(orca_mock.await_args.kwargs["contents"], [{"type": "text", "text": "ping"}])
            self.assertEqual(orca_mock.await_args.kwargs["max_attempts"], 1)
            self.assertEqual(orca_mock.await_args.kwargs["retry_delay"], 0)
            self.assertEqual(orca_mock.await_args.kwargs["error_context"], "unit-test")
            self.assertEqual(orca_mock.await_args.kwargs["config"]["system_prompt"], "system")
            self.assertEqual(orca_mock.await_args.kwargs["config"]["temperature"], 0.2)
            openrouter_mock.assert_not_awaited()
            gemini_mock.assert_not_awaited()

        asyncio.run(run())

    def test_routing_unchanged_when_orcarouter_key_is_unset(self) -> None:
        async def run() -> None:
            config = SimpleNamespace(
                system_instruction="system",
                temperature=1.0,
                candidate_count=1,
                max_output_tokens=64,
            )
            with (
                mock.patch.object(generation_utils, "orcarouter_client", None),
                mock.patch.object(generation_utils, "openrouter_client", object()),
                mock.patch.object(generation_utils, "gemini_client", object()),
                mock.patch.object(
                    generation_utils,
                    "call_openrouter_with_retry_async",
                    mock.AsyncMock(return_value=["via-openrouter"]),
                ) as openrouter_mock,
                mock.patch.object(
                    generation_utils,
                    "call_orcarouter_with_retry_async",
                    mock.AsyncMock(return_value=["via-orca"]),
                ) as orca_mock,
                mock.patch.object(
                    generation_utils,
                    "call_gemini_with_retry_async",
                    mock.AsyncMock(return_value=["via-gemini"]),
                ) as gemini_mock,
            ):
                result = await generation_utils.call_model_with_retry_async(
                    model_name="gemini-3.1-pro-preview",
                    contents=[{"type": "text", "text": "hello"}],
                    config=config,
                    max_attempts=1,
                    retry_delay=0,
                )

            self.assertEqual(result, ["via-openrouter"])
            self.assertEqual(
                openrouter_mock.await_args.kwargs["model_name"],
                "google/gemini-3.1-pro-preview",
            )
            orca_mock.assert_not_awaited()
            gemini_mock.assert_not_awaited()

            with (
                mock.patch.object(generation_utils, "orcarouter_client", None),
                mock.patch.object(generation_utils, "openrouter_client", None),
                mock.patch.object(generation_utils, "gemini_client", object()),
                mock.patch.object(
                    generation_utils,
                    "call_openrouter_with_retry_async",
                    mock.AsyncMock(return_value=["via-openrouter"]),
                ) as openrouter_mock,
                mock.patch.object(
                    generation_utils,
                    "call_gemini_with_retry_async",
                    mock.AsyncMock(return_value=["via-gemini"]),
                ) as gemini_mock,
                mock.patch.object(
                    generation_utils,
                    "call_orcarouter_with_retry_async",
                    mock.AsyncMock(return_value=["via-orca"]),
                ) as orca_mock,
            ):
                prefixed = await generation_utils.call_model_with_retry_async(
                    model_name="openrouter/google/gemini-3.1-pro-preview",
                    contents=[{"type": "text", "text": "hello"}],
                    config=config,
                    max_attempts=1,
                    retry_delay=0,
                )
                self.assertEqual(prefixed, ["via-openrouter"])
                self.assertEqual(
                    openrouter_mock.await_args.kwargs["model_name"],
                    "google/gemini-3.1-pro-preview",
                )
                gemini_mock.assert_not_awaited()
                orca_mock.assert_not_awaited()

                bare = await generation_utils.call_model_with_retry_async(
                    model_name="gemini-3.1-pro-preview",
                    contents=[{"type": "text", "text": "hello"}],
                    config=config,
                    max_attempts=1,
                    retry_delay=0,
                )
                self.assertEqual(bare, ["via-gemini"])
                gemini_mock.assert_awaited_once()
                self.assertEqual(
                    gemini_mock.await_args.kwargs["model_name"],
                    "gemini-3.1-pro-preview",
                )
                orca_mock.assert_not_awaited()

        asyncio.run(run())
