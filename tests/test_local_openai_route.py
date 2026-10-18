import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from agents.polish_agent import PolishAgent
from agents.vanilla_agent import VanillaAgent
from agents.visualizer_agent import VisualizerAgent
from utils import generation_utils
from utils.config import ExpConfig


LOCAL_ENV_VARS = (
    "LOCAL_OPENAI_BASE_URL",
    "LOCAL_OPENAI_API_KEY",
    "OLLAMA_BASE_URL",
    "OLLAMA_API_KEY",
)
CLIENT_GLOBALS = (
    "gemini_client",
    "anthropic_client",
    "openai_client",
    "openrouter_client",
    "openrouter_api_key",
    "orcarouter_client",
    "orcarouter_api_key",
    "atlascloud_client",
    "local_openai_client",
    "ollama_client",
    "model_config",
)


def _text_config() -> SimpleNamespace:
    return SimpleNamespace(
        system_instruction="Follow the user request.",
        temperature=0.25,
        candidate_count=2,
        max_output_tokens=1234,
    )


def _diagram_config(work_dir: Path, image_model_name: str) -> ExpConfig:
    return ExpConfig(
        dataset_name="PaperBananaBench",
        task_name="diagram",
        exp_mode="vanilla",
        retrieval_setting="none",
        main_model_name="gemini-3.1-pro-preview",
        image_gen_model_name=image_model_name,
        work_dir=work_dir,
    )


class LocalOpenAIRouteTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._saved_env = {name: os.environ.get(name) for name in LOCAL_ENV_VARS}
        self._saved_clients = {
            name: getattr(generation_utils, name) for name in CLIENT_GLOBALS
        }
        for name in LOCAL_ENV_VARS:
            os.environ.pop(name, None)
        generation_utils.model_config = {}
        generation_utils.local_openai_client = None
        generation_utils.ollama_client = None

    def tearDown(self) -> None:
        for name, value in self._saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        for name, value in self._saved_clients.items():
            setattr(generation_utils, name, value)

    async def test_local_prefix_reuses_client_and_strips_prefix(self) -> None:
        client = SimpleNamespace()
        generation_utils.local_openai_client = client
        contents = [{"type": "text", "text": "hello"}]

        with patch.object(
            generation_utils,
            "call_openai_with_retry_async",
            new=AsyncMock(return_value=["local ok"]),
        ) as call, patch.object(
            generation_utils,
            "call_gemini_with_retry_async",
            new=AsyncMock(return_value=["gemini"]),
        ) as gemini, patch.object(
            generation_utils,
            "_make_local_text_client",
            side_effect=AssertionError("local client should be reused"),
        ):
            result = await generation_utils.call_model_with_retry_async(
                "local/org/qwen2.5:14b",
                contents,
                _text_config(),
                max_attempts=1,
                retry_delay=0,
                error_context="unit test",
            )

        self.assertEqual(["local ok"], result)
        self.assertEqual("org/qwen2.5:14b", call.await_args.kwargs["model_name"])
        self.assertIs(client, call.await_args.kwargs["client"])
        self.assertEqual("Local OpenAI-compatible", call.await_args.kwargs["provider_name"])
        self.assertEqual(contents, call.await_args.kwargs["contents"])
        self.assertEqual("Follow the user request.", call.await_args.kwargs["config"]["system_prompt"])
        self.assertEqual(2, call.await_args.kwargs["config"]["candidate_num"])
        self.assertEqual(1234, call.await_args.kwargs["config"]["max_completion_tokens"])
        self.assertEqual(1, call.await_args.kwargs["max_attempts"])
        self.assertEqual("unit test", call.await_args.kwargs["error_context"])
        gemini.assert_not_awaited()

    async def test_local_routes_send_max_tokens_to_chat_completions(self) -> None:
        created = AsyncMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
            )
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=created))
        )
        generation_utils.local_openai_client = client
        generation_utils.ollama_client = client

        for model_name in ("local/qwen2.5:14b", "ollama/llama3.1:8b"):
            created.reset_mock()
            result = await generation_utils.call_model_with_retry_async(
                model_name,
                [{"type": "text", "text": "hello"}],
                _text_config(),
                max_attempts=1,
                retry_delay=0,
            )
            self.assertEqual(["ok", "ok"], result)
            self.assertEqual(2, created.await_count)
            for call in created.await_args_list:
                self.assertEqual(1234, call.kwargs["max_tokens"])
                self.assertNotIn("max_completion_tokens", call.kwargs)

        for provider_name in ("OpenAI", "Atlas Cloud"):
            created.reset_mock()
            result = await generation_utils.call_openai_with_retry_async(
                model_name="gpt-4.1",
                contents=[{"type": "text", "text": "hello"}],
                config={
                    "system_prompt": "system",
                    "temperature": 0.2,
                    "candidate_num": 2,
                    "max_completion_tokens": 321,
                },
                max_attempts=1,
                retry_delay=0,
                client=client,
                provider_name=provider_name,
            )
            self.assertEqual(["ok", "ok"], result)
            self.assertEqual(2, created.await_count)
            for call in created.await_args_list:
                self.assertEqual(321, call.kwargs["max_completion_tokens"])
                self.assertNotIn("max_tokens", call.kwargs)

    async def test_refine_helpers_reject_local_image_models(self) -> None:
        import app
        import demo

        async def fail_if_called(*args, **kwargs):
            raise AssertionError("hosted image provider should not be called")

        hosted = {
            "call_openrouter_image_generation_with_retry_async": fail_if_called,
            "call_orcarouter_image_generation_with_retry_async": fail_if_called,
            "call_gemini_with_retry_async": fail_if_called,
        }
        with patch.dict(os.environ, {"IMAGE_GEN_MODEL_NAME": "local/llava"}), patch.multiple(
            generation_utils, **hosted
        ):
            with self.assertRaisesRegex(ValueError, "text-only"):
                await app.refine_image_with_nanoviz(b"img", "make it clearer")
        with patch.dict(os.environ, {"IMAGE_GEN_MODEL_NAME": "ollama/llava"}), patch.multiple(
            generation_utils, **hosted
        ):
            with self.assertRaisesRegex(ValueError, "text-only"):
                await demo.refine_image_with_nanoviz(b"img", "make it clearer")

    async def test_ollama_prefix_defaults_to_local_server(self) -> None:
        with patch.object(
            generation_utils,
            "call_openai_with_retry_async",
            new=AsyncMock(return_value=["ollama ok"]),
        ) as call:
            result = await generation_utils.call_model_with_retry_async(
                "Ollama/Llama3.1",
                [{"type": "text", "text": "hello"}],
                _text_config(),
                max_attempts=1,
                retry_delay=0,
            )

        self.assertEqual(["ollama ok"], result)
        client = call.await_args.kwargs["client"]
        self.assertIs(generation_utils.ollama_client, client)
        self.assertEqual("Llama3.1", call.await_args.kwargs["model_name"])
        self.assertEqual("Ollama", call.await_args.kwargs["provider_name"])
        self.assertEqual(str(client.base_url).rstrip("/"), "http://localhost:11434/v1")
        self.assertEqual(client.api_key, "ollama")

    async def test_local_prefix_requires_base_url(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "LOCAL_OPENAI_BASE_URL"):
            await generation_utils.call_model_with_retry_async(
                "local/qwen2.5:14b",
                [],
                _text_config(),
            )

    async def test_empty_local_model_id_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "local model id cannot be empty"):
            await generation_utils.call_model_with_retry_async("local/", [], _text_config())
        with self.assertRaisesRegex(ValueError, "ollama model id cannot be empty"):
            await generation_utils.call_model_with_retry_async("ollama/", [], _text_config())

    def test_reinitialize_clients_reads_base_urls(self) -> None:
        os.environ["LOCAL_OPENAI_BASE_URL"] = "http://127.0.0.1:8000/v1/"
        os.environ["LOCAL_OPENAI_API_KEY"] = "secret"
        generation_utils.model_config = {
            "base_urls": {"ollama_base_url": "http://10.0.0.2:11434/v1"},
            "api_keys": {"ollama_api_key": "ollama-secret"},
        }

        initialized = generation_utils.reinitialize_clients()

        self.assertIn("Local OpenAI-compatible", initialized)
        self.assertIn("Ollama", initialized)
        local_client = generation_utils.local_openai_client
        ollama_client = generation_utils.ollama_client
        self.assertEqual(str(local_client.base_url).rstrip("/"), "http://127.0.0.1:8000/v1")
        self.assertEqual(local_client.api_key, "secret")
        self.assertEqual(str(ollama_client.base_url).rstrip("/"), "http://10.0.0.2:11434/v1")
        self.assertEqual(ollama_client.api_key, "ollama-secret")

    def test_image_helper_rejects_text_prefixes_only(self) -> None:
        self.assertTrue(generation_utils.is_local_openai_model_name("local/qwen2.5:14b"))
        self.assertTrue(generation_utils.is_local_openai_model_name(" ollama/llama3.1:8b "))
        self.assertFalse(
            generation_utils.is_local_openai_model_name("gemini-3.1-flash-image-preview")
        )

        with self.assertRaisesRegex(ValueError, "text-only"):
            generation_utils.assert_not_local_openai_image_model(
                "local/gpt-image-1",
                route_name="image generation",
            )
        generation_utils.assert_not_local_openai_image_model("gemini-3.1-flash-image-preview")

    async def test_image_agents_reject_prefixes_before_hosted_providers(self) -> None:
        async def fail_if_called(*args, **kwargs):
            raise AssertionError("hosted image provider should not be called")

        hosted = {
            "call_openai_image_generation_with_retry_async": fail_if_called,
            "call_openrouter_image_generation_with_retry_async": fail_if_called,
            "call_orcarouter_image_generation_with_retry_async": fail_if_called,
            "call_gemini_with_retry_async": fail_if_called,
        }
        with TemporaryDirectory() as tmp, patch.multiple(generation_utils, **hosted), patch.object(
            generation_utils, "openrouter_client", object()
        ):
            work_dir = Path(tmp)
            vanilla = VanillaAgent(
                exp_config=_diagram_config(work_dir, "local/gpt-image-1")
            )
            with self.assertRaisesRegex(ValueError, "text-only"):
                await vanilla.process(
                    {
                        "content": "Method text",
                        "visual_intent": "Draw the method",
                        "additional_info": {"rounded_ratio": "16:9"},
                    }
                )

            visualizer = VisualizerAgent(
                exp_config=_diagram_config(work_dir, "ollama/image-model")
            )
            with self.assertRaisesRegex(ValueError, "text-only"):
                await visualizer.process(
                    {
                        "target_diagram_desc0": "Render a compact architecture diagram.",
                        "additional_info": {"rounded_ratio": "4:3"},
                    }
                )

            polish = PolishAgent(exp_config=_diagram_config(work_dir, "local/image-model"))
            with self.assertRaisesRegex(ValueError, "text-only"):
                await polish.process({})


if __name__ == "__main__":
    unittest.main()
