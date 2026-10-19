"""Behavior checks for the subscription boundary and native image contract."""
import asyncio
import base64
from io import BytesIO
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from codex_backend import (BackendError, CodexBackend, image_bytes_from_item,
                           normalized_inputs, pack_references, retrieval_parts, subscription_env)


def sample_png():
    data = BytesIO()
    Image.new('RGB', (4, 3), 'white').save(data, format='PNG')
    return data.getvalue()


class BackendTests(unittest.TestCase):
    def test_large_retrieval_keeps_every_reference_under_request_limit(self):
        prefix = '**Target Input**\nA method\n**Candidate Pool**\n'
        blocks = [f'Candidate Diagram {i}:\n- Diagram ID: ref_{i}\n- Method: ' + 'x' * 600 + '\n'
                  for i in range(30)]
        suffix = '\nNow, based on the Target Input and the Candidate Pool, select ten.'
        head, candidates, tail = retrieval_parts(prefix + ''.join(blocks) + suffix)
        batches = pack_references(head, candidates, tail, 5000)
        self.assertGreater(len(batches), 1)
        self.assertEqual([k for batch in batches for k in batch], list(candidates))
        self.assertTrue(all(len(head + ''.join(batch.values()) + tail) <= 5000 for batch in batches))
        self.assertEqual(head + ''.join(candidates.values()) + tail, prefix + ''.join(blocks) + suffix)

    def test_interleaved_references_preserve_order_and_pixels(self):
        raw = sample_png()
        b64 = base64.b64encode(raw).decode()
        contents = [{'type': 'text', 'text': 'Reference 1'},
                    {'type': 'image', 'image_base64': b64},
                    {'type': 'text', 'text': 'Target'},
                    {'type': 'image', 'source': {'data': b64}}]
        with tempfile.TemporaryDirectory() as tmp:
            actual = normalized_inputs(contents, Path(tmp))
            self.assertEqual([x['type'] for x in actual], ['text', 'localImage', 'text', 'localImage'])
            self.assertEqual(actual[2]['text'], 'Target')
            with Image.open(actual[1]['path']) as im:
                self.assertEqual(im.size, (4, 3))

    def test_prose_and_failed_native_calls_cannot_masquerade_as_images(self):
        for item in ({'type': 'agentMessage', 'text': '/tmp/picture.png'},
                     {'type': 'imageGeneration', 'status': 'failed', 'result': ''}):
            with self.assertRaises(BackendError):
                image_bytes_from_item(item)

    def test_native_base64_output_is_validated(self):
        raw = sample_png()
        self.assertEqual(image_bytes_from_item({'type': 'imageGeneration', 'status': 'completed',
                                               'result': base64.b64encode(raw).decode()}), raw)
        with self.assertRaises(Exception):
            image_bytes_from_item({'type': 'imageGeneration', 'status': 'completed',
                                   'result': base64.b64encode(b'not an image').decode()})

    def test_api_keys_are_removed_only_from_child_environment(self):
        with patch.dict(os.environ, {'CODEX_API_KEY': 'test', 'OPENAI_API_KEY': 'test', 'PAPERBANANA_TEST': 'kept'}):
            child = subscription_env()
            self.assertNotIn('CODEX_API_KEY', child)
            self.assertNotIn('OPENAI_API_KEY', child)
            self.assertEqual(child['PAPERBANANA_TEST'], 'kept')
            self.assertEqual(os.environ['OPENAI_API_KEY'], 'test')

    def test_malformed_critic_output_cannot_be_treated_as_approval(self):
        class DummyConfig:
            system_instruction = 'Return JSON with "critic_suggestions" and "revised_description".'
        with tempfile.TemporaryDirectory() as tmp:
            backend = CodexBackend(tmp, model='test-model')
            async def invalid(*args, **kwargs):
                return 'not valid critic JSON'
            backend._infer = invalid
            with self.assertRaises(BackendError):
                asyncio.run(backend.text('test-model', [], DummyConfig()))


if __name__ == '__main__':
    unittest.main()
