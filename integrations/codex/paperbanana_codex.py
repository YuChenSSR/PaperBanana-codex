#!/usr/bin/env python3
"""Run the author's PaperBanana pipeline using a local Codex subscription."""
from __future__ import annotations

import argparse
import asyncio
import base64
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from codex_backend import CodexBackend, require_subscription


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


async def doctor(args):
    import shutil
    from codex_backend import codex_settings
    home, cfg = codex_settings()
    codex = shutil.which('codex')
    status = await require_subscription(codex) if codex else 'Codex CLI missing'
    refs = {}
    for task in ('diagram', 'plot'):
        p = ROOT / 'data/PaperBananaBench' / task / 'ref.json'
        rows = json.loads(p.read_text()) if p.exists() else []
        missing = sum(not (p.parent / r['path_to_gt_image']).is_file() for r in rows)
        refs[task] = {'count': len(rows), 'missing_images': missing}
    checks = {m: importlib.util.find_spec(m) is not None for m in
              ('google.genai', 'anthropic', 'openai', 'PIL', 'numpy', 'matplotlib', 'gradio', 'streamlit', 'pyarrow')}
    info = {'author_repository': str(ROOT), 'codex': codex, 'auth': status,
            'model': cfg.get('model'), 'effort': cfg.get('model_reasoning_effort'),
            'references': refs, 'dependencies': checks,
            'subscription_entry': str(Path(__file__).resolve()),
            'native_image_verification': 'Requires a successful generated image; availability is checked at runtime.'}
    info['ok'] = bool(codex and 'ChatGPT' in status and all(checks.values()) and
                      all(r['count'] and not r['missing_images'] for r in refs.values()))
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0 if info['ok'] else 1


async def generate(args):
    method = Path(args.input).expanduser().resolve()
    if not method.is_file():
        raise ValueError(f'Input not found: {method}')
    if method.suffix.lower() == '.pdf':
        raise ValueError('Extract the relevant method section to a UTF-8 text/Markdown file before running the author pipeline.')
    output = Path(args.output).expanduser().resolve()
    if output.suffix.lower() != '.png':
        raise ValueError('Output must end in .png.')
    targets = [output] if args.candidates == 1 else [
        output.with_name(f'{output.stem}_{i+1:02d}.png') for i in range(args.candidates)]
    if any(p.exists() for p in targets):
        raise ValueError('An output already exists. Choose a new output filename.')
    ref_path = ROOT / 'data/PaperBananaBench' / args.task / 'ref.json'
    if args.retrieval != 'none' and not ref_path.is_file():
        raise ValueError(f'Reference dataset missing. Run {ROOT / ".venv/bin/python"} {ROOT / "scripts/prepare_bench.py"}')
    content = method.read_text(encoding='utf-8')
    if args.task == 'plot':
        content = json.loads(content)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
    run_dir = output.parent / f'{output.stem}_paperbanana_{stamp}'
    run_dir.mkdir(parents=True)
    backend = CodexBackend(run_dir, model=args.model, effort=args.effort, timeout=args.timeout)
    await require_subscription(backend.codex)
    from utils import generation_utils as gu
    gu.call_model_with_retry_async = backend.text
    gu.dispatch_image_generation = backend.image
    from utils.config import ExpConfig
    from utils.paperviz_processor import PaperVizProcessor
    from agents.planner_agent import PlannerAgent
    from agents.stylist_agent import StylistAgent
    from agents.visualizer_agent import VisualizerAgent
    from agents.critic_agent import CriticAgent
    from agents.retriever_agent import RetrieverAgent
    from agents.vanilla_agent import VanillaAgent
    from agents.polish_agent import PolishAgent
    cfg = ExpConfig(dataset_name='PaperBananaBench', task_name=args.task, split_name='demo',
                    exp_mode=args.mode, retrieval_setting=args.retrieval,
                    max_critic_rounds=args.rounds, main_model_name='codex/' + backend.model,
                    image_gen_model_name='codex-native-image', work_dir=ROOT)
    processor = PaperVizProcessor(exp_config=cfg,
        planner_agent=PlannerAgent(exp_config=cfg), stylist_agent=StylistAgent(exp_config=cfg),
        visualizer_agent=VisualizerAgent(exp_config=cfg), critic_agent=CriticAgent(exp_config=cfg),
        retriever_agent=RetrieverAgent(exp_config=cfg), vanilla_agent=VanillaAgent(exp_config=cfg),
        polish_agent=PolishAgent(exp_config=cfg))
    sha = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    manifest = {'status': 'running', 'upstream': 'https://github.com/dwzhu-pku/PaperBanana',
                'repository_commit': sha, 'auth': 'ChatGPT subscription via local Codex app-server',
                'adapter_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                   for name in ('paperbanana_codex.py', 'codex_backend.py')},
                'model': backend.model, 'effort': backend.effort, 'task': args.task,
                'mode': args.mode, 'retrieval': args.retrieval, 'critic_rounds': args.rounds,
                'input_file': str(method), 'caption': args.caption, 'outputs': []}
    save_json(run_dir / 'manifest.json', manifest)
    last_stage = None
    def progress(state):
        nonlocal last_stage
        stage = state.get('current_stage', state.get('stage', str(state)))
        if stage != last_stage:
            print('[PaperBanana]', stage, flush=True)
            last_stage = stage
    try:
        for i, target in enumerate(targets):
            data = {'filename': f'codex_candidate_{i+1}', 'candidate_id': i,
                    'content': content, 'caption': args.caption, 'visual_intent': args.caption,
                    'additional_info': {'rounded_ratio': args.aspect_ratio, 'image_size': args.image_size},
                    'max_critic_rounds': args.rounds}
            result = None
            async for item in processor.process_queries_batch([data], max_concurrent=1,
                                                              do_eval=False, progress_callback=progress):
                result = item
            if result is None or result.get('error'):
                raise RuntimeError(f'Author pipeline failed: {result}')
            key = result.get('eval_image_field')
            encoded = result.get(key) if key else None
            if not encoded:
                raise RuntimeError('Author pipeline produced no final image; see call logs.')
            native = backend.native_images.get(hashlib.sha256(encoded.encode()).hexdigest())
            if native:
                target.write_bytes(native)
            else:
                from PIL import Image
                from io import BytesIO
                with Image.open(BytesIO(base64.b64decode(encoded))) as im:
                    im.save(target, format='PNG')
            # Store the actual author outputs, replacing embedded images with files.
            compact = {}
            for k, v in result.items():
                if k.endswith('_base64_jpg') and v:
                    img_path = run_dir / f'candidate_{i+1}_{k[:-11]}.jpg'
                    img_path.write_bytes(base64.b64decode(v))
                    compact[k] = {'image_path': str(img_path)}
                else:
                    compact[k] = v
            save_json(run_dir / f'candidate_{i+1}_result.json', compact)
            manifest['outputs'].append(str(target))
            print('SAVED', target, flush=True)
        manifest['status'] = 'completed'
    except BaseException as exc:
        manifest['status'] = 'failed'
        manifest['error'] = str(exc)
        raise
    finally:
        manifest['backend_calls'] = backend.calls
        save_json(run_dir / 'manifest.json', manifest)
    print(json.dumps({'outputs': manifest['outputs'], 'manifest': str(run_dir / 'manifest.json')}, indent=2))
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('doctor')
    g = sub.add_parser('generate')
    g.add_argument('--input', required=True)
    g.add_argument('--caption', required=True)
    g.add_argument('--output', required=True)
    g.add_argument('--task', choices=['diagram', 'plot'], default='diagram')
    g.add_argument('--rounds', type=int, choices=range(0, 6), default=1)
    g.add_argument('--candidates', type=int, choices=range(1, 21), default=1)
    g.add_argument('--mode', choices=['demo_full', 'demo_planner_critic'], default='demo_full')
    g.add_argument('--retrieval', choices=['auto', 'random', 'none'], default='auto')
    g.add_argument('--aspect-ratio', default='16:9')
    g.add_argument('--image-size', choices=['1k', '2k', '4k'], default='2k')
    g.add_argument('--model', default=None, help='Default: inherit the user Codex model')
    g.add_argument('--effort', default=None, help='Default: inherit the user Codex reasoning effort')
    g.add_argument('--timeout', type=int, default=900, help='Per-model-call timeout, seconds')
    args = p.parse_args()
    try:
        return asyncio.run(doctor(args) if args.command == 'doctor' else generate(args))
    except (Exception, KeyboardInterrupt) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
