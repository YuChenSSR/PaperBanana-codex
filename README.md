<h1 align="center">PaperBanana-Codex 🍌</h1>

<p align="center">
  <strong>从方法说明，到参考驱动、可迭代修改的论文插图。</strong><br>
  PaperBanana 作者版工作流 · Codex / ChatGPT 订阅接入
</p>

<p align="center">
  <a href="#quick-start">快速安装</a> ·
  <a href="#use-in-codex">在 Codex 中使用</a> ·
  <a href="integrations/codex/README.md">English guide</a> ·
  <a href="README.upstream.md">原作者项目</a>
</p>

本项目基于第一作者维护的 [dwzhu-pku/PaperBanana](https://github.com/dwzhu-pku/PaperBanana)，
保留作者核心实现，通过独立适配器接入本机 Codex 的 ChatGPT 登录。

## 工作方式

[![PaperBanana-Codex 概念总览：参考驱动的规划，以及 Visualizer 与 Critic 组成的迭代修改回路](assets/paperbanana-codex-overview.png)](assets/paperbanana-codex-overview.png)

以方法图为例：先检索参考、规划内容并调整风格，再生成草图，通过视觉审查迭代修改。
[概念图设计与来源](docs/figures/overview-visual.md) · [生成记录](docs/figures/workflow-generation.json)

| 参考驱动规划 | 原生图片生成 | 视觉反馈迭代 |
| --- | --- | --- |
| 从 PaperBananaBench 选取参考，组织模块、关系和布局。 | 使用 Codex 原生绘图工具呈现设计；保留草图和中间结果。 | Critic 对照原始内容检查图片，给出具体修改意见。 |

默认继承 Codex 中配置的模型和推理档位，订阅入口无需另配模型 API Key。
模型与绘图权限、可用额度取决于登录账号。

<details>
<summary><strong>展开完整工程流程：输入、模型后端、渲染分支与输出记录</strong></summary>

[![PaperBanana-Codex 完整工程流程](assets/paperbanana-codex-workflow-upstream-style.png)](assets/paperbanana-codex-workflow-upstream-style.png)

工程图包含方法图与统计图两种渲染路径，以及独立保留的原版 API 入口。
它通过本仓库的 `$paperbanana` 订阅流程生成，随后参照原作者配色进行了原生图片修订。
完整工程图与上方重新设计的概念图用途不同，两者分别保留。

[详细适配说明](integrations/codex/README.md) ·
[工程图绘图说明](docs/figures/workflow-method.md) ·
[配色修订](docs/figures/workflow-style-revision.md)

</details>

<a id="quick-start"></a>

## 快速安装（macOS / Linux）

准备 Git、[uv](https://docs.astral.sh/uv/getting-started/installation/)、Python 3.12，
以及支持 app-server 和原生图片生成的 Codex CLI。本适配器已在 Codex CLI 0.163.0 上验证。

```bash
git clone https://github.com/YuChenSSR/PaperBanana-codex.git
cd PaperBanana-codex

uv python install 3.12
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt

# 下载参考图库
.venv/bin/python scripts/prepare_bench.py --tasks diagram plot

# 安装调用命令和 Codex 技能
.venv/bin/python integrations/codex/install.py
```

确认 `codex login status` 显示通过 ChatGPT 登录。若尚未登录，运行 `codex login`
完成登录，然后检查安装：

```bash
~/.local/bin/paperbanana-author doctor
```

| 入口 | 名称 / 位置 |
| --- | --- |
| Codex 技能 | `$paperbanana` |
| 技能目录 | `$CODEX_HOME/skills/paperbanana`，默认 `~/.codex/skills/paperbanana` |
| 命令行工具 | `~/.local/bin/paperbanana-author` |
| 适配器源码 | `integrations/codex/` |

安装器会将旧的 `paperbanana-author` 技能迁移为 `paperbanana`，并备份它自己的旧文件。
安装后请保留仓库及 `.venv`，因为调用命令指向这个环境。

<a id="use-in-codex"></a>

## 在 Codex 中使用

新开一个 Codex 对话，提供方法说明或论文文件，然后输入：

> 用 $paperbanana 根据这份方法说明生成论文架构图，使用订阅模式。

也可以明确要求模块、语言、布局和修改轮数：

> 用 $paperbanana 画这个方法的流程图，英文标签、横向布局，保留所有模块与箭头关系，生成一个候选并检查修改一轮。

技能会准备输入、调用作者流程、检查最终图片，并给出输出路径。
PDF 需要先由 Codex 提取相关方法文本；适配器本身接收 UTF-8 文本或 Markdown。

## 命令行使用

将方法说明保存为 `method.md`，在自己的论文项目目录运行：

```bash
~/.local/bin/paperbanana-author generate \
  --input method.md \
  --caption "Overview of the proposed method" \
  --output figures/method.png \
  --rounds 1
```

如果 `~/.local/bin` 已在 PATH 中，可以直接使用 `paperbanana-author`。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--mode` | `demo_full` | 完整流程，包含风格调整 |
| `--retrieval` | `auto` | 自动检索参考图 |
| `--candidates` | `1` | 候选数量，依次生成 |
| `--rounds` | `1` | Critic 修改轮数；`0` 跳过审查 |
| `--aspect-ratio` | `16:9` | 请求的宽高比 |
| `--image-size` | `2k` | 请求的分辨率级别 |
| `--model` / `--effort` | 继承 Codex 配置 | 显式选择模型或推理档位 |

完整流程可能需要数分钟，参考检索和多轮出图会消耗订阅额度。
更多参数可用 `paperbanana-author generate --help` 查看。

输出包括最终 PNG，以及同目录下的 `<名称>_paperbanana_<时间>/` 运行目录，
其中保留参考图选择、规划描述、Critic 意见、模型调用记录和原生图片。
已有输出不会被覆盖；继续修改时使用新文件名。

统计图入口为 `--task plot`，输入应为 JSON 数据。它使用作者的 Matplotlib
代码生成流程，目前尚未完成与方法图相同的端到端实测。

## 适配范围与限制

- **作者流程与模型接入**：保留作者核心代码，在独立包装器中替换模型调用入口。
- **长参考库输入**：Codex 单次输入限制为 1 MiB。适配器分批评估作者原定参考范围，
  再对入选参考统一筛选；这可能与原始单次检索产生不同结果。
- **模型与画质**：文字模型和图片生成后端已发生变化，因此本项目的输出不代表原论文复现指标。
- **输出尺寸**：宽高比和分辨率作为绘图要求传入，原生工具不保证精确像素尺寸。
- **原版网页入口**：直接运行 `app.py`、`demo.py`、`main.py` 或 `skill/run.py`
  仍走原版 API 配置。订阅模式请使用 `$paperbanana` 或 `paperbanana-author`。
- **协议兼容性**：app-server 及部分调用配置会随 Codex 版本变化，更新后建议先运行小样例。

详细实现、诊断及原生图片验证方式见 [适配说明](integrations/codex/README.md)。

## 已完成的验证

2026-10-19，在 Codex CLI 0.163.0、ChatGPT 登录及 `gpt-6-astra / xhigh` 下：

- 8 项适配器测试通过，包括实际 Codex 后端启动检查。
- 完整方法图样例从作者的 200 条参考候选中选出 10 条。
- 完成 8 次文字/视觉调用、3 次原生图片生成，包含一轮 Critic 驱动修改。
- 检查了最终图片的三阶段结构、标签和箭头方向。

这是功能连通性验证，尚未进行系统性的图像质量对比。

```bash
.venv/bin/python -m unittest discover -s integrations/codex -p 'test_*.py' -v
~/.local/bin/paperbanana-author doctor
```

## 原作者、论文与许可

[PaperBanana 论文](https://arxiv.org/abs/2601.23265)和原始实现来自 Dawei Zhu、Rui Meng、Yale Song、Xiyu Wei、
Sujian Li、Tomas Pfister 和 Jinsung Yoon。作者版承接了 Google Research 的
[PaperVizAgent](https://github.com/google-research/papervizagent)。
本 fork 独立维护 Codex 接入，不代表原作者、Google 或 OpenAI 的官方支持。

原作者完整的功能说明、演示、贡献者名单、引用格式及声明保留在
[README.upstream.md](README.upstream.md)。项目许可见 [LICENSE](LICENSE)。
