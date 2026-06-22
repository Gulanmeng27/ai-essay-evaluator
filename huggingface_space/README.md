---
title: AI Essay Feedback LT6582
emoji: 📝
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

# AI Essay Feedback (LT6582 capstone demo)

Live demo of the essay-feedback system: real RAG (local BGE-small embeddings + FAISS over a
knowledge base) and an Agentic Workflow (4 dimension experts → self-correction → compose),
backed by DeepSeek. Runs on free CPU — no GPU.

## 部署步骤(在你个人设备上做)

1. 登录 https://huggingface.co → 右上 **New** → **Space**。
2. Space name 随便取;**SDK 选 Gradio**;硬件选免费的 **CPU basic**;创建。
3. 进入这个 Space → **Files** 标签 → **Add file → Upload files**,把本文件夹里的全部内容上传:
   - `app.py`
   - `requirements.txt`
   - `README.md`(本文件,头部的 YAML 是 Space 配置,必须保留)
   - `knowledge_base/` 整个文件夹(3 个 .md)
4. **设置密钥**:Space 页面 → **Settings** → **Variables and secrets** → **New secret**:
   - Name: `DEEPSEEK_API_KEY`
   - Value: 你的 DeepSeek key(`sk-...`)
   - ⚠️ 一定用 **Secret**(不是 Variable),别写进代码。
5. Space 会自动开始 build(装依赖 + 下载 embedding 模型,首次几分钟)。完成后就有一个公开网址,答辩点开即可演示。

## 注意
- 免费 CPU 版会休眠;第一次打开/冷启动要等十几秒。
- 本知识库为占位版;黄小倩的正式知识库做好后,替换 `knowledge_base/` 里的文件再重新上传即可。
- 本地试跑:`pip install -r requirements.txt gradio && DEEPSEEK_API_KEY=sk-... python app.py`
