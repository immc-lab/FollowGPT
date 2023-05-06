# 😎 **ChatEverything**

![ChatEveryThing](./images/Chat%20Everthing.png)

<img src="https://awesome.re/badge.svg"> <img src="https://img.shields.io/badge/lang-%E4%B8%AD%E6%96%87-red"> <img src="https://img.shields.io/badge/lang-En-red">

<img src="https://img.shields.io/github/stars/immc-lab/ChatEverything.svg"> <img src="https://img.shields.io/github/watchers/immc-lab/ChatEverything.svg">

---

随着深度学习的发展，大模型不断的涌现。大模型可以是深度学习模型中参数数量巨大的模型。最近来的一些工作，通过结合强化学习的相关基础，通过 RLHF 技术，将大模型的知识做了有效引导，使大模型发挥了惊人的效果。然而，现在互联网上，缺乏系统的，有逻辑的大模型资源整合文档，阻碍了许多人探究大模型技术的热情，因此，我们对现有网络上的资源进行了整合，以提供一个清晰的富有逻辑的脉络。

如果您发现这个库对您有帮助的话，请点点您可爱小手给我们 ⭐ 或者 Sharing ⬆️

With the development of deep learning, large models continue to emerge. Large models can be models with a huge number of parameters in deep learning models. Recently, some work has effectively guided the knowledge of large models by combining the relevant foundation of reinforcement learning through the RLHF technique, allowing large models to achieve amazing results. However, there is currently a lack of systematic and logical integration of large model resources on the internet, which hinders the enthusiasm of many people to explore large model technology. Therefore, we have integrated the existing resources on the network to provide a clear and logical context.

If you find this repository helpful, please give us a ⭐ or share it ⬆️.

## News

```
2023.05.06 add
```

## 内容

-   [学术&产业](#学术&产业)
    -   [上游](#上游)
        -   [基座模型](#基座模型)
        -   [模型微调](#模型微调)
        -   [模型评估](#模型评估)
    -   [中游](#中游)
        -   [量化](#量化)
        -   [拓展](#拓展)
        -   [与其他模型的结合](#与其他模型的结合)
        -   [二次开发](#二次开发)
    -   [下游](#下游)
        -   [Web 应用](#Web应用)
        -   [小程序](#小程序)
        -   [聊天机器人](#聊天机器人)
-   [教程](#教程)
    -   [视频教程](#视频教程)
    -   [博客](#博客)
-   [资源](#资源)
    -   [工具](#工具)
    -   [免费 API](#免费API)
    -   [镜像网站](#镜像网站)

## 学术&产业

### 上游

#### 基座模型

现有的大模型训练耗时耗力，往往需要大规模算力。训练成本远非常人能够承受，所以现在一版将大模型成为基座模型（basement model），以供下游使用。

##### 文本大模型

接受文本作为输入，进行文本生成

-   BELLE - 开源中文对话大模型，由 LLaMA 微调而来
    -   [open source] [[code]](https://github.com/LianjiaTech/BELLE)
-   ChatGLM - ChatGLM-6B 是一个开源的、支持中英双语的对话语言模型，基于 General Language Model (GLM) 架构，具有 62 亿参数。
    -   [open source] [[code]](https://github.com/THUDM/ChatGLM-6B)
-   Phoenix - 基于 LLaMA 微调的模型。
    -   [open source] [[code]](https://github.com/FreedomIntelligence/LLMZoo)
-   MOSS - MOSS 是一个支持中英双语和多种插件的开源对话语言模型，moss-moon 系列模型具有 160 亿参数，在 FP16 精度下可在单张 A100/A800 或两张 3090 显卡运行，在 INT4/8 精度下可在单张 3090 显卡运行。MOSS 基座语言模型在约七千亿中英文以及代码单词上预训练得到，后续经过对话指令微调、插件增强学习和人类偏好训练具备多轮对话能力及使用多种插件的能力。
    -   [open source] [[code]](https://github.com/OpenLMLab/MOSS)
-   Alpaca - 由 LLaMa 微调而来的大模型，训练语料共包含 52k 数据。
    -   [open source] [[code]](https://github.com/tatsu-lab/stanford_alpaca)
-   pandallm - 海外中文开源大语言模型，基于 Llama-7B, -13B, -33B, -65B 进行中文领域上的持续预训练。

    -   [open source] [[code]](https://github.com/dandelionsllm/pandallm)

-   ChatGPT-3.5 - OpenAI 文本大模型，闻名于世界的 ChatGPT
    -   [close source] [[website]](https://chat.openai.com)
-   Claude - 从 OpenAI 出走的部分人马原版打造的大模型
    -   [close source] [[website]](https://www.anthropic.com/index/introducing-claude)
-   通意千问 -
    -   [close source] [[website]](https://tongyi.aliyun.com/)
-   星火认知
    -   [close source] [[website]](https://xinghuo.xfyun.cn/)

##### 多模态大模型

能够接受图片、文本作为输入，进行文本生成

-   <img src="https://img.shields.io/badge/opensource-No-red">GPT-4 - GPT-4 是一个大型多模态模型（接受图像和文本输入，发出文本输出），虽然在许多现实世界场景中的能力不如人类，但在各种专业和学术基准上表现出人类水平的表现。
    以下是一些相关论文： 

-   <img src="https://img.shields.io/badge/opensource-No-red">Bard - 谷歌正在开发一种名为 LaMDA 的对话语言模型，该模型被设计为与 ChatGPT 相似，但是目前只支持英文对话，并且仅限于美国和英国的用户进行预约访问。除此之外的信息目前尚不明确。
-   <img src="https://img.shields.io/badge/opensource-Yes-green">LLaVA🌋 - LLaVA 是一种新型的端到端训练的大型多模态模型，它结合了视觉编码器和 Vicuna 来实现通用的视觉和语言理解。该模型能够模仿多模态 GPT-4 的精神，具有令人印象深刻的聊天功能，并引入了科学质量检查的艺术准确性作为新的标准。
    -   [[arXiv]](https://arxiv.org/abs/2304.08485) [[code]](https://github.com/haotian-liu/LLaVA) [[Demo]](https://llava.hliu.cc/) [[Dataset]](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) [[Model]](https://huggingface.co/liuhaotian/LLaVA-13b-delta-v0)
-   <img src="https://img.shields.io/badge/opensource-No-red">DALL·E2 - DALL·E 2 是一种人工智能系统，它可以根据自然语言描述创作出逼真的图像和艺术作品。
-   <img src="https://img.shields.io/badge/opensource-No-red">Wenxin -（文心一言）百度全新一代知识增强大语言模型，文心大模型家族的新成员，能够与人对话互动，回答问题，协助创作，高效便捷地帮助人们获取信息、知识和灵感。
-   <img src="https://img.shields.io/badge/opensource-Yes-green">MiniGPT - Mini GPT-4 是一个基于 PyTorch 实现的小型自然语言处理模型，它使用了类似于 GPT-3 的架构，但参数数量只有 GPT-3 的一小部分。Mini GPT-4 在多个自然语言处理任务上表现出色，包括语言建模、文本生成和问答系统等。
    -   [[arXiv]](https://arxiv.org/abs/2304.10592) [[code]](https://github.com/Vision-CAIR/MiniGPT-4) [[Demo]](https://16440e488436f49d99.gradio.live/) [[Dataset]](https://huggingface.co/datasets/Vision-CAIR/cc_sbu_align) [[Model]](https://huggingface.co/Vision-CAIR/MiniGPT-4)
-   <img src="https://img.shields.io/badge/opensource-Yes-green">mPLUG-Owl🦉 - mPLUG-Owl，一种新颖的训练范式，通过基础 LLM、视觉知识模块和视觉抽象模块的模块化学习，使 LLM 具备多模态能力。
    -   [[arXiv]](https://arxiv.org/abs/2304.14178) [[code]](https://github.com/x-plug/mplug-owl) [[Demo]](https://modelscope.cn/studios/damo/mPLUG-Owl/summary)

#### 相关论文

-   [Sparks of Artificial General Intelligence: Early experiments with GPT-4](https://arxiv.org/pdf/2303.12712.pdf)
-   [DeID-GPT: Zero-shot Medical Text De-Identification by GPT-4](https://arxiv.org/pdf/2303.11032.pdf)

#### 模型微调

##### PHT-LoRA

-   Chinese-LLaMA-Alpaca - https://github.com/LC1332/Chinese-alpaca-lora
-   Yaya - https://github.com/qiyuan-chen/Yaya-Moss-Alpaca-LoRA

##### PHT-adapter

#### 模型评估

### 中游

#### 量化

-   llama.cpp - https://github.com/ggerganov/llama.cpp

#### 拓展

#### 与其他模型的结合

#### 二次开发

### 下游

#### Web 应用

#### 小程序

#### 聊天机器人

## 教程

### 视频教程

### 博客

## 资源

### 工具

-   peft - https://github.com/huggingface/peft

### 免费 API

-   GPTFree - https://github.com/xtekky/gpt4free

### 镜像网站

## Contributions

<p align="center"><a href="https://github.com/huaiwen"><img src="https://avatars.githubusercontent.com/u/3187529?v=4" width="50px" alt="robinicus" /></a>&nbsp;&nbsp;<a href="https://github.com/YangYang"><img src="https://avatars.githubusercontent.com/u/17808880?v=4" width="50px" alt="0xmatchmaker" /></a>&nbsp;&nbsp;<a href="https://github.com/guozihang"><img src="https://avatars.githubusercontent.com/u/17142416?v=4" width="50px" alt="0xmatchmaker" /></a>&nbsp;&nbsp;<a href="https://github.com/LeeRoc-China"><img src="https://avatars.githubusercontent.com/u/59104898?s=400&u=c225a082a6a410e3d7c84ca29a07d723d7308dca&v=4" width="50px" alt="0xmatchmaker" /></a>&nbsp;&nbsp;</p>
