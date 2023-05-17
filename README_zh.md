# 😎 **FollowGPT**

![ChatEveryThing](./images/Chat%20Everthing.png)

<img src="https://awesome.re/badge.svg"> <img src="https://img.shields.io/badge/lang-%E4%B8%AD%E6%96%87-red"> <img src="https://img.shields.io/badge/lang-En-red">

<img src="https://img.shields.io/github/stars/immc-lab/ChatEverything.svg"> <img src="https://img.shields.io/github/watchers/immc-lab/ChatEverything.svg">

---

随着深度学习的发展，大模型不断的涌现。大模型可以是深度学习模型中参数数量巨大的模型。最近来的一些工作，通过结合强化学习的相关基础，通过 RLHF 技术，将大模型的知识做了有效引导，使大模型发挥了惊人的效果。然而，现在互联网上，缺乏系统的，有逻辑的大模型资源整合文档，阻碍了许多人探究大模型技术的热情，因此，我们对现有网络上的资源进行了整合，以提供一个清晰的富有逻辑的脉络。

如果您发现这个库对您有帮助的话，请点点您可爱小手给我们 ⭐ 或者 Sharing ⬆️

## 新闻📰

```
2023.05.06 add
```

---

## 内容

-   [了解大模型](#学术&产业)
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
-   [教程](#教程)
    -   [视频](#视频)
    -   [博客](#博客)
-   [资源](#资源)
    -   [工具](#工具)
    -   [API](#API)
    -   [镜像网站](#镜像网站)
-   [Contributions](#Contributions)

---

## 了解大模型

### 上游

#### 基座模型

现有的大模型训练耗时耗力，往往需要大规模算力。训练成本远非常人能够承受，所以现在一版将大模型成为基座模型（basement model），以供下游使用。

##### 文本大模型

文本是最早产生预训练和相关微调技术的领域。同时现在，文本大模型引领时代的步伐，启发其他领域的任务。文本大模型通常指参数量巨大（达到数亿级别参数的模型）。而目前，生成式文本大模型最为流行，其范式为接受文本作为输入，进行文本生成。

-   [BELLE ]()  - [[code]](https://github.com/LianjiaTech/BELLE) - 开源中文对话大模型，由 LLaMA 微调而来
-   [ChatGLM]()  - [[arXiv]](https://arxiv.org/abs/2210.02414) [[code]](https://github.com/THUDM/ChatGLM-6B) - ChatGLM-6B 是一个开源的、支持中英双语的对话语言模型，具有 62 亿参数。
-   [Phoenix]()   - [[TechReport]](https://github.com/FreedomIntelligence/LLMZoo/blob/main/assets/llmzoo.pdf) [[code]](https://github.com/FreedomIntelligence/LLMZoo) - 基于 BLOOMZ 微调的模型。
-   [MOSS]()  - [[code]](https://github.com/OpenLMLab/MOSS) - MOSS 是一个支持中英双语和多种插件的开源对话语言模型，moss-moon 系列模型具有 160 亿参数。
-   [Alpaca]()  - [[Blog]](https://crfm.stanford.edu/2023/03/13/alpaca.html) [[code]](https://github.com/tatsu-lab/stanford_alpaca) - 由 LLaMa 微调而来的大模型，训练语料共包含 52k 数据。
-   [pandallm ]() - [[ArXiv]](https://arxiv.org/pdf/2305.03025) [[code]](https://github.com/dandelionsllm/pandallm) - 海外中文开源大语言模型，基于 Llama-7B, -13B, -33B, -65B 进行中文领域上的持续预训练。
-   [Latin Phoenix: Chimera]()  - [[code]](https://github.com/dandelionsllm/pandallm) - 基于 LLaMA 微调的模型。
-   [Dolly]()  - [[Blog]](https://www.databricks.com/blog/2023/03/24/hello-dolly-democratizing-magic-chatgpt-open-models.html) [[code]](https://github.com/databrickslabs/dolly) - Databricks 的 Dolly，一个在 Databricks 机器学习平台上训练的大型语言模型。
-   [Guanaco]()  - [[code]](https://github.com/Guanaco-Model/Guanaco-Model.github.io) [[Dataset]](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset) - 基于 LLaMA 微调的模型。
-   [Luotuo]()  - [[code]](https://github.com/LC1332/Luotuo-Chinese-LLM) - 基于 LLaMA 微调的模型
-   [ChatYuan]()  - [[article]](https://mp.weixin.qq.com/s/-axa6XcjGl_Koeq_OrDq8w) [[code]](https://github.com/clue-ai/ChatYuan) [[Demo]](https://modelscope.cn/studios/ClueAI/ChatYuan-large-v2) - 首个中文版 ChatGPT。支持中英双语交互等多种新功能。
-   [ChatGPT-3.5]()  - [[website]](https://chat.openai.com) - OpenAI 文本大模型，闻名于世界的 ChatGPT。
-   [Claude]()  - [[website]](https://www.anthropic.com/index/introducing-claude) - 从 OpenAI 出走的部分人马原版打造的大模型
-   [通意千问]()  - [[website]](https://tongyi.aliyun.com/) - 阿里巴巴开源的大模型
-   [星火认知]()  - [[website]](https://xinghuo.xfyun.cn/) - 科大讯飞开源的大模型

##### 文本大模型对比

| 模型                     |   Backbone   |    支持语言    |   参数数量级   | 模型开源 |                  机构                   | 数据开源 |  发布时间  | 评价 |                            Stars                             |
| ------------------------ | :----------: | :------------: | :------------: | :------: | :-------------------------------------: | :------: | :--------: | :--: | :----------------------------------------------------------: |
| ChatGPT                  |      -       |     multi      |       -        |    x     |                 OpenAI                  |    x     | 2022-11-30 |      |                              -                               |
| Bard                     |      -       |       -        |      137B      |    x     |                 Google                  |    x     | 2023-02-06 |      |                              -                               |
| Claude                   |      -       |     zh, en     |      52B       |    x     |                    -                    |    x     | 2023-03-14 |      |                              -                               |
| ERNIE Bot(Wenxin)        |      -       |       zh       |      260B      |    x     |                  Baidu                  |    x     | 2023-03-16 |      |                              -                               |
| Tongyi Qianwen           |    TongYi    |     zh, en     |      ~10T      |    x     |                 Alibaba                 |    x     | 2023-04-07 |      |                              -                               |
| SparkDesk(Xinghuorenzhi) |      -       |     zh, en     |       -        |    x     |                 iFLYTEK                 |    x     | 2023-0506  |      |                              -                               |
| LLaMA                    |      -       |     multi      | 7B/13B/33B/65B |    √     |                 Meta AI                 |    √     | 2023-02-24 |      | ![img](https://img.shields.io/github/stars/facebookresearch/llama) |
| BELLE                    | BLOOMZ/LLaMA |       zh       |       7B       |    √     |                    -                    |    √     | 2023-03-26 |      | ![img](https://img.shields.io/github/stars/LianjiaTech/BELLE) |
| ChatGLM                  |     GLM      |     zh, en     |       6B       |    √     |           Tsinghua University           |    x     | 2023-03-16 |      | ![img](https://img.shields.io/github/stars/THUDM/ChatGLM-6B) |
| MOSS                     |      -       |     zh, en     |      16B       |    √     |            Fudan University             |    √     | 2023-04-21 |      |  ![img](https://img.shields.io/github/stars/OpenLMLab/MOSS)  |
| Alpace                   |    LLaMA     |       en       |       7B       |    √     |           Stanford NLP Group            |    √     | 2023-03-13 |      | ![img](https://img.shields.io/github/stars/tatsu-lab/stanford_alpaca) |
| pandallm                 |    LLaMA     |     zh, en     |     7B/13B     |    √     |    Nanyang Technological University     |    √     | 2023-05-04 |      | ![img](https://img.shields.io/github/stars/dandelionsllm/pandallm) |
| Phoenix                  |    BLOOMZ    |     multi      |       7B       |    √     | Shenzhen Research Institute of Big Data |    √     | 2023-04-08 |      | ![img](https://img.shields.io/github/stars/FreedomIntelligence/LLMZoo) |
| Latin Phoenix: Chimera   |    LLaMA     |     multi      |     7/13B      |    √     | Shenzhen Research Institute of Big Data |    √     | 2023-04-08 |      | ![img](https://img.shields.io/github/stars/dandelionsllm/pandallm) |
| Dolly                    |    GPT-J     |       en       |       6B       |    √     |                    -                    |    √     | 2023-03-24 |      | ![img](https://img.shields.io/github/stars/databrickslabs/dolly) |
| Guanaco                  |    LLaMA     | zh ,en, ja, de |       7B       |    √     |                    -                    |    √     | 2023-03-26 |      | ![img](https://img.shields.io/github/stars/Guanaco-Model/Guanaco-Model.github.io) |
| Luotuo                   |    LLaMA     |       zh       |       7B       |    √     |                    -                    |    √     | 2023-03-31 |      | ![img](https://img.shields.io/github/stars/LC1332/Luotuo-Chinese-LLM) |
| ChatYuan                 |      -       |     zh, en     |     ～10B      |    √     |                YuanYu.AI                |    x     | 2023-03-23 |      | ![img](https://img.shields.io/github/stars/clue-ai/ChatYuan) |

##### 多模态大模型

能够接受图片、文本作为输入，进行文本生成

-   [LLaVA🌋]()  - [[arXiv]](https://arxiv.org/abs/2304.08485) [[code]](https://github.com/haotian-liu/LLaVA) [[Demo]](https://llava.hliu.cc/) [[Dataset]](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) [[Model]](https://huggingface.co/liuhaotian/LLaVA-13b-delta-v0) - LLaVA 是一种新型的端到端训练的大型多模态模型，它结合了视觉编码器和 Vicuna 来实现通用的视觉和语言理解。
-   [MiniGPT]()  - [[arXiv]](https://arxiv.org/abs/2304.10592) [[code]](https://github.com/Vision-CAIR/MiniGPT-4) [[Demo]](https://16440e488436f49d99.gradio.live/) [[Dataset]](https://huggingface.co/datasets/Vision-CAIR/cc_sbu_align) [[Model]](https://huggingface.co/Vision-CAIR/MiniGPT-4) - 是一个小型自然语言处理模型，它使用了类似于 GPT-3 的架构，但参数数量只有 GPT-3 的一小部分。
-   [mPLUG-Owl🦉]() - [[arXiv]](https://arxiv.org/abs/2304.14178) [[code]](https://github.com/x-plug/mplug-owl) [[Demo]](https://modelscope.cn/studios/damo/mPLUG-Owl/summary) - 一种新颖的训练范式，通过基础 LLM、视觉知识模块和视觉抽象模块的模块化学习，使 LLM 具备多模态能力。
-   [GPT-4]()  - [[TechReport]](https://arxiv.org/abs/2303.08774) [[website]](https://openai.com/product/gpt-4) - GPT-4 是一个大型多模态模型，各种专业和学术基准上表现出人类水平的表现。

-   [Bard]()  - [[website]](https://bard.google.com/) - 谷歌开发一种名为 LaMDA 的对话语言模型，该模型设计与 ChatGPT 相似。

-   [PaLM2]()  - [[TechReport]](https://ai.google/static/documents/palm2techreport.pdf) [[website]](https://ai.google/discover/palm2) - 谷歌提出的 PaLM 二代模型，对标GPT-4 ，改进了数学、代码、推理、多语言翻译和自然语言生成能力。
  
-   [DALL·E2]()  - [[website]](https://openai.com/product/dall-e-2) - DALL·E 2 是一种人工智能系统，它可以根据自然语言描述创作出逼真的图像和艺术作品。

-   [Wenxin]()  - [[website]](https://yiyan.baidu.com/welcome) -（文心一言）百度全新一代知识增强大语言模型。
-   [DetGPT]()  - [[code]](https://github.com/OptimalScale/DetGPT) [[Demo]](https://detgpt.github.io/) - 由港科大 & 港大的研究人员提出的模型，只需微调三百万参数量，让模型拥有复杂推理和局部物体定位能力。


##### 专有领域大模型

- [MathGPT]()  - [[website]](https://mathgpt.streamlit.app/) - MathGPT 是面向全球数学爱好者和科研机构，以数学领域的解题和讲题算法为核心的大模型。

#### 相关论文

-   [Sparks of Artificial General Intelligence: Early experiments with GPT-4](https://arxiv.org/pdf/2303.12712.pdf)
-   [DeID-GPT: Zero-shot Medical Text De-Identification by GPT-4](https://arxiv.org/pdf/2303.11032.pdf)

#### 模型微调

##### PHT-LoRA

-   Chinese-LLaMA-Alpaca - https://github.com/LC1332/Chinese-alpaca-lora
-   Yaya - https://github.com/qiyuan-chen/Yaya-Moss-Alpaca-LoRA

##### 

#### 模型评估



### 中游



#### 量化

-   llama.cpp - https://github.com/ggerganov/llama.cpp

#### 拓展

#### 与其他模型的结合

#### 二次开发



### 下游

#### Web 应用



## 教程

### 视频

### 博客



## 资源

### 工具

-   peft - https://github.com/huggingface/peft

### API

-   GPTFree - https://github.com/xtekky/gpt4free

### 镜像网站

## Contributions

<p align="center"><a href="https://github.com/huaiwen"><img src="https://avatars.githubusercontent.com/u/3187529?v=4" width="50px" alt="robinicus" /></a>&nbsp;&nbsp;<a href="https://github.com/guozihang"><img src="https://avatars.githubusercontent.com/u/17142416?v=4" width="50px" alt="0xmatchmaker" /></a>&nbsp;&nbsp;<a href="https://github.com/LeeRoc-China"><img src="https://avatars.githubusercontent.com/u/59104898?s=400&u=c225a082a6a410e3d7c84ca29a07d723d7308dca&v=4" width="50px" alt="0xmatchmaker" /></a>&nbsp;&nbsp;<a href="https://github.com/YangYang"><img src="https://avatars.githubusercontent.com/u/17808880?v=4" width="50px" alt="0xmatchmaker" /></a>&nbsp;&nbsp;</p>