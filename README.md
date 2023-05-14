<p align="center"><a href="https://github.com/huaiwen"

# 😎 **FollowGPT**

![ChatEveryThing](/Users/lipeng/Programs/ChatEverything/images/Chat Everthing.png)

<img src="https://awesome.re/badge.svg"> <img src="https://img.shields.io/badge/lang-%E4%B8%AD%E6%96%87-red"> <img src="https://img.shields.io/badge/lang-En-red">

<img src="https://img.shields.io/github/stars/immc-lab/ChatEverything.svg"> <img src="https://img.shields.io/github/watchers/immc-lab/ChatEverything.svg">

---

With the development of deep learning, large models continue to emerge. Large models can be models with a huge number of parameters in deep learning models. Recently, some work has effectively guided the knowledge of large models by combining the relevant foundation of reinforcement learning through the RLHF technique, allowing large models to achieve amazing results. However, there is currently a lack of systematic and logical integration of large model resources on the internet, which hinders the enthusiasm of many people to explore large model technology. Therefore, we have integrated the existing resources on the network to provide a clear and logical context.

If you find this repository helpful, please give us a ⭐ or share it 🥰.

## News📰

```
2023.05.06 add
```

---

## Contents

-   [Learn about the large model](#Learn about the large model)
    -   [Upstream](#Upstream)
        -   [Foundation model](#Foundation model)
        -   [Model Finetune](#Model Finetune)
        -   [Model Evaluation](#Model Evaluation)
    -   [Midstream](#Midstream)
        -   [Quantify](#Quantify)
        -   [Expansion](#Expansion)
        -   [Combination with other models](#Combination with other models)
        -   [Secondary development](#Secondary development)
    -   [Downstream](#Downstream)
        -   [Web application](#Web application)
        -   [app](#app)
-   [Tutorial](#Tutorial)
    -   [Video tutorial](#Video tutorial)
    -   [Blog](#Blog)
-   [resource](#资源)
    -   [Tools](#Tools)
    -   [Free API](# Free API)
    -   [镜像网站](#镜像网站)

---

## Learn about the large model

### Upstream

#### Foundation model

Existing large-scale model training is time-consuming and labor-intensive, often requiring large-scale computing power. The training cost is far beyond the reach of people, so the current version uses the large model as a basement model for downstream use.

##### 🦙LLMs

Text is the earliest field to generate pre-training and related fine-tuning techniques. Meanwhile, textual mockups are now leading the way, inspiring tasks in other domains. A large text model usually refers to a model with a huge amount of parameters (up to hundreds of millions of parameters). At present, the generative text model is the most popular, and its paradigm is to accept text as input and generate text.

-   [BELLE ]() - [[code]](https://github.com/LianjiaTech/BELLE) - Open source Chinese dialogue model, fine-tuned by LLaMA.
-   [ChatGLM]()  - [[arXiv]](https://arxiv.org/abs/2210.02414) [[code]](https://github.com/THUDM/ChatGLM-6B) - ChatGLM-6B is an open-source, Chinese-English bilingual conversational language model with 6.2B parameters.
-   [Phoenix]()   - [[TechReport]](https://github.com/FreedomIntelligence/LLMZoo/blob/main/assets/llmzoo.pdf) [[code]](https://github.com/FreedomIntelligence/LLMZoo) - Model fine-tuned based on BLOOMZ.
-   [MOSS]()  - [[code]](https://github.com/OpenLMLab/MOSS) - MOSS is an open source dialogue language model that supports Chinese-English bilingual and various plug-ins. The moss-moon series models have 16 billion parameters.
-   [Alpaca]()  - [[Blog]](https://crfm.stanford.edu/2023/03/13/alpaca.html) [[code]](https://github.com/tatsu-lab/stanford_alpaca) - The large model fine-tuned by LLaMa, the training corpus contains a total of 52k data.
-   [pandallm]()  - [[ArXiv]](https://arxiv.org/pdf/2305.03025) [[code]](https://github.com/dandelionsllm/pandallm) - Overseas Chinese open source large language model, based on Llama-7B, 13B, 33B, 65B for continuous pre-training in the Chinese field.
-   [Latin Phoenix: Chimera]()  - [[code]](https://github.com/dandelionsllm/pandallm) - Model fine-tuned based on LLaMA.
-   [Dolly]()  - [[Blog]](https://www.databricks.com/blog/2023/03/24/hello-dolly-democratizing-magic-chatgpt-open-models.html) [[code]](https://github.com/databrickslabs/dolly) - Databricks' Dolly, a large-scale language model trained on the Databricks machine learning platform.
-   [Guanaco]()  - [[code]](https://github.com/Guanaco-Model/Guanaco-Model.github.io) [[Dataset]](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset) - Model fine-tuned based on LLaMA.
-   [Luotuo]()  - [[code]](https://github.com/LC1332/Luotuo-Chinese-LLM) - Model fine-tuned based on LLaMA.
-   [ChatYuan]()  - [[article]](https://mp.weixin.qq.com/s/-axa6XcjGl_Koeq_OrDq8w) [[code]](https://github.com/clue-ai/ChatYuan) [[Demo]](https://modelscope.cn/studios/ClueAI/ChatYuan-large-v2) - The first Chinese version of ChatGPT, It supports various new functions such as bilingual interaction between Chinese and English.
-   [ChatGPT-3.5]()  - [[website]](https://chat.openai.com) - OpenAI text model, world-famous ChatGPT.
-   [Claude]()  - [[website]](https://www.anthropic.com/index/introducing-claude) - A large model created by some of the original people who left OpenAI.
-   [TongYi QianWen]()  - [[website]](https://tongyi.aliyun.com/) - Large model of Alibaba's open source.
-   [SparkDesk(Xinghuorenzhi)]()  - [[website]](https://xinghuo.xfyun.cn/) - Large model of iFLYTEK open source.

##### LLMs .V.S

| Model                    | Backbone     | Claimed language | **Params**     | Open-source model | Open-source data | Institution                             | Release data | Evaluate | Stars                                                        |
| ------------------------ | ------------ | ---------------- | -------------- | ----------------- | ---------------- | --------------------------------------- | ------------ | -------- | ------------------------------------------------------------ |
| ChatGPT                  | -            | multi            | -              | x                 | x                | OpenAI                                  | 2022-11-30   |          | -                                                            |
| Bard                     | -            | -                | 137B           | x                 | x                | Google                                  | 2023-02-06   |          | -                                                            |
| Claude                   | -            | zh,en            | 52B            | x                 | x                | -                                       | 2023-03-14   |          | -                                                            |
| ERNIE Bot(Wenxin)        | -            | zh               | 260B           | x                 | x                | Baidu                                   | 2023-03-16   |          | -                                                            |
| Tongyi Qianwen           | TongYi       | zh,en            | ~10T           | x                 | x                | Alibaba                                 | 2023-04-07   |          | -                                                            |
| SparkDesk(Xinghuorenzhi) | -            | zh,en            | -              | x                 | x                | iFLYTEK                                 | 2023-0506    |          | -                                                            |
| LLaMA                    | -            | multi            | 7B/13B/33B/65B | √                 | √                | Meta AI                                 | 2023-02-24   |          | ![img](https://img.shields.io/github/stars/facebookresearch/llama) |
| BELLE                    | BLOOMZ/LLaMA | zh               | 7B             | √                 | √                | -                                       | 2023-03-26   |          | ![img](https://img.shields.io/github/stars/LianjiaTech/BELLE) |
| ChatGLM                  | GLM          | zh,en            | 6B             | √                 | x                | Tsinghua University                     | 2023-03-16   |          | ![img](https://img.shields.io/github/stars/THUDM/ChatGLM-6B) |
| MOSS                     | -            | zh,en            | 16B            | √                 | √                | Fudan University                        | 2023-04-21   |          | ![img](https://img.shields.io/github/stars/OpenLMLab/MOSS)   |
| Alpace                   | LLaMA        | en               | 7B             | √                 | √                | Stanford NLP Group                      | 2023-03-13   |          | ![img](https://img.shields.io/github/stars/tatsu-lab/stanford_alpaca) |
| pandallm                 | LLaMA        | zh,en            | 7B/13B         | √                 | √                | Nanyang Technological University        | 2023-05-04   |          | ![img](https://img.shields.io/github/stars/dandelionsllm/pandallm) |
| Phoenix                  | BLOOMZ       | multi            | 7B             | √                 | √                | Shenzhen Research Institute of Big Data | 2023-04-08   |          | ![img](https://img.shields.io/github/stars/FreedomIntelligence/LLMZoo) |
| Latin Phoenix: Chimera   | LLaMA        | multi            | 7/13B          | √                 | √                | Shenzhen Research Institute of Big Data | 2023-04-08   |          | ![img](https://img.shields.io/github/stars/dandelionsllm/pandallm) |
| Dolly                    | GPT-J        | en               | 6B             | √                 | √                | -                                       | 2023-03-24   |          | ![img](https://img.shields.io/github/stars/databrickslabs/dolly) |
| Guanaco                  | LLaMA        | zh,en,ja,de      | 7B             | √                 | √                | -                                       | 2023-03-26   |          | ![img](https://img.shields.io/github/stars/Guanaco-Model/Guanaco-Model.github.io) |
| Luotuo                   | LLaMA        | zh               | 7B             | √                 | √                | -                                       | 2023-03-31   |          | ![img](https://img.shields.io/github/stars/LC1332/Luotuo-Chinese-LLM) |
| ChatYuan                 | -            | zh,en            | ～10B          | √                 | x                | YuanYu.AI                               | 2023-03-23   |          | ![img](https://img.shields.io/github/stars/clue-ai/ChatYuan) |

##### Multimodal large model

Able to accept images and text as input for text generation

-   [LLaVA🌋]()  - [[arXiv]](https://arxiv.org/abs/2304.08485) [[code]](https://github.com/haotian-liu/LLaVA) [[Demo]](https://llava.hliu.cc/) [[Dataset]](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) [[Model]](https://huggingface.co/liuhaotian/LLaVA-13b-delta-v0) - LLaVA is a novel end-to-end trained large-scale multimodal model that combines a visual encoder and Vicuna for universal visual and language understanding.
-   [MiniGPT]()  - [[arXiv]](https://arxiv.org/abs/2304.10592) [[code]](https://github.com/Vision-CAIR/MiniGPT-4) [[Demo]](https://16440e488436f49d99.gradio.live/) [[Dataset]](https://huggingface.co/datasets/Vision-CAIR/cc_sbu_align) [[Model]](https://huggingface.co/Vision-CAIR/MiniGPT-4) - It is a small natural language processing model that uses an architecture similar to GPT-3 but with a fraction of the number of parameters.
-   [mPLUG-Owl🦉]() - [[arXiv]](https://arxiv.org/abs/2304.14178) [[code]](https://github.com/x-plug/mplug-owl) [[Demo]](https://modelscope.cn/studios/damo/mPLUG-Owl/summary) - A novel training paradigm that enables LLMs to be multimodal through modular learning of base LLMs, visual knowledge modules, and visual abstraction modules.
-   [GPT-4]()  - [[TechReport]](https://arxiv.org/abs/2303.08774) [[website]](https://openai.com/product/gpt-4) - GPT-4 is a large multimodal model that exhibits human-level performance on various professional and academic benchmarks.

-   [Bard]()  - [[website]](https://bard.google.com/) - Google has developed a conversational language model called LaMDA, which is similar in design to ChatGPT.

-   [PaLM2]()  - [[TechReport]](https://ai.google/static/documents/palm2techreport.pdf) [[website]](https://ai.google/discover/palm2) - The PaLM second-generation model proposed by Google, which is benchmarked against GPT-4, has improved mathematics, code, reasoning, multilingual translation and natural language generation capabilities.

-   [DALL·E2]()  - [[website]](https://openai.com/product/dall-e-2) - DALL·E 2 is an artificial intelligence system that can create realistic images and artwork based on natural language descriptions.

-   [Wenxin]()  - [[website]](https://yiyan.baidu.com/welcome) - (WenXin) Baidu's new generation of knowledge-enhanced large language model.
-   [DetGPT]()  - [[code]](https://github.com/OptimalScale/DetGPT) [[Demo]](https://detgpt.github.io/) - The model proposed by the researchers of HKUST & HKU only needs to fine-tune 3 million parameters, so that the model has complex reasoning and local object positioning capabilities.


##### Special field large model

- [MathGPT]()  - [[website]](https://mathgpt.streamlit.app/) - It is a large-scale model that focuses on problem-solving and lecture algorithms in the field of mathematics for global mathematics enthusiasts and scientific research institutions.

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

<p align="center"><a href="https://github.com/huaiwen"><img src="https://avatars.githubusercontent.com/u/3187529?v=4" width="50px" alt="robinicus" /></a>&nbsp;&nbsp;<a href="https://github.com/guozihang"><img src="https://avatars.githubusercontent.com/u/17142416?v=4" width="50px" alt="0xmatchmaker" /></a>&nbsp;&nbsp;<a href="https://github.com/LeeRoc-China"><img src="https://avatars.githubusercontent.com/u/59104898?s=400&u=c225a082a6a410e3d7c84ca29a07d723d7308dca&v=4" width="50px" alt="0xmatchmaker" /></a>&nbsp;&nbsp;<a href="https://github.com/YangYang"><img src="https://avatars.githubusercontent.com/u/17808880?v=4" width="50px" alt="0xmatchmaker" /></a>&nbsp;&nbsp;</p>
