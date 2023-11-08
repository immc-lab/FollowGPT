# 😎 **FollowGPT**

![ChatEveryThing](./images/Chat%20Everthing.png)

<img src="https://awesome.re/badge.svg"> <img src="https://img.shields.io/badge/lang-%E4%B8%AD%E6%96%87-red"> <img src="https://img.shields.io/badge/lang-En-red">

<img src="https://img.shields.io/github/stars/immc-lab/ChatEverything.svg"> <img src="https://img.shields.io/github/watchers/immc-lab/ChatEverything.svg">

---

The development of artificial intelligence has gone through three stages: computational intelligence, perceptual intelligence, and cognitive intelligence. Computational intelligence focuses on the basic computing and storage capabilities of machines, where machines have already surpassed humans. Perceptual intelligence emphasizes machine's pattern recognition abilities, such as speech and image recognition, and currently machines have reached or even exceeded human levels in perceptual intelligence. However, there still exists a significant gap between machines and humans in cognitive intelligence, which involves research areas like natural language processing, common-sense modeling, and reasoning.

Human language, also known as natural language, possesses pervasive ambiguity, high abstraction, almost infinite semantic combinations, and continuous evolution. Understanding language requires certain cognitive abilities such as knowledge and reasoning, which pose significant challenges for computers in processing natural language and create an insurmountable gap. Therefore, natural language processing is considered one of the bottlenecks that restrict greater breakthroughs and wider applications of artificial intelligence, and it is hailed as the 'crown jewel' on the crown of artificial intelligence.

Since its inception, natural language processing has undergone five paradigm shifts in research. It has evolved from initial methods based on small-scale expert knowledge to methods based on machine learning. Machine learning methods have also transitioned from early shallow machine learning models to deep learning models. To address the issue of deep learning models requiring a large amount of annotated data, a comprehensive shift towards methods based on large-scale pre-trained language models began in 2018. These methods leverage the strengths of 'big models, big data, and big computing' to achieve better results.

Recently, ChatGPT has demonstrated impressive language comprehension, generation, and knowledge reasoning abilities. It can effectively understand user intent, engage in multi-turn conversations, and provide complete, focused, concise, logical, and organized responses. ChatGPT's successful performance has shown a possible path to addressing the core problem of cognitive intelligence in natural language processing and is considered a solid step towards general artificial intelligence.

Of course, such powerful functionality cannot be achieved by a simple model alone. Training GPT models requires an enormous amount of training data, a vast number of model parameters, and powerful computing resources. The GPT series models adhere to the concept of continuously stacking transformers and achieve iterative updates by increasing the scale and quality of training data, as well as expanding the number of network parameters. GPT has also proven that by increasing model capacity and training data scale, the model's capabilities can continue to improve.

|  Model  | Release Time | Parameter Quantity | Volume of pretrained data |
| :-----: | :----------: | :----------------: | :-----------------------: |
|   GPT   |    2018.6    |    117 million     |         About 5GB         |
|  GPT-2  |    2019.2    |    1.5 billion     |           40GB            |
|  GPT-3  |    2020.5    |    175 billion     |           45TB            |
| chatGPT |   2022.11    |         -          |             -             |
|  GPT-4  |    2023.3    |         -          |             -             |

In 2018, OpenAI introduced the first-generation GPT (Generative Pretrained Transformer) model. GPT-1 employed the Transformer architecture and large-scale unsupervised pretraining, serving as a natural language generation model. It exhibited outstanding performance on various natural language processing tasks and achieved state-of-the-art results in language modeling for individual English sentences. The success of GPT-1 provided new ideas and methods for the development of pretrained models in natural language processing.

Despite the significant success of GPT-1, it did not receive much attention. Instead, Google's subsequent introduction of the BERT (Bidirectional Encoder Representations from Transformers) model created a greater sensation. However, OpenAI continued along the technical path of the first-generation GPT and subsequently released the GPT-2 and GPT-3 models.

GPT-2 aimed to train a more generalizable word embedding model. It did not introduce significant structural innovations or designs to GPT-1's network but utilized a larger number of network parameters and a larger dataset. The major contribution of GPT-2 was to validate that word embedding models trained on massive data and with a large number of parameters could be transferred to other downstream tasks without additional training. However, many experiments also indicated that there was still significant room for improvement with GPT-2.

GPT-3 demonstrated exceptional performance on downstream tasks with zero-shot or few-shot learning. In addition to several common NLP tasks, GPT-3 showcased remarkable capabilities in challenging tasks such as generating human-indistinguishable articles, writing SQL queries, and producing React or JavaScript code. These powerful abilities relied on GPT-3's enormous scale of 175 billion parameters and the introduction of the concept of "prompts." By providing specific task prompts, the model could accomplish the task even without fine-tuning.

It was not until the emergence of ChatGPT that people's perception of large models was completely changed. Based on GPT-3.5, ChatGPT utilized human feedback reinforcement learning techniques, using human preferences as reward signals to fine-tune the model and achieve logical conversational abilities. Its goal was to generate useful, realistic, and benign text content. However, despite ChatGPT's impressive performance in many aspects, it still has some limitations. For instance, it still falls far behind human cognitive levels in deep semantic understanding and generation.

GPT-4 represents OpenAI's latest milestone in expanding deep learning. It is a large-scale multimodal model capable of accepting both image and text inputs and generating text outputs. While its capabilities in real-world scenarios may not match those of humans, it exhibits human-level performance on various professional and academic benchmarks. For example, it passed a simulated lawyer exam, scoring in the top 10% of candidates, whereas GPT-3.5 scored in the bottom 10%. GPT-4 demonstrates increased creativity and collaboration. It can generate, edit, and iterate on user-provided creative and technical writing tasks, such as composing songs, writing scripts, or learning a user's writing style. GPT-4 can also accept images as input and generate captions, classify images, and perform analysis. Furthermore, GPT-4 can handle texts with over 25,000 words, allowing for use cases involving long-form content creation, expanded conversations, document searching, and analysis.

Although GPT has achieved remarkable accomplishments thus far, there are still numerous research directions to explore in the future. For example, a common issue when questioning machines is the contextual background knowledge. For instance, if we ask a machine to tell a joke and then ask it a serious question, humans can perceive the change from your facial expressions and know that the context is no longer a joke, whereas artificial intelligence would continue with the joke. This contextual awareness plays a significant role in interactions. Additionally, in terms of problem-solving difficulty, when solving the same mathematical equation, we may need to simplify it five or six times before transforming it into the correct form, continually learning these simplifications. Machine reasoning, on the other hand, is achieved through a hierarchy of linear descent in reasoning chains, and if simplification requires running 10 times, the machine might not proceed. Mathematics involves highly abstract reasoning, which remains a major weakness of artificial intelligence. Each weak link requires time to address, and it should be treated seriously. We need more innovative approaches, such as training models in mathematical aspects through prompts or exercises.

Currently, we are just at the beginning of a new phase in artificial intelligence. We are in a stage of enthusiasm, similar to the past enthusiasm for the internet. However, the lack of systematic and logically organized resources on the internet hinders many people's exploration of large model technologies. Therefore, we have integrated existing resources on the web to provide a clear and logically coherent framework.

If you find this repository helpful, please give us a ⭐ or share it 🥰.

## News📰

```
2023.08.23 Rearranged the contents
2023.05.29 Resource collection completed!
2023.05.17 Start project!
```

---

## Table of Content

-   1.Summary of GPT
    -   1.1 Huge Model Papers
        -   1.1.1 Huge Models for NLP
        -   1.1.2 Huge Models for CV
        -   1.1.3 Huge Models for Multimodal
    -   1.2 How to Understand the Foundation Model & How to Experience the Foundation Model
        -   1.2.1 A tutorial presented in video format
        -   1.2.2 Capabilities and Future of Large Language Models
        -   1.2.3 ChatGPT Prompt Engineering for Developers
        -   1.2.4 ChatGPT
        -   1.2.5 How to understand large-scale models
        -   1.2.6 Introduction to large-scale models and their terminology
        -   1.2.7 Commonly Used Corpora
        -   1.2.8 Publicly Available Models
        -   1.2.9 Library Resource
        -   1.2.10 Deep Learning Frameworks
    -   1.3 LLM Fine Tuning Principle Introduction
        -   1.3.1 Summary of LLM fine-tuning
        -   1.3.2 Parameter-Efficient Fine-Tuning (PEFT)
        -   1.3.3 Prompt-Tuning
            -   1.3.3.1 Prefix tuning
            -   1.3.3.2 P-Tuning
        -   1.3.4 RLHF Related Papers
    -   1.4 LLM Tutorial Resources and Evaluation Comparison
    -   1.5 LLM Related Papers: Prompt, Incontent Learning, and LLM PEFT e.t.
        -   1.5.1 Incontext Learning Papers
        -   1.5.2 Prompt-based Papers
-   2.Tutorials

## 0.Openai News

### 2023.11.06 

*   发布了 GPT-4 Turbo
    *   **提供了更长的文本处理能力：**Token 数量从 32K 提升到了 128 K。
    *   **更多的控制功能**：增加了 JSON mode，让模型响应有效的 JSON。可以一次性调用多个函数。可以通过固定种子参数使模型返回一致的结果。将推出在 API 查看日志功能。
    *   **最新的世界知识**：现在拥有截止 2023 年 4 月之前的知识。
    *   **新的模态接口**：GPT4、DALLE 和 TTS 的整合。
    *   **自定义模型**：可以微调 16K 的模型。
    *   **更高的速率限制**：每分钟的 Token 数量翻倍。可以设置限速。引入的版权保护。
    *   **价格**：GPT4 Turbo 便宜比 GPT4 三倍。输出的 Token 下降一半。旧模型价格下调。
*   发布了 GPTs
    *   **GPT Builder**：用于创建 GPTs。可以选择具有的能力，网络搜索、图像生成、代码执行。同时可以上传知识
*   将推出 Assistants API

## 1.Summary of GPT

Generative Pretrained Transformer (GPT) models have demonstrated significant proficiency in understanding and generating language, one of their standout characteristics is their versatility across domains. Regardless of the text data in question, GPT models can generate coherent, contextually accurate, and often insightful responses without being explicitly trained on the task. This adaptability is quite remarkable and sets GPT models apart in the landscape of language models.Another point of praise for GPT is its ability to engage in semantic search. Instead of merely matching keywords, as with traditional search algorithms, GPT can comprehend the meaning behind the words. This enables it to provide more relevant, precise, and contextually fitting results, highlighting its utility in data mining and information retrieval applications.

### 1.1 Huge Model Papers

The use of pre trained large-scale models for fine-tuning downstream tasks is currently a popular deep learning paradigm. Especially with the outstanding performance of the recently pre trained language model ChatGPT, this technical paradigm has been widely recognized. We mainly answered the question of what big models are available, and from a macro perspective, introduced the readers to what big models are available and what scope they cover. We have collected relevant papers from 2021 to the present from three aspects of natural language processing, computer vision and multimodality. Large models started early in the field of natural language processing, and then gradually evolved into computer vision related fields, such as SAM, which can split everything. At present, integrating large models from multiple professional fields has become a new paradigm for training large models in multimodal fields, such as Huawei's Pangu model, which covers fields such as finance and meteorology.

#### 1.1.1 Huge Models for NLP

| Year | Model Name         | Title                                                        | Venue                                        | Paper                                                        | Code                                                         |
| ---- | ------------------ | ------------------------------------------------------------ | -------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 2023.07.06 | 曹植 |  |  |  |  |
| 2023.05.06 | 星火 |  |  |  |  |
| 2023.04.24 | MOSS |  |  |  |  |
| 2023.04.11 | 通义千问 |  |  |  |  |
| 2023.03.20 | 文心一言 |  |  |  |  |
| 2021.08.11 | 紫东·太初 |  |  |  |  |
|  | 封神榜MindBot |  |  |  |  |
|  | 白玉兰 |  |  |  |  |
|  | K2 |  |  |  |  |
|  | PICA |  |  |  |  |
|  | TechGPT |  |  |  |  |
|  | TigerBot |  |  |  |  |
|  | XVERSE-13B |  |  |  |  |
|  | 凤凰 |  |  |  |  |
|  | 华佗 |  |  |  |  |
|  | CPM-Bee |  |  |  |  |
|  | CPM |  |  |  |  |
|  | 山海 |  |  |  |  |
|  | 活字 |  |  |  |  |
|  | 本草 |  |  |  |  |
|  | BELLE |  |  |  |  |
|  | OpenMEDLab浦医 |  |  |  |  |
|  | 书生·浦语 |  |  |  |  |
|  | baichuan-13B |  |  |  |  |
|  | baichuan-7B |  |  |  |  |
| 2023 | 百川 |  |  |  |  |
| 2023 | TableGPT |  |  |  |  |
| 2023 | PromptProtein |  |  |  |  |
| 2023 | 启真 |  |  |  |  |
| 2023 | 悟道EMU |  |  | [link](https://arxiv.org/abs/2307.05222) | [link](https://github.com/baaivision/Emu) |
| 2023 | 悟道天鹰 |  |  |  |  |
| 2023 | 盘古-$\sum$ |  |  |  |  |
| 2023 | 盘古气象 |  |  |  |  |
| 2023 | 盘古 |  |  |  |  |
| 2023 | 灵医bot            |                                                              |                                              |                                                              |                                                              |
| 2023 | LLAMA-2            | Llama 2: Open Foundation and Fine-Tuned Chat Models          | Arxiv                                        | [link](https://scontent-sea1-1.xx.fbcdn.net/v/t39.2365-6/10000000_662098952474184_2584067087619170692_n.pdf?_nc_cat=105&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=DCwZC7aH-soAX8RMdR5&_nc_ht=scontent-sea1-1.xx&oh=00_AfDfQiss1S25Ho6kyhIqlSvlaj4SMcYTKGDZZfDdIgL4Uw&oe=64DE0AFF) |                                                              |
| 2023 | 百聆               | BayLing: Bridging Cross-lingual Alignment and Instruction Following through Interactive Translation for Large Language Models | arxiv                                        | [link](https://arxiv.org/abs/2306.10968)                     | [link](https://github.com/ictnlp/BayLing)                    |
| 2023 | Dolly 2.0          | -                                                            | -                                            | -                                                            | [link](https://github.com/zhengzangw/awesome-huge-models/tree/main) |
| 2023 | StableLM           | -                                                            | -                                            | -                                                            | [link](https://huggingface.co/stabilityai)                   |
| 2023 | Cerabras-GPT       | Training Compute-Optimal Large Language Models               | arxiv                                        | [link](https://arxiv.org/abs/2203.15556)                     | -                                                            |
| 2023 | LLaMa              | Open and Efficient Foundation Language Models                | arxiv                                        | [link](https://arxiv.org/pdf/2302.13971v1.pdf)               | [link](https://github.com/facebookresearch/llama)            |
| 2022 | BLOOM              | A 176B-Parameter Open-Access Multilingual Language Model     | arxiv                                        | [link](https://arxiv.org/pdf/2211.05100.pdf)                 | [link](https://huggingface.co/bigscience/bloom)              |
| 2022 | Galactica          | A scientific language model trained on over 48 million scientific texts | arxiv                                        | [linl](https://arxiv.org/pdf/2211.09085.pdf)                 | [link](https://huggingface.co/facebook/galactica-1.3b)       |
| 2022 | GLM-130B           | GLM-130B: An Open Bilingual Pre-trained Model                | ICLR 2023                                    | [link](https://arxiv.org/pdf/2210.02414.pdf)                 | [link](https://github.com/THUDM/GLM-130B)                    |
| 2022 | UL2                | Unifying Language Learning Paradigms                         | arxiv                                        | [link](https://arxiv.org/pdf/2205.05131.pdf)                 | [link](https://huggingface.co/google/ul2)                    |
| 2022 | OPT                | OPT: Open Pre-trained Transformer Language Models            | arxiv                                        | [link](https://arxiv.org/pdf/2205.01068.pdf)                 | [link](https://github.com/facebookresearch/metaseq)          |
| 2022 | PaLM               | PaLM: Scaling Language Modeling with Pathways                | arxiv                                        | [link](https://arxiv.org/pdf/2204.02311.pdf)                 | -                                                            |
| 2022 | GPT-NeoX           | GPT-NeoX-20B: An Open-Source Autoregressive Language Model   | ACL 2022                                     | [link](https://arxiv.org/pdf/2204.06745.pdf)                 | [link](https://github.com/EleutherAI/gpt-neox)               |
| 2022 | InstructGPT        | Training language models to follow instructions with human feedback | [link](https://arxiv.org/pdf/2203.02155.pdf) | -                                                            |                                                              |
| 2022 | EVA 2.0            | EVA2.0: Investigating Open-Domain Chinese Dialogue Systems with Large-Scale Pre-Trainingraining language mode | arxiv                                        | [link](https://arxiv.org/pdf/2203.09313.pdf)                 | [link](https://openi.pcl.ac.cn/BAAI/WuDao-Model/src/branch/master) |
| 2022 | AlphaCode          | Competition-Level Code Generation with AlphaCode             | arxiv                                        | [link](https://arxiv.org/pdf/2203.07814.pdf)                 | -                                                            |
| 2022 | ST-MoE             | ST-MoE: Designing Stable and Transferable Sparse Expert Models | arxiv                                        | [link](https://arxiv.org/pdf/2202.08906.pdf)                 | -                                                            |
| 2022 | LaMDA              | LaMDA: Language Models for Dialog Applications               | arxiv                                        | [link](https://arxiv.org/pdf/2201.08239.pdf)                 | -                                                            |
| 2022 | ERNIE-ViLG         | ERNIE-ViLG: Unified Generative Pre-training for Bidirectional Vision-Language Generation | arxiv                                        | [link](https://arxiv.org/pdf/2112.15283.pdf)                 | -                                                            |
| 2021 | GLaM               | GLaM: Efficient Scaling of Language Models with Mixture-of-Experts | ICML 2022                                    | [link](https://arxiv.org/pdf/2112.06905.pdf)                 | -                                                            |
| 2021 | Gopher             | Scaling Language Models: Methods, Analysis & Insights from Training Gopher | arxiv                                        | [link](https://arxiv.org/pdf/2112.11446.pdf)                 | -                                                            |
| 2021 | Yuan 1.0           | Yuan 1.0: Large-Scale Pre-trained Language Model in Zero-Shot and Few-Shot Learning | arxiv                                        | [link](https://arxiv.org/pdf/2110.04725.pdf)                 | -                                                            |
| 2021 | MT-NLG             | Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B, A Large-Scale Generative Language Model | arxiv                                        | [link](https://arxiv.org/pdf/2201.11990.pdf)                 | -                                                            |
| 2021 | Codex              | Evaluating Large Language Models Trained on Code             | arxiv                                        | [link](https://arxiv.org/pdf/2107.03374.pdf)                 | -                                                            |
| 2021 | ERNIE 3.0          | ERNIE 3.0: Large-scale Knowledge Enhanced Pre-training for Language Understanding and Generation | arxiv                                        | [link](https://arxiv.org/pdf/2107.02137.pdf)                 | -                                                            |
| 2021 | HyperClova         | What Changes Can Large-scale Language Models Bring? Intensive Study on HyperCLOVA: Billions-scale Korean Generative Pretrained Transformers | EMNLP 2021                                   | [link](https://arxiv.org/pdf/2109.04650v1.pdf)               | -                                                            |
| 2021 | ByT5               | ByT5: Towards a token-free future with pre-trained byte-to-byte models | TACL 2022                                    | [link](https://arxiv.org/pdf/2105.13626.pdf)                 | [link](https://github.com/google-research/byt5)              |
| 2021 | PanGu-α            | PanGu-α: Large-scale Autoregressive Pretrained Chinese Language Models with Auto-parallel Computation | arxiv                                        | [link](https://arxiv.org/pdf/2104.12369.pdf)                 | -                                                            |
| 2021 | mT5                | mT5: A massively multilingual pre-trained text-to-text transformer | arxiv                                        | [link](https://arxiv.org/pdf/2010.11934.pdf)                 | [link](https://github.com/google-research/multilingual-t5)   |
| 2021 | GLM                | GLM: General Language Model Pretraining with Autoregressive Blank Infilling | ACL 2022                                     | [link](https://arxiv.org/pdf/2103.10360.pdf)                 | [link](https://openi.pcl.ac.cn/BAAI/WuDao-Model/src/branch/master/GLM) |
| 2021 | Switch Transformer | Scaling to Trillion Parameter Models with Simple and Efficient Sparsity | JMLR 2021                                    | [link](https://arxiv.org/pdf/2101.03961.pdf)                 | -                                                            |
| 2020 | CPM                | CPM: A Large-scale Generative Chinese Pre-trained Language Model | arxiv                                        | [link](https://arxiv.org/pdf/2012.00413.pdf)                 | [link](https://github.com/TsinghuaAI/CPM)                    |
| 2020 | GPT-3              | Language Models are Few-Shot Learners                        | arxiv                                        | [link](https://papers.nips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf) | -                                                            |
| 2020 | Blender            | Recipes for building an open-domain chatbot                  | arxiv                                        | [link](https://arxiv.org/pdf/2004.13637.pdf)                 | -                                                            |
| 2020 | Meena              | Towards a Human-like Open-Domain Chatbot                     | arxiv                                        | [link](https://arxiv.org/pdf/2001.09977.pdf)                 | -                                                            |
| 2019 | DialoGPT           | DialoGPT: Large-Scale Generative Pre-training for Conversational Response Generation | ACL 2020                                     | [link](https://arxiv.org/pdf/1911.00536.pdf)                 | [link](https://github.com/microsoft/DialoGPT)                |
| 2019 | T5                 | Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer | JMLR 2019                                    | [link](https://arxiv.org/pdf/1910.10683.pdf)                 | [link](https://github.com/google-research/text-to-text-transfer-transformer) |
| 2019 | Megatron-LM        | Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism | arxiv                                        | [link](https://arxiv.org/pdf/1909.08053.pdf)                 | [link](https://github.com/NVIDIA/Megatron-LM)                |
| 2019 | RoBERTa            | RoBERTa: A Robustly Optimized BERT Pretraining Approach      | arxiv                                        | [link](https://arxiv.org/pdf/1907.11692.pdf)                 | [link](https://github.com/facebookresearch/fairseq)          |
| 2019 | XLNet              | XLNet:Generalized Autoregressive Pretraining for Language Understanding | NeurIPS 2019                                 | [link](https://papers.nips.cc/paper/2019/hash/dc6a7e655d7e5840e66733e9ee67cc69-Abstract.html) | [link](https://github.com/zihangdai/xlnet)                   |
| 2019 | GPT-2              | Language Models are Unsupervised Multitask Learners          | arxiv                                        | [link](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | [link](https://github.com/NVIDIA/Megatron-LM)                |
| 2018 | GPT                | Improving Language Understanding by Generative Pre-Training  | -                                            | [link](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) | [link](https://github.com/google-research/vmoe)              |
| 2018 | BERT               | BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | NAACL 2018                                   | [link](https://arxiv.org/pdf/1810.04805.pdf)                 | [link](https://github.com/google-research/bert)              |

#### 1.1.2 Huge Models for CV

| Year | Model Name       | Title                                                                      | Venue    | Paper                                                                                                                                                                                                                                                                  | Code                                                                            |
| ---- | ---------------- | -------------------------------------------------------------------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| 2023 | DINOv2           | DINOv2: Learning Robust Visual Features without Supervision                | arxiv    | [link](https://arxiv.org/pdf/2304.07193.pdf)                                                                                                                                                                                                                           | -                                                                               |
| 2023 | SEEM             | Segment Everything Everywhere All at Once                                  | arxiv    | [link](https://arxiv.org/pdf/2304.06718.pdf)                                                                                                                                                                                                                           | [link](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once) |
| 2023 | Visual ChatGPT   | Visual ChatGPT: Talking, Drawing and Editing with Visual Foundation Models | arxiv    | [link](https://arxiv.org/pdf/2303.04671.pdf)                                                                                                                                                                                                                           | [link](https://github.com/microsoft/TaskMatrix)                                 |
| 2023 | PICASSO          | -                                                                          | -        | -                                                                                                                                                                                                                                                                      | [official website](https://www.nvidia.com/zh-tw/gpu-cloud/picasso/)             |
| 2023 | SAM              | Segment Anything                                                           | -        | [link](https://scontent-lax3-2.xx.fbcdn.net/v/t39.2365-6/10000000_900554171201033_1602411987825904100_n.pdf?_nc_cat=100&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=DsUiPPmfKCcAX9z0vCN&_nc_ht=scontent-lax3-2.xx&oh=00_AfCJOMTowYIhhAmJ9CBVsh-pg2LgSwqscB-g_KJAY_bWVQ&oe=646E5A27) | [official website](https://segment-anything.com/)                               |
| 2023 | 书生             | -                                                                          | -        | [official news](https://www.sensetime.com/cn/news-detail/51166318?categoryId=72)                                                                                                                                                                                       | -                                                                               |
| 2022 | AllInOne         | All in One: Exploring Unified Video-Language Pre-training                  | arxiv    | [link](https://arxiv.org/pdf/2203.07303.pdf)                                                                                                                                                                                                                           | [link](https://github.com/showlab/all-in-one)                                   |
| 2022 | CoCa             | CoCa: Contrastive Captioners are Image-Text Foundation Models              | arxiv    | [link](https://arxiv.org/pdf/2205.01917.pdf)                                                                                                                                                                                                                           | -                                                                               |
| 2012 | CoAtNet          | CoAtNet: Marrying Convolution and Attention for All Data Sizes             | arxiv    | [link](https://arxiv.org/pdf/2106.04803.pdf)                                                                                                                                                                                                                           | -                                                                               |
| 2021 | Swin Transformer | Swin Transformer: Hierarchical Vision Transformer using Shifted Windows    | arxiv    | [link](https://arxiv.org/pdf/2103.14030.pdf)                                                                                                                                                                                                                           | [link](https://github.com/microsoft/Swin-Transformer)                           |
| 2021 | V-MOE            | Scaling Vision with Sparse Mixture of Experts                              | arxiv    | [link](https://arxiv.org/pdf/2106.05974.pdf)                                                                                                                                                                                                                           | -                                                                               |
| 2021 | ViT              | An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale | ICLR2021 | [link](https://arxiv.org/pdf/2010.11929.pdf)                                                                                                                                                                                                                           | [link](https://github.com/google-research/vision_transformer)                   |

#### 1.1.3 Huge Models for Multimodal

| Year      | Model Name   | Title                                                                                                                                        | Venue                                     | Paper                                                                                         | Code |
| --------- | ------------ | -------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- | --------------------------------------------------------------------------------------------- | ---- |
| 2022      | DALL·E2      | Hierarchical Text-Conditional Image Generation with CLIP Latents                                                                             | -                                         | [link](https://cdn.openai.com/papers/dall-e-2.pdf)                                            | -    |
| 2022      | Imagen       | Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding                                                               | arxiv                                     | [link](https://arxiv.org/pdf/2205.11487.pdf)                                                  | -    |
| 2022      | LayoutLM v3  | LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking                                                                 | arxiv                                     | [link](https://arxiv.org/pdf/2204.08387.pdf)                                                  | -    |
| 2022      | VPT          | Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos                                                                 | arxiv                                     | [link ](https://arxiv.org/pdf/2206.11795.pdf)                                                 | -    |
| 2022      | Gato         | A Generalist Agent                                                                                                                           | Transactions on Machine Learning Research | [link](https://openreview.net/pdf?id=1ikK0kHjvj)                                              | -    |
| 2022      | 孟子         | -                                                                                                                                            | arxiv                                     | [official website](https://arxiv.org/pdf/2105.13290.pdf)                                      | -    |
| 2021      | 紫东太初     | CogView: Mastering Text-to-Image Generation via Transformers                                                                                 | arxiv                                     | [link](https://cdn.openai.com/papers/dall-e-2.pdf)                                            | -    |
| 2021      | 盘古         | PanGu-α: Large-scale Autoregressive Pretrained Chinese Language Models with Auto-parallel Computation                                        | arxiv                                     | [link](https://arxiv.org/pdf/2104.12369.pdf)                                                  | -    |
| 2021-2022 | Chinese CLIP | Learning Transferable Visual Models From Natural Language Supervision <br/> Chinese CLIP: Contrastive Vision-Language Pretraining in Chinese | arxiv                                     | [link](https://arxiv.org/pdf/2103.00020.pdf)<br/>[link](https://arxiv.org/pdf/2211.01335.pdf) | -    |

### 1.2 How to Understand the Foundation Model & How to Experience the Foundation Model

In recent times, significant advancements have been made in the research of LLMs by both academia and industry. One notable achievement is the introduction of ChatGPT, a powerful AI chatbot developed based on LLMs, which has garnered considerable attention from society. The technical evolution of LLMs is having a profound impact on the entire AI community, revolutionizing the way we develop and utilize AI algorithms.
For the convenience of those interested in the LLMs, we have found some papers, blogs, etc. in Part IV to answer the question of how to understand the foundation model, and some open source resources on the LLMs for the base model including but not limited to: **Publicly Available Model Checkpoints or APIs**, **Commonly Used Corpora**, **Library Resource**, etc. for people who can experience and replicate the foundation model.

#### 1.2.1 A tutorial presented in video format

bilibili：https://www.bilibili.com/video/BV1UG411p7zv/?spm_id_from=333.337.search-card.all.click&vd_source=bfa4daa4cd37650d377d43f7aa846a79

bilibili：https://www.bilibili.com/video/BV1mM4y147qw/

bilibili：https://www.bilibili.com/video/BV16N411K7aT/?vd_source=64c7856b444da4b1308dc078ccd41d80

#### 1.2.2 Capabilities and Future of Large Language Models

大语言模型介绍：https://www.51cto.com/article/753233.html

#### 1.2.3 ChatGPT Prompt Engineering for Developers

吴恩达新课 chatGPT Prompt Engineering for Developers 笔记：https://zhuanlan.zhihu.com/p/625917566

#### 1.2.4 ChatGPT

1. B 站：https://www.bilibili.com/video/BV1cg4y1j7F9/?spm_id_from=333.999.0.0&vd_source=4feabcdf8e3d49724afd39c33e65e9a4
   (介绍 chatGPT 模型的基本过程、原理、处理流程和参数组成，持续更新)

2. GPT-3.5 免费用，GPT-4 要充一点：https://ai.usesless.com/scene/home

#### 1.2.5 How to understand large-scale models

1、[知乎：什么是大模型？超大模型和 Foundation Model](https://www.zhihu.com/question/498275802/answer/2221187242?utm_campaign=shareopn&utm_content=group3_Answer&utm_medium=social&utm_oi=1317776601833263104&utm_psn=1642132929626337280&utm_source=wechat_session) （讲的不算很系统，但是对 gpt 的特性的还可以）

2、[知乎：gpt 训练的理解](https://www.zhihu.com/question/585091993/answer/2902038825?utm_campaign=shareopn&utm_content=group3_Answer&utm_medium=social&utm_oi=1317776601833263104&utm_psn=1642134784066473984&utm_source=wechat_session)（对 gpt 训练的方式进行了简述）

3、[知乎：如何理解 chatgpt](https://www.zhihu.com/question/598243591/answer/3016818013)（把 chatgpt 的方法进行了简述，用来了解 gpt 的方法很不错）

4、[知乎：如何理解 gpt 中的 RLHF](https://zhuanlan.zhihu.com/p/614284159)（把 RLHF 进行了讲解，不错）

5、[知乎：# GPT 模型成功的背后用到了哪些以数据为中心的人工智能（Data-centric AI）技术？](https://zhuanlan.zhihu.com/p/617057227)

6、[知乎：如何理解斯坦福大学的论文“大模型涌现能力是海市蜃楼，那是度量选择的结果”的观点？ ](https://www.zhihu.com/question/599186065/answer/3027731268)

#### 1.2.6 Introduction to large-scale models and their terminology

了解大模型：https://www.zhihu.com/question/498275802/answer/2221187242
术语学习：https://zhuanlan.zhihu.com/p/615074572

#### 1.2.7 Commonly Used Corpora

1. BookCorpus: "Aligning Books and Movies: Towards Story-like Visual Explanations by Watching Movies and Reading Books". Yukun Zhu et al. ICCV 2015. [[Paper](http://arxiv.org/abs/1506.06724v1)] [[Source](https://huggingface.co/datasets/bookcorpus)]
2. Guntenburg: [[Source](https://www.gutenberg.org/)]
3. CommonCrawl: [[Source](https://commoncrawl.org/)]
4. C4: "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer". Colin Raffel et al. JMLR 2019. [[Paper](http://arxiv.org/abs/1910.10683v3)] [[Source](https://www.tensorflow.org/datasets/catalog/c4)]
5. CC-stories-R: "A Simple Method for Commonsense Reasoning". Trieu H. Trinh el al. arXiv 2018. [[Paper](http://arxiv.org/abs/1806.02847v2)] [[Source](https://huggingface.co/datasets/spacemanidol/cc-stories)]
6. CC-NEWS: "RoBERTa: A Robustly Optimized BERT Pretraining Approach". Yinhan Liu et al. arXiv 2019. [[Paper](http://arxiv.org/abs/1907.11692v1)] [[Source](https://huggingface.co/datasets/cc_news)]
7. REALNEWs: "Defending Against Neural Fake News". Rowan Zellers et al. NeurIPS 2019. [[Paper](http://arxiv.org/abs/1905.12616v3)] [[Source](https://github.com/rowanz/grover/tree/master/realnews)]
8. OpenWebText: [[Source](https://skylion007.github.io/OpenWebTextCorpus/)]
9. Pushshift.io: "The Pushshift Reddit Dataset". Jason Baumgartner et al. AAAI 2020. [[Paper](http://arxiv.org/abs/2001.08435v1)] [[Source](https://files.pushshift.io/reddit/)]
10. Wikipedia: [[Source](https://dumps.wikimedia.org/)]
11. BigQuery: [[Source](https://cloud.google.com/bigquery/public-data?hl=zh-cn)]
12. The Pile: "The Pile: An 800GB Dataset of Diverse Text for Language Modeling". Leo Gao et al. arxiv 2021. [[Paper](http://arxiv.org/abs/2101.00027v1)] [[Source](https://pile.eleuther.ai/)]
13. ROOTS: "The BigScience ROOTS Corpus: A 1.6TB Composite Multilingual Dataset". Laurençon et al. NeurIPS 2022 Datasets and Benchmarks Track. [[paper](https://arxiv.org/abs/2303.03915)]

#### 1.2.8 Publicly Available Models

1. T5: "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer". Colin Raffel et al. JMLR 2019. [[Paper](https://arxiv.org/abs/1910.10683)] [[Checkpoint](https://huggingface.co/t5-11b)]
2. mT5: "mT5: A massively multilingual pre-trained text-to-text transformer". Linting Xue et al. NAACL 2021. [[Paper](https://arxiv.org/abs/2010.11934)] [[Checkpoint](https://huggingface.co/google/mt5-xxl/tree/main)]
3. PanGu-α: "PanGu-α: Large-scale Autoregressive Pretrained Chinese Language Models with Auto-parallel Computation". Wei Zeng et al. arXiv 2021. [[Paper](https://arxiv.org/abs/2104.12369)] [[Checkpoint](https://openi.pcl.ac.cn/PCL-Platform.Intelligence/PanGu-Alpha)]
4. CPM-2: "CPM-2: Large-scale Cost-effective Pre-trained Language Models". Zhengyan Zhang et al. arXiv 2021. [[Paper](https://arxiv.org/abs/2106.10715)] [[Checkpoint](https://github.com/TsinghuaAI/CPM)]
5. T0: "Multitask Prompted Training Enables Zero-Shot Task Generalization". Victor Sanh et al. ICLR 2022. [[Paper](https://arxiv.org/abs/2110.08207)] [[Checkpoint](https://huggingface.co/bigscience/T0)]
6. GPT-NeoX-20B: "GPT-NeoX-20B: An Open-Source Autoregressive Language Model". Sid Black et al. arXiv 2022. [[Paper](https://arxiv.org/abs/2204.06745)] [[Checkpoint](https://huggingface.co/EleutherAI/gpt-neox-20b/tree/main)]
7. CodeGen: "CodeGen: An Open Large Language Model for Code with Multi-Turn Program Synthesis". Erik Nijkamp et al. arXiv 2022. [[Paper](https://arxiv.org/abs/2203.13474)] [[Checkpoint](https://huggingface.co/Salesforce/codegen-16B-nl)]
8. Tk-Instruct: "Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks". Yizhong Wang et al. EMNLP 2022. [[Paper](https://arxiv.org/abs/2204.07705)] [[Checkpoint](https://huggingface.co/allenai/tk-instruct-11b-def-pos)]
9. UL2: "UL2: Unifying Language Learning Paradigms". Yi Tay et al. arXiv 2022. [[Paper](https://arxiv.org/abs/2205.05131)] [[Checkpoint](https://github.com/google-research/google-research/tree/master/ul2)]
10. OPT: "OPT: Open Pre-trained Transformer Language Models". Susan Zhang et al. arXiv 2022. [[Paper](https://arxiv.org/abs/2205.01068)] [[Checkpoint](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT)]
11. NLLB: "No Language Left Behind: Scaling Human-Centered Machine Translation". NLLB Team. arXiv 2022. [[Paper](https://arxiv.org/abs/2207.04672)] [[Checkpoint](https://github.com/facebookresearch/fairseq/tree/nllb)]
12. BLOOM: "BLOOM: A 176B-Parameter Open-Access Multilingual Language Model". BigScience Workshop. arXiv 2022. [[Paper](https://arxiv.org/abs/2211.05100)] [[Checkpoint](https://huggingface.co/bigscience/bloom)]
13. GLM: "GLM-130B: An Open Bilingual Pre-trained Model". Aohan Zeng et al. arXiv 2022. [[Paper](https://arxiv.org/abs/2210.02414)] [[Checkpoint](https://github.com/THUDM/GLM-130B)]
14. Flan-T5: "Scaling Instruction-Finetuned Language Models". Hyung Won Chung et al. arXiv 2022. [[Paper](https://arxiv.org/abs/2210.11416)] [[Checkpoint](https://github.com/google-research/t5x/blob/main/docs/models.md#flan-t5-checkpoints)]
15. mT0 && BLOOMZ: "Crosslingual Generalization through Multitask Finetuning". Niklas Muennighoff et al. arXiv 2022. [[Paper](https://arxiv.org/abs/2211.01786)] [[Checkpoint](https://github.com/bigscience-workshop/xmtf)]
16. Galactica: "Galactica: A Large Language Model for Science". Ross Taylor et al. arXiv 2022. [[Paper](https://arxiv.org/abs/2211.09085)] [[Checkpoint](https://huggingface.co/facebook/galactica-120b)]
17. OPT-IML: "OPT-IML: Scaling Language Model Instruction Meta Learning through the Lens of Generalization". Srinivasan et al. . arXiv 2022. [[Paper](https://arxiv.org/abs/2212.12017)] [[Checkpoint](https://huggingface.co/facebook/opt-iml-30b)]
18. CodeGeeX: "CodeGeeX: A Pre-Trained Model for Code Generation with Multilingual Evaluations on HumanEval-X". Qinkai Zheng et al. . arXiv 2023. [[Paper](https://arxiv.org/abs/2303.17568)] [[Checkpoint](https://github.com/THUDM/CodeGeeX)]
19. Pythia: "Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling". Stella Biderman et al. . arXiv 2023. [[Paper](https://arxiv.org/abs/2304.01373)] [[Checkpoint](https://github.com/EleutherAI/pythia)]
20. LLaMA: "LLaMA: Open and Efficient Foundation Language Models". Hugo Touvron et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2302.13971v1)] [[Checkpoint](https://github.com/facebookresearch/llama)]

#### 1.2.9 Library Resource

1. Transformers: "Transformers: State-of-the-Art Natural Language Processing". Thomas Wolf et al. EMNLP 2020. [[Paper](https://arxiv.org/abs/1910.03771)] [[Source](https://huggingface.co/)]
2. DeepSpeed: "Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters". Rasley et al. KDD 2020. [[Paper](https://dl.acm.org/doi/10.1145/3394486.3406703)] [[Source](https://github.com/microsoft/DeepSpeed)]
3. Megatron-LM: "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism". Mohammad Shoeybi et al. arXiv 2019. [[Paper](https://arxiv.org/abs/1909.08053)]] [[Source](https://github.com/NVIDIA/Megatron-LM)]
4. JAX: [[Source](https://github.com/google/jax)]
5. Colossal-AI: "Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training". Zhengda Bian et al. arXiv 2021. [[Paper](http://arxiv.org/abs/2110.14883v2)] [[Source](https://github.com/hpcaitech/ColossalAI)]
6. BMTrain: [[Source](https://github.com/OpenBMB/BMTrain)]
7. FastMoE: "FastMoE: A Fast Mixture-of-Expert Training System". Jiaao He et al. arXiv 2021. [[Paper](https://arxiv.org/abs/2103.13262)]] [[Source](https://github.com/laekov/fastmoe)]

#### 1.2.10 Deep Learning Frameworks

1. Pytorch: "PyTorch: An Imperative Style, High-Performance Deep Learning Library". Adam Paszke el al. NeurIPS 2019. [[Paper](https://arxiv.org/abs/1912.01703)] [[Source](https://pytorch.org/)]
2. TensorFlow: "TensorFlow: A system for large-scale machine learning". Martín Abadi et al. OSDI 2016. [[Paper](https://arxiv.org/abs/1605.08695)] [[Source](https://www.tensorflow.org/)]
3. MXNet: "MXNet: A Flexible and Efficient Machine Learning Library for Heterogeneous Distributed Systems". Tianqi Chen et al. arXiv 2015. [[Paper](https://arxiv.org/abs/1512.01274)] [[Source](https://github.com/apache/mxnet)]
4. PaddlePaddle: "PaddlePaddle: An Open-Source Deep Learning Platform from Industrial Practice" . Yanjun Ma et al. Frontiers of Data and Domputing 2019. [[Paper](http://www.jfdc.cnic.cn/EN/abstract/abstract2.shtml)] [[Source](https://github.com/PaddlePaddle/Paddle)]
5. MindSpore: "Huawei MindSpore AI Development Framework" . Huawei Technologies Co., Ltd. Artificial Intelligence Technology 2022. [[Paper](https://link.springer.com/chapter/10.1007/978-981-19-2879-6_5)] [[Source](https://github.com/mindspore-ai/mindspore)]
6. OneFlow: "OneFlow: Redesign the Distributed Deep Learning Framework from Scratch" . Jinhui Yuan et al. arXiv 2021. [[Paper](https://arxiv.org/abs/2110.15032)] [[Source](https://github.com/Oneflow-Inc/oneflow)]

### 1.3 LLM Fine Tuning Principle Introduction

Fine-tuning is a technique used in natural language processing (NLP) to adapt pre-trained language models to specific tasks or domains. In this part, we make a brief introduction to the fine tuning technology, and systematically introduce the fine tuning technology commonly used in LLM. At the same time, we made a summary of RLHF related technologies and collected relevant papers from 2021 to 2023.

#### 1.3.1 Summary of LLM fine-tuning

Fine-tuning is a technique used in natural language processing (NLP) to adapt pre-trained language models to specific tasks or domains. The basic idea of fine-tuning is to take a pre-trained
language model that has been trained on a large amount of text and continue training it on a smaller set of task-specific text.The concept of fine-tuning has been around for many years and
has been utilized in various contexts. One of the earliest known applications of fine-tuning in NLP was in the domain of Neural Machine Translation (NMT), where researchers used pre-trained
neural networks to initialize the weights of a smaller network and then fine-tuned it for specific translation tasks.Classic fine-tuning methods involve continuing the training of a pre-trained
model with a small amount of task-specific data. During this process, the weights of the pre-trained model are updated to better adapt to the task at hand. The amount of fine-tuning required
depends on the similarity between the pre-training corpus and the task-specific corpus. If the two are similar, only a small amount of fine-tuning may be needed. If they are dissimilar, more
extensive fine-tuning may be required.One of the most well-known examples of fine-tuning in NLP is the OpenAI GPT (Generative Pre-trained Transformer) model developed by OpenAI. The GPT model
undergoes pre-training on a large corpus of text and is then fine-tuned on various tasks, such as language modeling, question answering, and summarization. The fine-tuned models have achieved
state-of-the-art performance on these tasks.

#### 1.3.2 Parameter-Efficient Fine-Tuning (PEFT)

Parameter-Efficient Fine-Tuning (PEFT) is a set of methods in natural language processing (NLP) aimed at achieving effective fine-tuning of pre-trained language models while minimizing the
required parameters and computational resources. PEFT focuses on reducing the number of parameters and computational resources compared to traditional fine-tuning methods. It enables efficient
adaptation of pre-trained language models to specific tasks in NLP.From a different perspective, the parameter-efficient fine-tuning technique addresses the resource-intensive nature of traditional
fine-tuning methods by training only a small set of parameters, which can be a subset of existing model parameters or a newly added set of parameters. These methods vary in terms of parameter
efficiency, memory efficiency, training speed, the overall quality of the model, and potential additional inference costs, if any. The goal is to strike a balance between achieving effective
fine-tuning and minimizing the computational burden associated with training and deploying the models.Indeed, these techniques are highly valuable for researchers and developers who may not have
access to powerful hardware or need to fine-tune models on low-resource devices. By reducing the parameter and computational requirements, parameter-efficient fine-tuning enables a wider range of
users to leverage pre-trained language models effectively. This accessibility opens up opportunities for individuals and organizations with limited resources to participate in NLP research and
application development, fostering innovation and democratizing access to advanced language processing capabilities.

A review of parameter-efficient fine-tuning.This paper presents a systematic overview and comparison of parameter-efficient finetuning methods covering over 40 papers published between February 2019
and February 2023.

| Year | Title                                                                | Venue | Paper                                    | Code |
| ---- | -------------------------------------------------------------------- | ----- | ---------------------------------------- | ---- |
| 2023 | Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning | -     | [Link](https://arxiv.org/abs/2303.15647) | -    |

##### 1.Distillation

The method involves training a smaller model to mimic the behavior of a larger pre-trained model. The pre-trained model generates "teacher" predictions, which are then used to train the smaller "student"
model. By doing so, the student model can learn from the knowledge of the larger model without needing to store all of its parameters.

##### 2.Adapter training

Adapters are small neural networks added to pre-trained models for fine-tuning on specific tasks. These adapters only occupy a small portion of the original model's size, which enables faster training and
lower memory requirements. Adapters can be trained for multiple tasks and then inserted into the pre-trained model to perform new tasks.

##### 3.Progressive shrinking

This technique involves gradually reducing the size of the pre-trained model during fine-tuning. Starting from a large model, the number of parameters is gradually decreased until the desired performance
is achieved. This approach can result in smaller models with better performance compared to training from scratch.

##### 4.Paper of PEFT

| Adapter  | Title                                                                   | Paper                                        | Code                                                             |
| -------- | ----------------------------------------------------------------------- | -------------------------------------------- | ---------------------------------------------------------------- |
| LoRA     | LORA:LOW-RANK ADAPTATION OF LARGE LANGUAGE MODEL                        | [Link](https://arxiv.org/pdf/2106.09685.pdf) | [Link](https://github.com/microsoft/LoRA)                        |
| AdapterH | Parameter-Efficient Transfer Learning for NLP                           | [Link](https://arxiv.org/pdf/1902.00751.pdf) | [Link](https://github.com/google-research/adapter-bert)          |
| AdapterP | MAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer | [Link](https://arxiv.org/pdf/2005.00052.pdf) | [Link](https://adapterhub.ml/)                                   |
| Parallel | TOWARDS A UNIFIED VIEW OF PARAMETER-EFFICIENT TRANSFER LEARNING         | [Link](https://arxiv.org/pdf/2110.04366.pdf) | [Link](https://github.com/jxhe/unify-parameter-efficient-tuning) |

#### 1.3.3 Prompt-Tuning

Prompt-tuning is a more recent approach to fine-tuning pre-trained language models that focuses on adjusting the input prompts rather than modifying the model parameters. This means that the pre-trained model
remains unchanged, and only the input prompts are modified to adapt to downstream tasks. By designing and optimizing a set of prompts, the pre-trained model can be made to perform specific tasks effectively.
The main difference between prompt-tuning and traditional fine-tuning lies in the extent to which the pre-trained model is modified. While fine-tuning modifies the model's weights, prompt-tuning only adjusts
the model's input. As a result, prompt-tuning incurs lower computational costs, requires fewer resources, and takes less training time compared to fine-tuning. Additionally, prompt-tuning is more flexible than
fine-tuning because it allows for the creation of task-specific prompts that can adapt to a wide range of tasks.

##### 1.3.3.1 Prefix tuning

Proposed by Li and Liang in the paper "Prefix-Tuning: Optimizing Continuous Prompts for Generation" (2021), prefix-tuning involves learning continuous prompts specific to a task and adding them before the input
during the inference process. By optimizing this continuous prompt, the model can adapt to specific tasks without modifying the underlying model parameters, resulting in computational resource savings and achieving
efficient fine-tuning.

##### 1.3.3.2 P-Tuning

Proposed by Liu et al. in the paper "P-Tuning: GPT Understands, Learns, and Generates Any Language" (2021), P-Tuning involves training learnable parameters called "prompt tokens" that are concatenated with the input
sequence. These prompt tokens are task-specific and optimized during the fine-tuning process, enabling the model to perform well on new tasks while keeping the original model parameters unchanged.

| Adapter       | Title                                                                                          | Paper                                               | Code                                                |
| ------------- | ---------------------------------------------------------------------------------------------- | --------------------------------------------------- | --------------------------------------------------- |
| Prefix Tuning | Prefix-Tuning: Optimizing Continuous Prompts for Generation                                    | [Link](https://aclanthology.org/2021.acl-long.353/) | [Link](https://github.com/XiangLi1999/PrefixTuning) |
| P-Tuning      | GPT Understands, Too                                                                           | [Link](https://arxiv.org/pdf/2103.10385.pdf)        | [Link](https://github.com/THUDM/P-tuning)           |
| P-Tuning v2   | P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Task | [Link](https://arxiv.org/pdf/2110.07602.pdf)        | [Link](https://github.com/THUDM/P-tuning-v2)        |
| Prompt Tuning | The Power of Scale for Parameter-Efficient Prompt Tuning                                       | [Link](https://arxiv.org/pdf/2104.08691.pdf)        | -                                                   |

#### 1.3.4 RLHF Related Papers

**2023**

| Year | Title                                                                                     | Venue | Paper                                    | Code                                            |
| ---- | ----------------------------------------------------------------------------------------- | ----- | ---------------------------------------- | ----------------------------------------------- |
| 2023 | GPT-4 Technical Report                                                                    | -     | [link](https://arxiv.org/abs/2303.08774) | -                                               |
| 2023 | RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment                  | -     | [link](https://arxiv.org/abs/2304.06767) | -                                               |
| 2023 | RRHF: Rank Responses to Align Language Models with Human Feedback without tears           | -     | [link](https://arxiv.org/abs/2304.05302) | [link](https://github.com/GanjinZero/RRHF)      |
| 2023 | Better Aligning Text-to-Image Models with Human Preference                                | -     | [link](https://arxiv.org/abs/2303.14420) | [link](https://tgxs002.github.io/align_sd_web/) |
| 2023 | ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation       | -     | [link](https://arxiv.org/abs/2304.05977) | [link](https://github.com/THUDM/ImageReward)    |
| 2023 | Aligning Text-to-Image Models using Human Feedback                                        | -     | [link](https://arxiv.org/abs/2302.12192) | -                                               |
| 2023 | Visual ChatGPT: Talking, Drawing and Editing with Visual Foundation Models                | -     | [link](https://arxiv.org/abs/2303.04671) | [link](https://github.com/microsoft/TaskMatrix) |
| 2023 | Pretraining Language Models with Human Preferences                                        | -     | [link](https://arxiv.org/abs/2302.08582) | -                                               |
| 2023 | Aligning Language Models with Preferences through f-divergence Minimization               | -     | [link](https://arxiv.org/abs/2302.08215) | -                                               |
| 2023 | Principled Reinforcement Learning with Human Feedback from Pairwise or K-wise Comparisons | -     | [link](https://arxiv.org/abs/2301.11270) | -                                               |
| 2023 | The Capacity for Moral Self-Correction in Large Language Models                           |       | [link](https://arxiv.org/abs/2302.07459) | -                                               |
|      |                                                                                           |       |                                          |                                                 |

**2022** [[Back to Top⇪](#RLHF-Related-Papers)]

| Year | Title                                                                                        | Venue | Paper                                    | Code                                                              |
| ---- | -------------------------------------------------------------------------------------------- | ----- | ---------------------------------------- | ----------------------------------------------------------------- |
| 2022 | Few-shot Preference Learning for Human-in-the-Loop RL                                        | -     | [link](https://arxiv.org/abs/2212.03363) | [link](https://sites.google.com/view/few-shot-preference-rl/home) |
| 2022 | Improving alignment of dialogue agents via targeted human judgements                         | -     | [link](https://arxiv.org/abs/2209.14375) | -                                                                 |
| 2022 | Scaling Laws for Reward Model Overoptimization                                               | -     | [link](https://arxiv.org/abs/2210.10760) | -                                                                 |
| 2022 | Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned | -     | [link](https://arxiv.org/abs/2209.07858) | -                                                                 |
| 2022 | Dynamic Planning in Open-Ended Dialogue using Reinforcement Learning                         | -     | [link](https://arxiv.org/abs/2208.02294) | -                                                                 |
| 2022 | Quark: Controllable Text Generation with Reinforced Unlearning                               | -     | [link](https://arxiv.org/abs/2205.13636) | -                                                                 |
| 2022 | Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback    | -     | [link](https://arxiv.org/abs/2204.05862) | [link](https://github.com/anthropics/hh-rlhf)                     |
| 2022 | Training language models to follow instructions with human feedback                          | -     | [link](https://arxiv.org/abs/2203.02155) | -                                                                 |
| 2022 | Discovering Language Model Behaviors with Model-Written Evaluations                          | -     | [link](https://arxiv.org/abs/2212.09251) | [link](https://github.com/anthropics/evals)                       |
|      |                                                                                              |       |                                          |                                                                   |

**2021**

| Year | Title                                                                              | Venue        | Paper                                    | Code |
| ---- | ---------------------------------------------------------------------------------- | ------------ | ---------------------------------------- | ---- |
| 2021 | Recursively Summarizing Books with Human Feedback                                  | -            | [link](https://arxiv.org/abs/2109.10862) | -    |
| 2021 | Revisiting the Weaknesses of Reinforcement Learning for Neural Machine Translation | -            | [link](https://arxiv.org/abs/2106.08942) | -    |
| 2020 | Learning to summarize from human feedback                                          | NeurIPS 2020 | [link](https://arxiv.org/abs/2009.01325) | -    |

### 1.4 LLM Tutorial Resources and Evaluation Comparison

In this section, We've collected existing LLM tutorial resources, as well as large model evaluation comparisons.

-   [The Ultimate Chat GPT Course](https://www.notion.so/69ed24a317a942d288e740419b1ad6f6) - 这个指导课程有 1000 多个资源，帮助你学习如何使用 ChatGPT 来提高你的生活。(免费！)
-   [Advanced ChatGPT: Full Guide:](https://www.notion.so/ac6aa68840bc427c83f4611dd2642f83) - 他的指南包括初级和高级 ChatGPT 教程，以及一些实用的技巧和例子。(免费！)
-   [ChatGPT Tutorial - A Crash Course on Chat GPT for Beginners:](https://www.youtube.com/watch?v=JTxsNm9IdYU) - 本视频解释了 ChatGPT 的基本概念和用法。你将学习如何使用 ChatGPT 生成各种类型的文本，如购物清单、JavaScript 代码、小故事、简历等。(免费！)
-   [Complete ChatGPT Tutorial - [Become A Power User in 30 Minutes\]](https://www.youtube.com/watch?v=jHv63Uvk5VA) - 你可以学习 10 大类命令，让 ChatGPT 为你提供各种有用的信息和服务。(免费！)
-   [ChatGPT Tutorial for Developers - 38 Ways to 10x Your Productivity:](https://www.youtube.com/watch?v=sTeoEFzVNSc) - 38 个 ChatGPT 实例，帮助你学习如何使用 Python、JavaScript、HTML、CSS、React、SQL 等。(免费！)
-   [ChatGPT Prompt Engineering for Developers](https://learn.deeplearning.ai/chatgpt-prompt-eng/lesson/1/introduction) 吴恩达讲 prompt
-   [Examples of Prompts | Prompt Engineering Guide (promptingguide.ai)](https://www.promptingguide.ai/introduction/examples)

-   [ChatGPT 保姆级使用教程：注册、体验、底层逻辑原理解读！](https://www.bilibili.com/video/BV1HT411R7Lj/) ChatGPT 介绍，偏底技术，偏底层原理。
-   [【渐构】万字科普 ChatGPT-4 为什么会颠覆人类社会](https://www.bilibili.com/video/BV1MY4y1R7EN/?spm_id_from=333.880.my_history.page.click&vd_source=6faef52e732ccc3a4a525fe406ce9808): 视频前 25 分钟深入浅出的讲解了 GPT 原理
-   [ChatGPT (可能)是怎麼煉成的 - GPT 社會化的過程](https://www.bilibili.com/video/BV1U84y167i3?p=1&vd_source=71b548de6de953e10b96b6547ada83f2)
-   [深度學習之應用 | ADL 17.3: OpenAI ChatGPT 驚驗眾人的對話互動式 AI](https://www.bilibili.com/video/BV1U84y167i3?p=3&vd_source=71b548de6de953e10b96b6547ada83f2)
-   [InstructGPT 论文精读【论文精读·48】](https://www.bilibili.com/video/BV1hd4y187CR/?spm_id_from=333.788&vd_source=71b548de6de953e10b96b6547ada83f2)
-   [chatgpt 基本工作原理简单清晰介绍](https://www.youtube.com/watch?v=e0aKI2GGZNg&t=24s)

### 1.5 LLM Related Papers: Prompt, Incontent Learning, and LLM PEFT e.t.

#### 1.5.1 Incontext Learning Papers

**2023**
| Year | Title | Venue | Paper | Code |
| ---- | ----------------------------------------------------------------------------------------------------------------------- | --------- | -------------------------------------------- | ------------------------------------------------------------------------------ |
| 2023 | Resources and Few-shot Learners for In-context Learning in Slavic Languages | EACL 2023 | [link](https://arxiv.org/pdf/2304.01922.pdf) | [link](https://github.com/fewshot-goes-multilingual/slavic-incontext-learning) |
| 2023 | Improving Language Model Negotiation with Self-Play and In-Context Learning from AI Feedback | Pre-print | [link](https://arxiv.org/pdf/2305.10142.pdf) | [link](https://github.com/FranxYao/GPT-Bargaining) |
| 2023 | Least-to-Most Prompting Enables Complex Reasoning in Large Language Models | ICLR 2023 | [link](https://arxiv.org/pdf/2205.10625.pdf) | - |
| 2023 | Prompting GPT-3 To Be Reliable | ICLR 2023 | [link](https://arxiv.org/pdf/2210.09150.pdf) | [link](https://github.com/NoviScl/GPT3-Reliability) |
| 2023 | Self-Adaptive In-Context Learning: An Information Compression Perspective for In-Context Example Selection and Ordering | ICLR 2023 | [link](https://arxiv.org/pdf/2212.10375.pdf) | [link](https://github.com/Shark-NLP/self-adaptive-ICL) |
| 2023 | On the Relation between Sensitivity and Accuracy in In-Context Learning | - | [link](https://arxiv.org/pdf/2209.07661.pdf) | [link](https://github.com/yandachen/ICLSensitivity) |
| 2023 | Why Can GPT Learn In-Context? Language Models Implicitly Perform Gradient Descent as Meta-Optimizers | ACL 2023 | [link](https://arxiv.org/pdf/2212.10559.pdf) | [link](https://github.com/microsoft/LMOps) |
| 2023 | Transformers as Algorithms: Generalization and Stability in In-context Learning | - | [link](https://arxiv.org/pdf/2301.07067.pdf) | [link](https://github.com/yingcong-li/transformers-as-algorithms) |
| 2023 | Can In-context Learners Learn a Reasoning Concept from Demonstrations? | - | [link](https://arxiv.org/pdf/2212.01692.pdf) | - |
| 2023 | The Flan Collection: Designing Data and Methods for Effective Instruction Tuning | - | [link](https://arxiv.org/pdf/2301.13688.pdf) | [link](https://github.com/google-research/FLAN/tree/main/flan/v2) |

**2022**

| Year | Title                                                                                                                                                     | Venue          | Paper                                                          | Code                                                                              |
| ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- | -------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| 2022 | What learning algorithm is in-context learning? Investigations with linear models                                                                         | -              | [link](https://arxiv.org/pdf/2211.15661.pdf)                   | [link](https://github.com/ekinakyurek/google-research/tree/master/incontext)      |
| 2022 | A Survey for In-context Learning                                                                                                                          | -              | [link](https://arxiv.org/pdf/2301.00234.pdf)                   | -                                                                                 |
| 2022 | MetaICL: Learning to Learn In Context NAACL 2022 a pretrained language model is tuned to do in-context learning on a large set of training tasks          | NACCL 2022     | [link](https://arxiv.org/pdf/2110.15943.pdf)                   | [link](https://github.com/dongguanting/In-Context-Learning_PaperList/tree/master) |
| 2022 | Improving In-Context Few-Shot Learning via Self-Supervised Training                                                                                       | NACCL 2022     | [link](https://aclanthology.org/2022.naacl-main.260.pdf)       | [link](https://github.com/dongguanting/In-Context-Learning_PaperList/tree/master) |
| 2022 | Calibrate Before Use: Improving Few-shot Performance of Language Models.                                                                                  | ICML 2021      | [link](http://proceedings.mlr.press/v139/zhao21c.html)         | [link](https://img.shields.io/badge/additional_calibration_parameters-D8D0E1)     |
| 2022 | Chain-of-Thought Prompting Elicits Reasoning in Large Language Models                                                                                     | NeurIPS 2022   | [link](https://arxiv.org/abs/2201.11903)                       | -                                                                                 |
| 2022 | Self-Generated In-Context Learning: Leveraging Auto-regressive Language Models as a Demonstration Generator                                               | NACCL 2022     | [link](https://arxiv.org/pdf/2206.08082.pdf)                   | -                                                                                 |
| 2022 | Iteratively Prompt Pre-trained Language Models for Chain of Though                                                                                        | EMNLP 2022     | [link](https://arxiv.org/pdf/2203.08383.pdf)                   | [link](https://github.com/sunlab-osu/IterPrompt)                                  |
| 2022 | Automatic Chain of Thought Prompting in Large Language Models                                                                                             | -              | [link](https://arxiv.org/abs/2210.03493)                       | [link](https://github.com/amazon-science/auto-cot)                                |
| 2022 | Learning To Retrieve Prompts for In-Context Learning                                                                                                      | NAACL-HLT 2022 | [link](https://arxiv.org/pdf/2112.08633.pdf)                   | [link](https://github.com/OhadRubin/EPR)                                          |
| 2022 | Finetuned Language Models Are Zero-Shot Learners instruction tuning.                                                                                      | ICLR 2022      | [link](https://arxiv.org/abs/2109.01652)                       | [link](https://github.com/google-research/flan)                                   |
| 2022 | Active Example Selection for In-Context Learning.                                                                                                         | EMNLP 2022     | [link](https://arxiv.org/pdf/2211.04486.pdf)                   | [link](https://github.com/chicagohai/active-example-selection)                    |
| 2022 | An Information-theoretic Approach to Prompt Engineering Without Ground Truth Labels                                                                       | ACL 2022       | [link](https://aclanthology.org/2022.acl-long.60.pdf)          | [link](https://github.com/BYU-PCCL/information-theoretic-prompts)                 |
| 2022 | Demystifying Prompts in Language Models via Perplexity Estimation                                                                                         | -              | [link](https://arxiv.org/pdf/2212.04037.pdf)                   | [link](https://github.com/bigscience-workshop/promptsource)                       |
| 2022 | Structured Prompting:Scaling In-Context Learning to 1,000 Examples                                                                                        | -              | [link](https://arxiv.org/pdf/2212.06713.pdf)                   | [link](https://github.com/microsoft/LMOps)                                        |
| 2022 | Fantastically Ordered Prompts and Where to Find Them:Overcoming Few-Shot Prompt Order Sensitivity                                                         | ACL 2022       | [link](https://arxiv.org/pdf/2104.08786.pdf)                   | -                                                                                 |
| 2022 | Can language models learn from explanations in context?                                                                                                   | -              | [link](https://arxiv.org/pdf/2204.02329.pdf)                   | -                                                                                 |
| 2022 | Prototypical Calibration for Few-shot Learning of Language Models                                                                                         | -              | [link](https://arxiv.org/pdf/2205.10183.pdf)                   | [link](https://github.com/microsoft/unilm)                                        |
| 2022 | Cross-Task Generalization via Natural Language Crowdsourcing Instructions                                                                                 | ACL 2022       | [link](https://arxiv.org/pdf/2104.08773.pdf)                   | [link](https://github.com/allenai/natural-instructions-v1)                        |
| 2022 | Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?                                                                               | EMNLP 2022     | [link](https://arxiv.org/pdf/2202.12837.pdf)                   | [link](https://github.com/Alrope123/rethinking-demonstrations)                    |
| 2022 | Emergent Abilities of Large Language Models                                                                                                               | TMLR 2022      | [link](https://arxiv.org/pdf/2206.07682.pdf)                   | [link](https://github.com/inverse-scaling/prize)                                  |
| 2022 | Ground-Truth Labels Matter:A Deeper Look into Input-Label Demonstrations                                                                                  | EMNLP 2022     | [link](https://arxiv.org/pdf/2205.12685.pdf)                   | -                                                                                 |
| 2022 | On the Effect of Pretraining Corpora on In-context Learning by a Large-scale Language Model                                                               | NAACL 2022     | [link](https://arxiv.org/pdf/2204.13509.pdf)                   | -                                                                                 |
| 2022 | Rethinking the Role of Scale for In-Context Learning: An Interpretability-based Case Study at 66 Billion Scale                                            | -              | [link](https://arxiv.org/pdf/2212.09095.pdf)                   | [link](https://github.com/amazon-science/llm-interpret)                           |
| 2022 | Data Distributional Properties Drive Emergent In-Context Learning in Transformers                                                                         | NeurIPS 2022   | [link](https://arxiv.org/pdf/2205.05055.pdf)                   | [link](https://github.com/deepmind/emergent_in_context_learning)                  |
| 2022 | Diverse Demonstrations Improve In-context Compositional Generalization                                                                                    | -              | [link](https://arxiv.org/pdf/2212.06800.pdf)                   | [link](https://github.com/itayle/diverse-demonstrations)                          |
| 2022 | Towards Understanding Chain-of-Thought Prompting:An Empirical Study of What Matters                                                                       | -              | [link](https://arxiv.org/pdf/2212.10001.pdf)                   | [link](https://github.com/sunlab-osu/Understanding-CoT)                           |
| 2022 | An Explanation of In-context Learning as Implicit Bayesian Inference                                                                                      | ICLR 2022      | [link](https://arxiv.org/pdf/2111.02080.pdf)                   | [link](https://github.com/p-lambda/incontext-learning)                            |
| 2022 | In-context Learning and Induction Heads                                                                                                                   | -              | [link](https://arxiv.org/ftp/arxiv/papers/2209/2209.11895.pdf) | -                                                                                 |
| 2022 | What Can Transformers Learn In-Context? A Case Study of Simple Function Classes                                                                           | NeurIPS 2022   | [link](https://openreview.net/pdf?id=flNZJ2eOet)               | [link](https://github.com/dtsip/in-context-learning)                              |
| 2022 | Data Distributional Properties Drive Emergent In-Context Learning in Transformers                                                                         | NeurIPS 2022   | [link](https://arxiv.org/pdf/2205.05055.pdf)                   | [link](https://github.com/deepmind/emergent_in_context_learning)                  |
| 2022 | What learning algorithm is in-context learning? Investigations with linear models                                                                         | -              | [link](https://arxiv.org/pdf/2211.15661.pdf)                   | [link](https://github.com/ekinakyurek/google-research/tree/master/incontext)      |
| 2022 | Transformers learn in-context by gradient descent                                                                                                         | -              | [link](https://arxiv.org/pdf/2212.07677.pdf)                   | -                                                                                 |
| 2022 | Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models                                                              | -              | [link](https://arxiv.org/pdf/2206.04615.pdf)                   | [link](https://github.com/google/BIG-bench)                                       |
| 2022 | SUPER-NATURALINSTRUCTIONS: Generalization via Declarative Instructions on 1600+ NLP Task.                                                                 | EMNLP 2022     | [link](https://arxiv.org/pdf/2204.07705.pdf)                   | [link](https://github.com/allenai/natural-instructions)                           |
| 2022 | Language Models are Multilingual Chain-of-Thought Reasoners.                                                                                              | -              | [link](https://arxiv.org/pdf/2210.03057.pdf)                   | [link](https://github.com/google-research/url-nlp)                                |
| 2022 | Instruction Induction: From Few Examples to Natural Language Task Descriptions                                                                            | -              | [link](https://arxiv.org/pdf/2205.10782.pdf)                   | [link](https://github.com/orhonovich/instruction-induction)                       |
| 2022 | Unnatural Instructions: Tuning Language Models with (Almost) No Human Labor.                                                                              | -              | [link](https://arxiv.org/pdf/2212.09689.pdf)                   | [link](https://github.com/orhonovich/unnatural-instructions)                      |
| 2022 | SELF-INSTRUCT: Aligning Language Model with Self Generated Instructions                                                                                   | -              | [link](https://arxiv.org/pdf/2212.10560.pdf)                   | [link](https://github.com/yizhongw/self-instruct)                                 |
| 2022 | Meta-learning via Language Model In-context Tuning                                                                                                        | ACL 2022       | [link](https://arxiv.org/pdf/2110.07814.pdf)                   | [link](https://github.com/yandachen/In-context-Tuning)                            |
| 2022 | Does GPT-3 Generate Empathetic Dialogues? A Novel In-Context Example Selection Method and Automatic Evaluation Metric for Empathetic Dialogue Generation. | COLING 2022    | [link](https://aclanthology.org/2022.coling-1.56.pdf)          | [link](https://github.com/passing2961/EmpGPT-3)                                   |
| 2022 | In-context Learning Distillation: Transferring Few-shot Learning Ability of Pre-trained Language Models.                                                  | -              | [link](https://arxiv.org/pdf/2212.10670.pdf)                   | -                                                                                 |
| 2022 | The Inductive Bias of In-Context Learning: Rethinking Pretraining Example Design                                                                          | ICLR 2022      | [link](https://arxiv.org/pdf/2110.04541.pdf)                   | -                                                                                 |

#### 1.5.2 Prompt-based Papers

**2023**

| Year | Title                                                                                                   | Venue     | Paper                                              | Code                                                                                     |
| ---- | ------------------------------------------------------------------------------------------------------- | --------- | -------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| 2023 | Self-Supervised Convolutional Visual Prompts                                                            | -         | [link](https://arxiv.org/abs/2303.00198)           | -                                                                                        |
| 2023 | Multimodal Prompting with Missing Modalities for Visual Recognition                                     | CVPR 2023 | [link](https://arxiv.org/abs/2303.03369)           | [link](https://github.com/YiLunLee/Missing_aware_prompts)                                |
| 2023 | From Visual Prompt Learning to Zero-Shot Transfer: Mapping Is All You Need                              | -         | [link](https://arxiv.org/abs/2303.05266)           | -                                                                                        |
| 2023 | Diversity-Aware Meta Visual Prompting                                                                   | CVPR 2023 | [link](https://arxiv.org/abs/2303.08138)           | [link](https://github.com/shikiw/DAM-VP)                                                 |
| 2023 | Patch-Token Aligned Bayesian Prompt Learning for Vision-Language Models                                 | -         | [link](https://arxiv.org/abs/2303.09100)           | -                                                                                        |
| 2023 | Visual Prompt Multi-Modal Tracking                                                                      | CVPR 2023 | [link](https://arxiv.org/abs/2303.10826)           | -                                                                                        |
| 2023 | Explicit Visual Prompting for Low-Level Structure Segmentations                                         | CVPR 2023 | [link](https://arxiv.org/abs/2303.10826)           | [link](https://github.com/NiFangBaAGe/Explict-Visual-Prompt)                             |
| 2023 | Multi-modal Prompting for Low-Shot Temporal Action Localization                                         | -         | [link](https://arxiv.org/abs/2303.11732)           | -                                                                                        |
| 2023 | LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention                        | -         | [link](https://arxiv.org/abs/2303.16199)           | [link](https://github.com/ZrrSkywalker/LLaMA-Adapter)                                    |
| 2023 | Zero-shot Generative Model Adaptation via Image-specific Prompt Learning                                | -         | [link](https://arxiv.org/abs/2304.03119)           | [link](https://github.com/Picsart-AI-Research/IPL-Zero-Shot-Generative-Model-Adaptation) |
| 2023 | Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing | -         | [link](https://dl.acm.org/doi/pdf/10.1145/3560815) | -                                                                                        |

**2022**
| Year | Title | Venue | Paper | Code |
| ---- | ----------------------------------------------------------------------------------------------- | --------- | -------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| 2022 | Domain Adaptation via Prompt Learning | - | [link](https://arxiv.org/abs/2202.06687) | - |
| 2022 | Conditional Prompt Learning for Vision-Language Models | CVPR 2022 | [link](https://openaccess.thecvf.com/content/CVPR2022/html/Zhou_Conditional_Prompt_Learning_for_Vision-Language_Models_CVPR_2022_paper.html) | [link](https://github.com/KaiyangZhou/CoOp) |
| 2022 | Visual Prompt Tuning | ECCV 2022 | [link](https://arxiv.org/abs/2203.12119) | [link](https://github.com/kmnp/vpt) |
| 2022 | Pro-tuning: Unified Prompt Tuning for Vision Tasks | - | [link](https://arxiv.org/abs/2207.14381) | - |
| 2022 | Unified Vision and Language Prompt Learning | - | [link](https://arxiv.org/abs/2210.07225) | [link](https://github.com/yuhangzang/UPT) |
| 2022 | CPL: Counterfactual Prompt Learning for Vision and Language Models | - | [link](https://arxiv.org/abs/2210.10362) | [link](https://github.com/eric-ai-lab/CPL) |
| 2022 | Texts as Images in Prompt Tuning for Multi-Label Image Recognition | - | [link](https://arxiv.org/abs/2211.12739) | [link](https://github.com/guozix/TaI-DPT) |
| 2022 | VoP: Text-Video Co-operative Prompt Tuning for Cross-Modal Retrieval | - | [link](https://arxiv.org/abs/2211.12764) | [link](https://github.com/bighuang624/VoP) |
| 2022 | P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks | - | [link](https://arxiv.org/abs/2110.07602) | [link](https://github.com/THUDM/P-tuning-v2) |

**2021**

| Year | Title                                                              | Venue      | Paper                                    | Code                                                     |
| ---- | ------------------------------------------------------------------ | ---------- | ---------------------------------------- | -------------------------------------------------------- |
| 2021 | Learning to Prompt for Vision-Language Models(CoOP)                | IJCV 2022  | [link](https://arxiv.org/abs/2109.01134) | [link](https://github.com/KaiyangZhou/CoOp)              |
| 2021 | Prompting Visual-Language Models for Efficient Video Understanding | ECCV 2022  | [link](https://arxiv.org/abs/2112.04478) | [link](https://github.com/ju-chen/Efficient-Prompt)      |
| 2021 | P-Tuning: GPT Understands, Too                                     | -          | [link](https://arxiv.org/abs/2103.10385) | [link](https://github.com/THUDM/P-tuning)                |
| 2021 | Prefix-Tuning Optimizing Continuous Prompts for Generation         | ACL 2021   | [link](https://arxiv.org/abs/2101.00190) | [link](https://github.com/XiangLi1999/PrefixTuning)      |
| 2021 | The Power of Scale for Parameter-Efficient Prompt Tuning           | EMNLP 2021 | [link](https://arxiv.org/abs/2104.08691) | [link](https://github.com/google-research/prompt-tuning) |

## 2.Tutorials

## Contributions

<p align="center"><a href="https://github.com/huaiwen"><img src="https://avatars.githubusercontent.com/u/3187529?v=4" width="50px" alt="robinicus" /></a>&nbsp;&nbsp;<a href="https://github.com/guozihang"><img src="https://avatars.githubusercontent.com/u/17142416?v=4" width="50px" alt="0xmatchmaker" /></a>&nbsp;&nbsp;<a href="https://github.com/LeeRoc-China"><img src="https://avatars.githubusercontent.com/u/59104898?s=400&u=c225a082a6a410e3d7c84ca29a07d723d7308dca&v=4" width="50px" alt="0xmatchmaker" /></a>&nbsp;&nbsp;<a href="https://github.com/YangYang"><img src="https://avatars.githubusercontent.com/u/17808880?v=4" width="50px" alt="0xmatchmaker" /></a>&nbsp;&nbsp;<a href="https://github.com/yangqqq-yq"><img src="https://avatars.githubusercontent.com/u/64053857?v=4" width="50px" alt="0xmatchmaker" /></a>&nbsp;&nbsp;</p>
