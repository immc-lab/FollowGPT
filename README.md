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
2023.05.17 start project!
```

---

## Summary of GPT 
<h3><b>1.</b> <span style="font-weight: normal; font-size: medium;">Generative Pretrained Transformer (GPT) models have demonstrated significant proficiency in understanding and generating language, one of their standout characteristics is their versatility across domains. Regardless of the text data in question, GPT models can generate coherent, contextually accurate, and often insightful responses without being explicitly trained on the task. This adaptability is quite remarkable and sets GPT models apart in the landscape of language models.Another point of praise for GPT is its ability to engage in semantic search. Instead of merely matching keywords, as with traditional search algorithms, GPT can comprehend the meaning behind the words. This enables it to provide more relevant, precise, and contextually fitting results, highlighting its utility in data mining and information retrieval applications.</span></h3>

### 2.
### 3.
The use of pre trained large-scale models for fine-tuning downstream tasks is currently a popular deep learning paradigm. Especially with the outstanding performance of the recently pre trained language model ChatGPT, this technical paradigm has been widely recognized. We mainly answered the question of what big models are available, and from a macro perspective, introduced the readers to what big models are available and what scope they cover. We have collected relevant papers from 2021 to the present from three aspects of natural language processing, computer vision and multimodality. Large models started early in the field of natural language processing, and then gradually evolved into computer vision related fields, such as SAM, which can split everything. At present, integrating large models from multiple professional fields has become a new paradigm for training large models in multimodal fields, such as Huawei's Pangu model, which covers fields such as finance and meteorology. Our link[Resources3](./Resources/3)
### 4. How to Understand the Foundation Model & How to Experience the Foundation Model
In recent times, significant advancements have been made in the research of LLMs by both academia and industry. One notable achievement is the introduction of ChatGPT, a powerful AI chatbot developed based on LLMs, which has garnered considerable attention from society. The technical evolution of LLMs is having a profound impact on the entire AI community, revolutionizing the way we develop and utilize AI algorithms. 
For the convenience of those interested in the LLMs, we have found some papers, blogs, etc. in Part IV to answer the question of how to understand the foundation model, and some open source resources on the LLMs for the base model including but not limited to: **Publicly Available Model Checkpoints or APIs**, **Commonly Used Corpora**, **Library Resource**, etc. for people who can experience and replicate the foundation model. Our link [Resources4](https://github.com/immc-lab/FollowGPT/tree/main/Resources/4)
### 5. Paper of Prompt and Incontent Learning
In this section, we have collected all relevant papers on In-context Learning and Prompt Learning from 2021 to 2023.
### 6. LLM Fine Tuning Principle Introduction and Paper
Fine-tuning is a technique used in natural language processing (NLP) to adapt pre-trained language models to specific tasks or domains. In this part, we make a brief introduction to the fine tuning technology, and systematically introduce the fine tuning technology commonly used in LLM. At the same time, we made a summary of RLHF related technologies and collected relevant papers from 2021 to 2023.
### 7. LLM Tutorial Resources and Evaluation Comparison

## Tutorials



## Contributions

<p align="center"><a href="https://github.com/huaiwen"><img src="https://avatars.githubusercontent.com/u/3187529?v=4" width="50px" alt="robinicus" /></a>&nbsp;&nbsp;<a href="https://github.com/guozihang"><img src="https://avatars.githubusercontent.com/u/17142416?v=4" width="50px" alt="0xmatchmaker" /></a>&nbsp;&nbsp;<a href="https://github.com/LeeRoc-China"><img src="https://avatars.githubusercontent.com/u/59104898?s=400&u=c225a082a6a410e3d7c84ca29a07d723d7308dca&v=4" width="50px" alt="0xmatchmaker" /></a>&nbsp;&nbsp;<a href="https://github.com/YangYang"><img src="https://avatars.githubusercontent.com/u/17808880?v=4" width="50px" alt="0xmatchmaker" /></a>&nbsp;&nbsp;<a href="https://github.com/yangqqq-yq"><img src="https://avatars.githubusercontent.com/u/64053857?v=4" width="50px" alt="0xmatchmaker" /></a>&nbsp;&nbsp;</p>

