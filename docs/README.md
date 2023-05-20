# Summary of LLM fine-tuning
## Fine-Tuning
Fine-tuning is a technique used in natural language processing (NLP) to adapt pre-trained language models to specific tasks or domains. The basic idea of fine-tuning is to take a pre-trained 
language model that has been trained on a large amount of text and continue training it on a smaller set of task-specific text.The concept of fine-tuning has been around for many years and 
has been utilized in various contexts. One of the earliest known applications of fine-tuning in NLP was in the domain of Neural Machine Translation (NMT), where researchers used pre-trained 
neural networks to initialize the weights of a smaller network and then fine-tuned it for specific translation tasks.Classic fine-tuning methods involve continuing the training of a pre-trained 
model with a small amount of task-specific data. During this process, the weights of the pre-trained model are updated to better adapt to the task at hand. The amount of fine-tuning required 
depends on the similarity between the pre-training corpus and the task-specific corpus. If the two are similar, only a small amount of fine-tuning may be needed. If they are dissimilar, more 
extensive fine-tuning may be required.One of the most well-known examples of fine-tuning in NLP is the OpenAI GPT (Generative Pre-trained Transformer) model developed by OpenAI. The GPT model 
undergoes pre-training on a large corpus of text and is then fine-tuned on various tasks, such as language modeling, question answering, and summarization. The fine-tuned models have achieved 
state-of-the-art performance on these tasks.

## Parameter-Efficient Fine-Tuning (PEFT)
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

| Year | Title                                                                 | Venue | Paper                                      | Code |
|------|-----------------------------------------------------------------------|-------|--------------------------------------------|------|
| 2023 | Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning  | -     | [Link](https://arxiv.org/abs/2303.15647)   | -    |

### Distillation
The method involves training a smaller model to mimic the behavior of a larger pre-trained model. The pre-trained model generates "teacher" predictions, which are then used to train the smaller "student" 
model. By doing so, the student model can learn from the knowledge of the larger model without needing to store all of its parameters.
### Adapter training
Adapters are small neural networks added to pre-trained models for fine-tuning on specific tasks. These adapters only occupy a small portion of the original model's size, which enables faster training and
lower memory requirements. Adapters can be trained for multiple tasks and then inserted into the pre-trained model to perform new tasks.
### Progressive shrinking
This technique involves gradually reducing the size of the pre-trained model during fine-tuning. Starting from a large model, the number of parameters is gradually decreased until the desired performance 
is achieved. This approach can result in smaller models with better performance compared to training from scratch.

| Adapter       | Title                                                                                          | Paper                                              | Code                                                             |
|---------------|------------------------------------------------------------------------------------------------|----------------------------------------------------|------------------------------------------------------------------|
| LoRA          | LORA:LOW-RANK ADAPTATION OF LARGE LANGUAGE MODEL                                               | [Link](https://arxiv.org/pdf/2106.09685.pdf)       | [Link](https://github.com/microsoft/LoRA)                        |
| AdapterH      | Parameter-Efficient Transfer Learning for NLP                                                  | [Link](https://arxiv.org/pdf/1902.00751.pdf)       | [Link](https://github.com/google-research/adapter-bert)          |
| AdapterP      | MAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer                        | [Link](https://arxiv.org/pdf/2005.00052.pdf)       | [Link](https://adapterhub.ml/)                                   |
| Parallel      | TOWARDS A UNIFIED VIEW OF PARAMETER-EFFICIENT TRANSFER LEARNING                                | [Link](https://arxiv.org/pdf/2110.04366.pdf)       | [Link](https://github.com/jxhe/unify-parameter-efficient-tuning) |
## prompt-tuning
Prompt-tuning is a more recent approach to fine-tuning pre-trained language models that focuses on adjusting the input prompts rather than modifying the model parameters. This means that the pre-trained model 
remains unchanged, and only the input prompts are modified to adapt to downstream tasks. By designing and optimizing a set of prompts, the pre-trained model can be made to perform specific tasks effectively.
The main difference between prompt-tuning and traditional fine-tuning lies in the extent to which the pre-trained model is modified. While fine-tuning modifies the model's weights, prompt-tuning only adjusts 
the model's input. As a result, prompt-tuning incurs lower computational costs, requires fewer resources, and takes less training time compared to fine-tuning. Additionally, prompt-tuning is more flexible than 
fine-tuning because it allows for the creation of task-specific prompts that can adapt to a wide range of tasks.
### Prefix tuning
Proposed by Li and Liang in the paper "Prefix-Tuning: Optimizing Continuous Prompts for Generation" (2021), prefix-tuning involves learning continuous prompts specific to a task and adding them before the input 
during the inference process. By optimizing this continuous prompt, the model can adapt to specific tasks without modifying the underlying model parameters, resulting in computational resource savings and achieving 
efficient fine-tuning.
### P-Tuning
Proposed by Liu et al. in the paper "P-Tuning: GPT Understands, Learns, and Generates Any Language" (2021), P-Tuning involves training learnable parameters called "prompt tokens" that are concatenated with the input 
sequence. These prompt tokens are task-specific and optimized during the fine-tuning process, enabling the model to perform well on new tasks while keeping the original model parameters unchanged.

| Adapter       | Title                                                                                          | Paper                                              | Code                                                             |
|---------------|------------------------------------------------------------------------------------------------|----------------------------------------------------|------------------------------------------------------------------|
| Prefix Tuning | Prefix-Tuning: Optimizing Continuous Prompts for Generation                                    | [Link](https://aclanthology.org/2021.acl-long.353/) | [Link](https://github.com/XiangLi1999/PrefixTuning)              |
| P-Tuning      | GPT Understands, Too                                                                           | [Link](https://arxiv.org/pdf/2103.10385.pdf)       | [Link](https://github.com/THUDM/P-tuning)                        |
| P-Tuning v2   | P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Task | [Link](https://arxiv.org/pdf/2110.07602.pdf)       | [Link](https://github.com/THUDM/P-tuning-v2)                     |
| Prompt Tuning | The Power of Scale for Parameter-Efficient Prompt Tuning                                       | [Link](https://arxiv.org/pdf/2104.08691.pdf)       | -                                                                |
