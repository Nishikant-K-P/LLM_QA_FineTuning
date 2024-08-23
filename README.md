# Comparative Study for Fine-Tuning Large Language Models (LLMs)

## Abstract

This project compares two methodologies for optimizing a large language model tailored for a specific task. The study focuses on:
- Fully fine-tuning a smaller model, **MobileBert**.
- Parameter-efficient fine-tuning (PEFT) of a larger model, **DistillBert**, using the **Low-Rank Adaptation (LoRA)** algorithm.

The primary objective is to evaluate the performance of both models on the task of Extractive Question Answering, using the **Stanford Question Answering Dataset (SQuAD)**. Results indicate that the fully fine-tuned MobileBert significantly outperforms the PEFT-trained DistillBert in terms of Exact Match (EM) and F1 Score.

## Introduction

Transformer models have introduced a new family of models capable of impressive results when fine-tuned for specific tasks. However, this process is computationally intensive, necessitating optimization to enable efficient model training. **Parameter-efficient fine-tuning (PEFT)** is an emerging technique that partially fine-tunes the model, reducing resource requirements.

This study investigates whether partially fine-tuning a relatively larger model (DistillBert) can outperform fully fine-tuning a smaller model (MobileBert) for a specific task and dataset.

## Method

### Models
- **MobileBert**: ~26 million parameters
- **DistillBert**: ~66 million parameters

### Task
- **Extractive Question Answering**: The model is provided with a context and a question. The answer lies within the given context.

### Dataset
- **Stanford Question Answering Dataset (SQuAD)**: ~150,000 question-answer pairs.

### Process
1. **Preprocessing**: Tokenizing contexts, handling context size overflow, truncation, and padding.
2. **Training**: Fine-tuning MobileBert fully and DistillBert partially using LoRA.
3. **Postprocessing**: Calculating start and end logits for answers.
4. **Evaluation**: Calculating Exact Match (EM) and F1 scores.

## Experiments and Results

- **DistillBert Performance**: LoRA ranks from 32 to 1024 showed minimal variation in EM and F1 scores.
- **Comparison**: MobileBert outperformed DistillBert, achieving significantly higher EM and F1 scores.
- **Impact of Data Volume**: Increasing training data improved MobileBert’s performance more significantly than DistillBert’s.

## Conclusion

The fully fine-tuned MobileBert outperformed the partially fine-tuned DistillBert in all metrics. While increasing data volume improved the performance of both models, MobileBert remained superior. These results suggest that, despite its lower parameter count, fully fine-tuning a smaller model may be more effective than partially fine-tuning a larger one.

## References

- [MobileBert](https://huggingface.co/google/mobilebert-uncased)
- [DistilBert](https://huggingface.co/distilbert/distilbert-base-uncased)
- [SQuAD Dataset](https://rajpurkar.github.io/SQuAD-explorer)
- [LoRA Algorithm](https://huggingface.co/docs/peft/en/package_reference/lora)
