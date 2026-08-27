# S2S End-to-End Voice Agent
This repository implements an end-to-end multimodal AI pipeline that directly bridges continuous acoustic waveforms with the discrete semantic latent space of Large Language Models (LLMs). Rather than relying on traditional cascading systems (where a standalone ASR transcribes text and feeds it to an LLM), this architecture natively aligns audio vectors into the language model's embedding space, enabling the LLM to directly "listen" to and process human speech.

The current pipeline is specifically tuned for Hinglish (Hindi-English) speech recognition and foundational voice-agent adaptation, utilizing a parameter-efficient fine-tuning (PEFT) approach optimized for consumer hardware.

- [Architecture](#architecture)
- [Multi-stage Training Pipeline](#multi-stage-training-pipeline)
- [Funstions](#functions)

### Architecture

### Multi-stage training pipeline

#### Stage 1: Latent Space Alignment
The LLM is completely frozen. The MLP projector is trained in isolation using Mean Squared Error (MSE) loss to map the FastConformer's acoustic embeddings directly to the ground-truth text token embeddings extracted from Qwen's internal lookup tables.

#### Stage 2: End-To-End Adaption
The pre-trained projector is plugged into the LLM via the inputs_embeds layer. Low-Rank Adaptation (LoRA) modules are injected into Qwen's attention mechanisms (Query, Key, Value, Output projections). The system is jointly optimized using Cross-Entropy Loss to perform Automatic Speech Recognition (ASR), teaching the LLM's attention heads to extract meaning from continuous audio tensors.

#### Stage 3: Instruction Fine-tuning
With foundational speech recognition established, the system will be trained on complex prompt-completion pairs to function as an interactive, instruction-following voice agent.

### Functions