# S2S End-to-End Voice Agent
This repository implements an end-to-end multimodal AI pipeline that directly bridges continuous acoustic waveforms with the discrete semantic latent space of Large Language Models (LLMs). Rather than relying on traditional cascading systems (where a standalone ASR transcribes text and feeds it to an LLM), this architecture natively aligns audio vectors into the language model's embedding space, enabling the LLM to directly "listen" to and process human speech.

The current pipeline is specifically tuned for Hinglish (Hindi-English) speech recognition and foundational voice-agent adaptation, utilizing a parameter-efficient fine-tuning (PEFT) approach optimized for consumer hardware.

- [Architecture](#architecture)
- [Multi-stage Training Pipeline](#multi-stage-training-pipeline)
- [Funstions](#functions)

### Architecture
graph TD
    classDef encoder fill:#2E3440,stroke:#88C0D0,stroke-width:2px,color:#D8DEE9;
    classDef projector fill:#3B4252,stroke:#EBCB8B,stroke-width:2px,color:#D8DEE9;
    classDef llm fill:#434C5E,stroke:#A3BE8C,stroke-width:2px,color:#D8DEE9;
    classDef text fill:#2E3440,stroke:#B48EAD,stroke-width:2px,color:#D8DEE9;

    A[Raw Audio Input] -->|16kHz Waveform| B(NVIDIA FastConformer)
    B:::encoder -->|512-dim Acoustic Tensor| C{MLP Modality Projector}
    
    T[Text Transcript] -->|Tokenizer| U(Input Embeddings)
    U:::text -->|1536-dim Text Tensor| D
    
    C:::projector -->|1536-dim Semantic Projection| D((Vector Concatenation))
    D --> E[Qwen2.5-1.5B Attention Layers]
    
    E:::llm <--> F([LoRA Adapters])
    F:::llm --> G[Next Token Prediction]

#### 1. Acoustic Encoder
The system utilizes NVIDIA's FastConformer as the sensory backbone. It processes 16kHz audio waveforms through an 8x downsampling convolutional sub-sampling layer, followed by Conformer blocks. This extracts dense, 512-dimensional continuous latent representations of the audio, capturing phonetics, tone, and pacing while discarding background noise.

#### 2. Modality Projector
Acoustic embeddings and text embeddings exist in vastly different mathematical latent spaces. To bridge this gap, a 2-layer Multi-Layer Perceptron (MLP) acts as a universal translator.

- Structure: Linear(512, 1536) -> GELU -> Dropout(0.1) -> Linear(1536, 1536)

- Function: The GELU non-linearity allows the network to warp the acoustic physics space to mimic the exact semantic coordinate space expected by the language model.

#### 3. Cognitive Engine
The core reasoning engine is Qwen2.5-1.5B, processed in bfloat16 precision for memory efficiency.

- Input Injection: The projected audio tensors are passed directly into the LLM via the inputs_embeds parameter, entirely bypassing the text tokenizer.

- Parameter-Efficient Adaptation: To prevent catastrophic forgetting of language capabilities, the base LLM weights remain completely frozen. Instead, Low-Rank Adaptation (LoRA) matrices are injected into the attention mechanisms (q_proj, k_proj, v_proj, o_proj). This trains the model's cross-modal attention, allowing it to "listen" to continuous audio vectors just as effectively as it reads discrete text tokens.

### Multi-stage training pipeline

#### Stage 1: Latent Space Alignment
The LLM is completely frozen. The MLP projector is trained in isolation using Mean Squared Error (MSE) loss to map the FastConformer's acoustic embeddings directly to the ground-truth text token embeddings extracted from Qwen's internal lookup tables.

#### Stage 2: End-To-End Adaption
The pre-trained projector is plugged into the LLM via the inputs_embeds layer. Low-Rank Adaptation (LoRA) modules are injected into Qwen's attention mechanisms (Query, Key, Value, Output projections). The system is jointly optimized using Cross-Entropy Loss to perform Automatic Speech Recognition (ASR), teaching the LLM's attention heads to extract meaning from continuous audio tensors.

#### Stage 3: Instruction Fine-tuning
With foundational speech recognition established, the system will be trained on complex prompt-completion pairs to function as an interactive, instruction-following voice agent.

### Functions