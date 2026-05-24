# mytorch

A collection of PyTorch demonstrations and experiments.

## Expert AI/PyTorch Learning Plan (12-Week Intensive)

This accelerated plan is designed to take you from foundational knowledge to expert-level proficiency in AI and PyTorch with a strong focus on transformers and practical applications in just 3 months (12 weeks).

### Prerequisites
Before starting, ensure you have:
- Proficiency in Python (functions, classes, OOP, numpy/pandas basics)
- Mathematics: Linear algebra (vectors, matrices), Calculus (derivatives, gradients), Probability (basic distributions)
- Basic understanding of machine learning concepts (training/test split, overfitting)

Note: We'll briefly review PyTorch basics, but you should be willing to learn quickly.

---

## **12-Week Intensive Plan**

### **Weeks 1-2: PyTorch Fundamentals & Deep Learning Basics (Accelerated)**
**Goals:** Rapidly cover essential PyTorch concepts and basic deep learning workflow.
**Topics:**
- Tensor creation, operations, broadcasting, GPU usage
- Automatic differentiation (`requires_grad`, `backward()`)
- `torch.nn.Module`, loss functions, optimizers
- `Dataset` and `DataLoader` basics
- Simple training/validation loops
**Mini Project:** Implement and train a feedforward network on MNIST.
**Outcome:** Achieve >97% accuracy on MNIST with a simple neural network.
**Time Allocation:** 20% (condensed from original 4 weeks)

### **Weeks 3-5: Transformer Architecture & Deep Dive (Extended Focus)**
**Goals:** Thoroughly master transformer architectures, attention mechanisms, and modern variants.
**Topics:**
- Scaled dot-product attention, multi-head attention from scratch
- Positional embeddings (sinusoidal, learned, RoPE)
- Transformer encoder (BERT-style) and decoder (GPT-style) implementations
- Masking: padding masks, causal/look-ahead masks
- Layer normalization, residual connections
- Vision Transformers (ViT) basics
- Hugging Face Transformers library deep dive
- Tokenization strategies (BPE, WordPiece, SentencePiece)
**Mini Projects:**
  - Week 3: Build a transformer encoder from scratch for sequence classification
  - Week 4: Implement a decoder-only transformer (GPT-style) for text generation
  - Week 5: Fine-tune a pretrained BERT model on a custom dataset using Hugging Face
**Outcome:** 
  - Ability to implement transformer variants from scratch
  - Proficiency with Hugging Face ecosystem for practical applications
  - Understanding of architectural trade-offs between encoder/decoder designs
**Time Allocation:** 40% (significantly increased from original 2 weeks)

### **Weeks 6-7: Transformer Applications & Advanced NLP**
**Goals:** Apply transformers to real-world NLP tasks and understand advanced techniques.
**Topics:**
- Text classification (sentiment, topic, intent detection)
- Named Entity Recognition (NER) with transformer-based models
- Question Answering (extractive QA with models like BERT)
- Text summarization (abstractive and extractive)
- Translation with sequence-to-sequence transformers
- Handling long documents (Longformer, BigBird, sliding window attention)
- Prompt engineering basics and few-shot learning
**Mini Projects:**
  - Week 6: Build a pipeline for sentiment analysis and NER using Hugging Face models
  - Week 7: Implement a question answering system on SQuAD-style data
**Outcome:** Deployable NLP applications with >90% F1 on standard benchmarks.
**Time Allocation:** 15%

### **Weeks 8-9: Transformer Optimization, Deployment & Production**
**Goals:** Prepare transformer models for real-world deployment and optimization.
**Topics:**
- Model distillation (teacher-student networks)
- Quantization (INT8, FP16) for transformers
- Pruning and sparsity techniques
- Efficient inference: caching, batching, sequence length optimization
- ONNX export and TorchScript for transformers
- Serving frameworks: TensorRT, Triton Inference Server, TorchServe basics
- Monitoring, logging, and A/B testing for ML services
- Hardware considerations (GPU vs CPU inference trade-offs)
**Mini Projects:**
  - Week 8: Optimize a BERT model (quantize, prune, distill) and benchmark inference speed
  - Week 9: Deploy a transformer model as a REST API with Docker
**Outcome:** 3x+ faster inference with <3% accuracy drop; deployable API service.
**Time Allocation:** 15% (new focus area)

### **Week 10: Vision Transformers & Multimodal Applications**
**Goals:** Apply transformers to computer vision and multimodal tasks.
**Topics:**
- Vision Transformers (ViT): patch embeddings, classification head
- DEtection TRansformer (DETR) for object detection
- Segmenter and MaskFormer for segmentation
- Vision-Language models (CLIP, BLIP basics)
- Video transformers (TimeSformer, Video Swin)
**Mini Project:** Implement or fine-tune a Vision Transformer for image classification and compare with CNN baseline.
**Outcome:** Understand when to use ViTs vs CNNs; able to work with multimodal models.
**Time Allocation:** 5%

### **Week 11: Efficient Transformers & Latest Research**
**Goals:** Stay current with efficient transformer architectures and recent advances.
**Topics:**
- Sparse attention patterns (Longformer, BigBird, Routing Transformers)
- Linear complexity transformers (Performer, Linformer)
- Memory-efficient training (gradient checkpointing, ZeRO)
- Optimization techniques (AdamW, learning rate schedules with warmup)
- Recent architectures (LLaMA, Mistral, Phi variants - concepts only)
**Mini Project:** Implement one efficient attention mechanism and compare complexity/scaling.
**Outcome:** Knowledge of state-of-the-art efficient transformer techniques.
**Time Allocation:** 5%

### **Week 12: Capstone Project & Portfolio**
**Goals:** Synthesize learning into a comprehensive project demonstrating expertise.
**Project Requirements:** Build an end-to-end application that solves a real-world problem using transformers, including:
  - Problem definition and dataset collection/curation
  - Model selection, fine-tuning, or custom architecture
  - Optimization for deployment (quantization, distillation, etc.)
  - Deployment as a web service or application
  - Evaluation with appropriate metrics
  - Documentation and code quality
**Examples:** 
  - Medical report generator from clinical notes
  - Multilingual customer support chatbot
  - Legal document summarization system
  - Code generation assistant for specific domains
**Outcome:** Portfolio-worthy project with live demo, deployed model, and detailed documentation.
**Time Allocation:** 5%

---

## **How to Use This Plan with This Repository**

This `mytorch` repository already contains relevant demonstrations you can build upon:

- **Transformer Foundations:** 
  - `src/inference/prefix.py` - Shows prefix sharing concepts relevant to transformer KV-caching
  - `src/inference/kv-cache.py` - Direct implementation of key-value caching for efficient transformer inference
  
- **Parallel Training (relevant for large transformers):**
  - `src/parallel/` - Contains examples of Data Parallelism, Tensor Parallelism, FSDP, and Pipeline Parallelism
  
- **Evaluation & Perplexity:**
  - `src/evaluation/perplexity-demo.py` - Shows how to evaluate language models
  
- **Tokenization & SFT:**
  - `src/tokenization/token-demo.py` - Tokenization visualization and experimentation
  - `src/sft/sft.py` - Supervised Fine-Tuning examples

**Suggested Approach:** As you complete each week's mini project or capstone, create a clean, well-documented version in this repository:
- Create new directories like `projects/week03_transformer_encoder/` or `projects/capstone_medical_report_generator/`
- Or enhance existing demonstrations in relevant `src/` directories with your improved implementations

---

## **Tracking Your Progress in this Accelerated Plan**

1. **Bi-weekly Check-Ins (Every 2 weeks):**
   - Can I implement the core concepts from scratch when needed?
   - Have I completed the mini projects successfully?
   - What transformer concepts still feel unclear?

2. **Maintain a Focused Learning Journal:**
   - Key transformer insights each week
   - Code snippets for attention mechanisms you reuse
   - Resources that clarified difficult concepts
   - Deployment and optimization lessons learned

3. **Build Your Transformer Portfolio:**
   - Week 3-5: Scratch implementations (encoder, decoder)
   - Week 6-7: Hugging Face fine-tuning projects
   - Week 8-9: Optimization and deployment examples
   - Week 10-11: Vision transformers and efficient architectures
   - Week 12: Capstone project

4. **Leverage Community Resources:**
   - Hugging Face forums and model hub
   - Papers with Code transformer implementations
   - PyTorch discussion boards
   - GitHub trending transformer repositories

---

**Important Notes for this Accelerated Plan:**

- **Intensity:** Expect 15-20 hours per week of focused learning and coding
- **Practice Focus:** Code every concept immediately; don't just watch tutorials
- **Transformer Emphasis:** ~60% of your time will be directly on transformer-related topics
- **Practical Outcome:** Each week produces a tangible, runnable project
- **Portfolio Focus:** By week 12, you'll have 5-6 transformer projects plus a capstone

**Remember:** Expertise comes from building, breaking, fixing, and rebuilding. In this plan, you'll code transformer components from scratch, optimize them for production, and deploy real applications—all within 12 weeks.

**Start your transformer expertise journey now — Week 1 begins today!**