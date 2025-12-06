# **Transformer Architecture — From-Scratch Implementation**

This repository contains a clean, educational, from-scratch implementation of the **Transformer** model introduced in the paper:

> **“Attention Is All You Need”**
> *Vaswani et al., 2017*

The goal is to build the system step-by-step, following the paper closely while keeping the code modular, readable, and easy to extend.

---

## 📄 **Paper Reference**

This implementation directly follows the structure and equations from the paper, especially these sections:

* **3.1** – Encoder & Decoder Stacks
* **3.2** – Attention Mechanism
* **3.3** – Position-wise Feed-Forward Networks
* **3.4** – Embeddings + Softmax
* **3.5** – Positional Encoding
* **5.4** – Regularization & Dropout Strategy

---

## 🧱 **Implementation Strategy**

The architecture is implemented incrementally, building confidence and correctness at each stage.

### **1. Foundational Components**

* Token Embedding
* Positional Encoding
* Single-Head Self-Attention

### **2. Attention Mechanism (Core of the Transformer)**

* Multi-Head Attention
* Scaled Dot-Product Attention
* Cross-Attention (Decoder → Encoder outputs)
* Attention Masks

  * Padding mask
  * Look-Ahead (causal) mask

### **3. Larger Building Blocks**

#### **Encoder Layer**

* Self-Attention
* Feed-Forward Network (FFN)
* Residual Connections + LayerNorm

#### **Decoder Layer**

* Masked Self-Attention
* Cross-Attention
* Feed-Forward Network

### **4. Final Assembly**

* Full Encoder Stack
* Full Decoder Stack
* Output linear layer + softmax
* Training-ready Transformer model

---

## 🔍 **Visualization & Planning Tips**

Before writing each module:

* Sketch tensor shapes for every operation
* Draw attention matrices
* Map code directly to relevant equations in the paper
* Maintain a running log of expected vs actual tensor dimensions

**Most bugs in Transformer implementations come from shape mismatches—visualization prevents these.**

---

## 📐 **Dimension Cheat Sheet**

| Component          | Shape                                       |
| ------------------ | ------------------------------------------- |
| Input tokens       | `[batch_size, seq_len]`                     |
| Embeddings         | `[batch_size, seq_len, d_model]`            |
| Q / K / V matrices | `[batch_size, num_heads, seq_len, d_k]`     |
| Attention scores   | `[batch_size, num_heads, seq_len, seq_len]` |
| FFN hidden layer   | `[batch_size, seq_len, d_ff]`               |
| Output logits      | `[batch_size, seq_len, vocab_size]`         |

---

## ⚠️ **Common Pitfalls (and How to Avoid Them)**

* ❗ *Forgetting the scale factor* → Always divide attention scores by **√d_k**
* ❗ *Wrong mask shape* → Masks must broadcast across **batch, seq_len, and heads**
* ❗ *Padding mask applied incorrectly* → Must be applied to **attention scores**
* ❗ *Missing residual connections*
* ❗ *Incorrect ordering*:
  **LayerNorm → Attention/FFN → Dropout → Residual**
* ❗ *Mismatched dimensions in multi-head attention*
* ❗ *Incorrect handling of padded tokens / variable-length sequences*

---

## 🚀 **Performance Considerations**

* Attention complexity is **O(n²)** w.r.t. sequence length
* Memory usage also scales with **seq_len²**
* Choose `d_model` and `num_heads` carefully for your GPU
* Larger models require smaller batch sizes
* Use **mixed precision training (fp16/bf16)** when possible

---

## 🧪 **Testing Strategy**

Each component is verified independently before being integrated:

* Confirm tensor shape preservation
* Verify attention weights sum to 1
* Ensure masks correctly block future tokens
* Check gradients propagate through all layers
* Watch for NaNs in softmax / normalization steps
* Validate outputs using small, deterministic examples

