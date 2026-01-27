# BarkGPT - Tiny GPT implementation

<div align="center">
  <img src="./og.png" alt="BarkGPT - The Smallest LLM" width="100%" height="500"/>
</div>

This is a hands-on tutorial about how to use Large Language Models and in particular, train one from scratch. Our goal here, is to train an LLM from scratch, with very low vocabulary, only dog barking.

While there might not be many applications for such a model (unless you want to build a robot dog), I always felt intimidated by LLMs and the fact that they are trained on vast amounts of datasets. Having tiny dataset, with small vocabulary, removes one of the complexities in the learning curves. Unsurprisingly understanding an LLM, even that small is still quite hard.

This project is heavily inspired on [CatGPT](https://www.cat-gpt.com/), an AI that meows. I am a cat owner myself, although the market for meow AIs feels quite saturated with clear market dominance. I haven't seen any worthy dog GPTs, only mere amateur projects, so I think this project will just fit right in.

## Demo - AI LLM Project that barks trained from scratch. 🚀

Try demo at [BarkGPT](https://www.bark-slm.com/)

## Overview 🧠

To get our BarkLLM from scratch, we need to do the following steps:

This repository is split into series of self-contained guides. Each one builds on the previous, but you can jump around if you want.

### 🧱 [Building BarkGPT from scratch](./docs/01-building-from-scratch/README.md)

This is the foundation where we define a tiny vocabulary, build a GPT‑like transformer by hand, train it on synthetic data and generate tokens using pure PyTorch.

You’ll see very clearly how next‑token prediction works, why prompt contracts exist and how hallucinations naturally emerge.

If you only read one document, read this one.

### 📚 [Training BarkGPT on WebText‑2 dataset](./docs/02-training-on-webtext/README.md)

Once the toy model makes sense, we move to something closer to reality. Here we introduce real text datasets and add a real language tokenizer

### 🌐 [Integrating into Web App – from BarkGPT to BarkAgent](./docs/03-barkgpt-to-barkagent/README.md)

In this part, we stop thinking like ML engineers and start thinking like system builders. Here our model is ready to be used in chat interfaces, we wrap it in [langchain](https://www.langchain.com/), serve it with [FastAPI](https://fastapi.tiangolo.com/) and deep dive into why not using LLM `/generate` directly, and why is building extra chat assistant adapter layer.

This is essentially a miniature version of how real LLM systems are shipped.

### ☁️ [Deploying Small Language Models on AWS](./docs/04-deploying-small-language-models/README.md)

In this part, we will examine how to get the ready language model and deploy it on AWS. For demonstration purposes, an actual small language model was used: [Qwen](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF). We will see how to work with `Gguf` files and package them reliably to serve the inferrence. As an extra mile, we will try to apply the same engineering approach to deploy our improved BarkGPT using improved compression with Gguf.
