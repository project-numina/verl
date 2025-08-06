<h1 align="center">Kimina Prover RL</h1>

<p align="center">
<b>A slimmed-down training pipeline from Kimina Prover, with core features and full compatibility with <a href="https://github.com/takikawa/verl">Verl</a>. ⚡️</b>
</p>

<p align="center">
    <a href="https://projectnumina.ai/"><img alt="Project Numina" src="images/logo_projectNumina_light.png" style="height:20px; width:auto; vertical-align:middle; border-radius:4px;"></a>
    <a href="https://huggingface.co/AI-MO"><img alt="HF AI-MO" src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo-with-title.svg" style="height:20px;vertical-align:middle; border-radius:4px;"></a>
</p>

---

**Kimina-Prover-RL** is an open-source training pipeline for formal theorem proving in **Lean 4**, based on a structured *reasoning-then-generation* paradigm inspired by [DeepSeek-R1](https://arxiv.org/abs/2405.14552).

This pipeline is a simplified version of the system used to train **[Kimina Prover](https://arxiv.org/abs/2504.11354)**, retaining its core components and offering full compatibility with the open-source **Verl** framework.

As a result of this training, we are releasing **[`AI-MO/Kimina-Prover-RL-1.7B`](https://huggingface.co/AI-MO/Kimina-Prover-RL-1.7B)** — a 1.7B parameter model that achieves **76.63% Pass@32** on **MiniF2F**, setting a new **state of the art** for open-source models at this scale.

![alt text](images/best8_performances.png)

👉 **[Read the full blog post →](INSERT LINK TO BLOG)**

## 🚀 Installation

First, follow the installation instructions for **Verl** in the [main README](../README.md).

To verify Lean 4 proofs efficiently, we use the [**kimina-lean-server**](https://github.com/project-numina/kimina-lean-server), which supports high-throughput parallel checking.

We recommend using the Docker image we provide:

```bash
docker run -d \
  --name server \
  --restart unless-stopped \
  --env-file .env \
  -p 80:8000 \
  projectnumina/kimina-lean-server:2.0.0
```

Then install our Python client, kimina-client, from PyPI:

```
pip install kimina-client
```

## 📦 Running the Recipe
```
cd recipe/kimina_prover_rl

export LEAN_SERVER_API_URL="http://localhost:8000"
```
If your server is configured with an API key you also need to export it:

```
export LEAN_SERVER_API_KEY="your-api-key"
```

To start training simply launch:

```
sh kimina_prover_1.7B.sh
```

This script will download the dataset and launch the training. Your can edit it to adapt it to your hardware and environment.

Alternatively, you can also launch a smaller version of the training for debugging purposes:

```
sh kimina_prover_1.7B_dry_run.sh
```

## 📊 Expected Results

During training, you should see on wandb the mean response length growing and the number of formatting errors decreasing:

![alt text](images/response_length.png)

![alt text](images/formatting_errors.png)

You should also see the best@8 mean performances growing on MiniF2F before and after error correction:

![alt text](image.png)

After training, the model achieves 76.63% Pass@32 on MiniF2F

## 🧠 Resources

- [Blog Post (training details)](INSERT LINK TO BLOG)
- [Model on Hugging Face](https://huggingface.co/collections/AI-MO/kimina-prover-686b72614760ed23038056c5)
- [Dataset on Hugging Face](https://huggingface.co/datasets/AI-MO/NuminaMath-LEAN)
- [Kimina Lean Server](https://github.com/project-numina/kimina-lean-server)
- [Kimina Prover Preview paper](https://arxiv.org/abs/2504.11354)
- [Kimina Prover blog post](https://huggingface.co/blog/AI-MO/kimina-prover)