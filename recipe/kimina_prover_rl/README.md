<h1 align="center">Kimina Prover RL</h1>

<p align="center">
<b>A slimmed-down training pipeline from Kimina Prover, with core features and full compatibility with verl. ⚡️</b>

</p>

<p align="center">
    <a href="https://projectnumina.ai/"><img alt="Project Numina" src="images/logo_projectNumina_light.png" style="height:20px; width:auto; vertical-align:middle; border-radius:4px;"></a>
    <a href="https://huggingface.co/AI-MO"><img alt="HF AI-MO" src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo-with-title.svg" style="max-width:20%;vertical-align:middle; border-radius:4px;"></a>
</p>

Kimina-prover-rl is an open-source training pipeline for formal theorem proving in Lean 4, based on a structured reasoning-then-generation paradigm inspired by DeepSeek-R1.

This training pipeline is a simplified version of the system we used to train Kimina Prover, preserving the key components of the system and offering full compatibility with the open-source verl framework.

As a result of this training pipeline, we are releasing **AI-MO/Kimina-prover-RL-1.7B**, a 1.7B-parameter model that achieves **76.63% Pass@32** on the MiniF2F benchmark — setting a new state of the art for open-source models in this size category.


INSERT LINK TO THE BLOG

## Installation

To run this recipe, you need first to follow the verl installation instruction in the README at the root of this repository.

Our reward function performs API call to a [kimina-lean-server](https://github.com/project-numina/kimina-lean-server) to verify proofs efficiently. You need to start a kimina-lean-server enable lean proofs verifications.

We recommand to use the docker image that we provide.

```
docker run -d \
  --name server \
  --restart unless-stopped \
  --env-file .env \
  -p 80:8000 \
  projectnumina/kimina-lean-server:2.0.0
```


You then need to install our client, `kiminia-client`, to interact with the `kimina-lean-server`.

```
pip install kiminia-client
```

## Recipe

## Launch the recipe

```
cd recipe/kimina_prover_rl

export LEAN_SERVER_API_URL="http://localhost:8000"
```

If you've added an api key to your server config, you also need to export it:

```
export LEAN_SERVER_API_KEY="your-api-key"
```

To run download and preprocess the data and then launch the training, you can then simply use

```
sh kimina_prover_1.7B.sh
```


## Expected results
