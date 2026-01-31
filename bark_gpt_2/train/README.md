# Training Model

To train the model, we rely on Hugging Faces datasets and tokenizers so make sure to have Hugging Faces API key and authenticate.

## Hugging Faces API Key

Make sure you have the following in your `.env`

```sh
export HUGGING_FACE_TOKEN=""
```

Then, source file to ensure your token is in the process: `source .env`.

## Install Hugging Faces CLI

On MacOS, install it with brew:

```sh
brew install huggingface-cli
```

## Login to Hugging Faces

Next, login to Hugging faces:

```sh
hf auth login
```

## Train Model

Finally, train the model:

```sh
make train_bark_gpt_2
```
