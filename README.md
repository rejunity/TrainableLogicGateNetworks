# Trainable Logic Gate Networks

## Prerequisites

This repo uses the *uv* package manager, see the [uv repo](https://github.com/astral-sh/uv/) for installation instructions or simply install *uv* via pip:

```bash
pip install uv
```

## How to run

The script will run as is with default settings, no config needed.
```bash
uv run mnist.py
```
It should produce a [1300,1300,1300] logical gate network with an accuracy of 90% on the MNIST testset.

## Config

To manage configuration settings, use the `.env` file. 
A sample configuration is available in `.env.example`. 
To set up, copy `.env.example` to `.env` and update it as needed.
For security reasons, the `.env` file is excluded from version control 
to prevent accidental exposure of sensitive data like API keys.
