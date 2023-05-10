# Huggingface to CoreML

This repository contains a script and information to convert a Huggingface models to CoreML.

## Info
- Model to transform: "facebook/blenderbot-400M-distill"[Example of usage](https://huggingface.co/spaces/LamaAl/chatbot/blob/main/app.py) Try it by executing in command line `$ python3 try.py`

### Links
- [HuggingFace](https://huggingface.co/)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [HuggingFace Core ML Models](https://huggingface.co/coreml#models)
- [Using Stable Diffusion with Core ML on Apple Silicon](https://huggingface.co/blog/diffusers-coreml)
- [Export Hugging Face models to Core ML and TensorFlow Lite](https://github.com/huggingface/exporters)
- [Swift Core ML implementations of Transformers: GPT-2, DistilGPT-2, BERT, DistilBERT, more coming soon!](https://github.com/huggingface/swift-coreml-transformers)
- [Figuring out the shape of a Transformer Model To translate it to a coreML model](https://developer.apple.com/forums/thread/682408)
- [Core ML Stable Diffusion](https://github.com/apple/ml-stable-diffusion)
- [Deploying Transformers on the Apple Neural Engine](https://machinelearning.apple.com/research/neural-engine-transformers)
- [Swift app demonstrating Core ML Stable Diffusion](https://github.com/huggingface/swift-coreml-diffusers)
- [Apple Core ML](https://developer.apple.com/documentation/coreml)

## Initial steps

1. Execute chatbot to download model, `$ python3 try.py`
2. Install huggingface exporters, `$ git clone https://github.com/huggingface/exporters.git`
3. `$ cd exporters && pip3 install -e .`
4. `$ cd ..`
5. `python3 -m exporters.coreml --model=facebook/blenderbot-400M-distill exported/ --feature=text2text-generation`