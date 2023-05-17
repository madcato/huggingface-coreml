# Huggingface to CoreML

This repository contains a script and information to convert a Huggingface models to CoreML.

## Info
- Model to transform: "gorkemgoknar/gpt2chatbotenglish"[Example of usage](https://huggingface.co/gorkemgoknar/gpt2chatbotenglish?text=Hello+there) Try it by executing in command line `$  python3 create-chatgpt-model.py`

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
-[BenderbotTokenizer doc](https://huggingface.co/docs/transformers/model_doc/blenderbot#transformers.BlenderbotTokenizer)
- [BenderbotTokenizer implementation](https://github.com/huggingface/transformers/blob/3335724376319a0c453049d0cd883504f530ff52/src/transformers/models/blenderbot/tokenization_blenderbot.py#L4)
- https://huggingface.co/microsoft/DialoGPT-small?text=Hi.
- [Converting from PyTorch](https://coremltools.readme.io/docs/pytorch-conversion)

## Initial steps

0. Clone repo: `$ git clone https://github.com/madcato/huggingface-coreml.git`
1. Execute chatbot to try chatbot (optional), `$ python3 try.py`
2. Install huggingface exporters submodule, 
    `$ git submodule init && git submodule update`
3. `$ cd exporters && pip3 install -e . && cd ..`
4. `python3 export`

The last step will create two `mlpackage` files in the `exported` folder: one for decoder and another for decoder part of the model. These files can be opened with Xcode and the model can be tested in the playground or project.

### Try

- `$ python3 -m exporters.coreml --model=t5-small --feature=text2text-generation exported`
- `$ python3 -m exporters.coreml --model=facebook/blenderbot-400M-distill --feature=text2text-generation exported`
- `$ python3 -m exporters.coreml --model=distilgpt2 --feature=text2text-generation exported`

## Exporters features

- 'feature-extraction', 
- 'feature-extraction-with-past', 
- 'fill-mask', 
- 'image-classification', 
- 'masked-im', 
- 'multiple-choice', 
- 'next-sentence-prediction', 
- 'object-detection', 
- 'question-answering', 
- 'semantic-segmentation', 
- 'text-classification', 
- 'text-generation', 
- 'text-generation-with-past', 
- 'text2text-generation', 
- 'token-classification', 
- 'sequence-classification', 
- 'causal-lm', 
- 'causal-lm-with-past', 
- 'seq2seq-lm', 
- 'seq2seq-lm-with-past', 
- 'speech2seq-lm', 
- 'speech2seq-lm-with-past', 
- 'masked-lm', 
- 'vision2seq-lm', 
- 'default', 
- 'default-with-past', 
- 'automatic-speech-recognition', 
- 'ctc'
