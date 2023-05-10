from exporters.coreml import export
from exporters.coreml.models import BlenderbotCoreMLConfig
from transformers import BlenderbotTokenizer
from transformers import BlenderbotForConditionalGeneration

model_ckpt = "facebook/blenderbot-400M-distill"
base_model = BlenderbotForConditionalGeneration.from_pretrained(
    model_ckpt, torchscript=True
)
preprocessor = BlenderbotTokenizer.from_pretrained(model_ckpt)

coreml_config = BlenderbotCoreMLConfig(
    base_model.config, 
    task="text2text-generation",
    use_past=False,
    seq2seq="encoder"
)
mlmodel = export(
    preprocessor, base_model, coreml_config
)

mlmodel.save(f"exported/{model_ckpt}.mlpackage")