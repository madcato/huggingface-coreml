from exporters.coreml import export
from exporters.coreml.models import BlenderbotSmallCoreMLConfig
from transformers import BlenderbotSmallTokenizer
from transformers import BlenderbotSmallForConditionalGeneration

model_ckpt = "facebook/blenderbot_small-90M"

base_model = BlenderbotSmallForConditionalGeneration.from_pretrained(
    model_ckpt, torchscript=True
)
preprocessor = BlenderbotSmallTokenizer.from_pretrained(model_ckpt)

# Encoder
coreml_config = BlenderbotSmallCoreMLConfig(
    base_model.config, 
    task="text2text-generation",
    use_past=False,
    seq2seq="encoder"
)
mlmodel = export(
    preprocessor, base_model, coreml_config
)

mlmodel.save(f"exported/{model_ckpt}_encoder.mlpackage")

# Decoder
coreml_config = BlenderbotSmallCoreMLConfig(
    base_model.config, 
    task="text2text-generation",
    use_past=False,
    seq2seq="decoder"
)
mlmodel = export(
    preprocessor, base_model, coreml_config
)

mlmodel.save(f"exported/{model_ckpt}_decoder.mlpackage")