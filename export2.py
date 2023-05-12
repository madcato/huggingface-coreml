from exporters.coreml import export
from exporters.coreml.models import BartCoreMLConfig
from transformers import BartTokenizer
from transformers import BartForConditionalGeneration

model_ckpt = "facebook/bart-large"

base_model = BartForConditionalGeneration.from_pretrained(
    model_ckpt, torchscript=True
)
preprocessor = BartTokenizer.from_pretrained(model_ckpt)

# Encoder
coreml_config = BartCoreMLConfig(
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
coreml_config = BartCoreMLConfig(
    base_model.config, 
    task="text2text-generation",
    use_past=False,
    seq2seq="decoder"
)
mlmodel = export(
    preprocessor, base_model, coreml_config
)

mlmodel.save(f"exported/{model_ckpt}_decoder.mlpackage")