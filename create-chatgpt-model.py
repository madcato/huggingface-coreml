from exporters.coreml import export, validate_model_outputs, CoreMLConfig
from exporters.coreml.models import *
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM


class DialogGPT2CoreMLConfig(GPT2CoreMLConfig):
    @property
    def inputs(self) -> OrderedDict[str, InputDescription]:
        input_descs = super().inputs
        input_descs["input_ids"].sequence_length = 64
        return input_descs

feature = "text2text-generation"
# feature = "text-generation"

models = [
        ("gorkemgoknar/gpt2chatbotenglish", DialogGPT2CoreMLConfig),
        ("Alethea/GPT2-chitchat", DialogGPT2CoreMLConfig),
        ("microsoft/DialoGPT-small", DialogGPT2CoreMLConfig),
        ("microsoft/DialoGPT-medium", DialogGPT2CoreMLConfig)
    ]

for model_ckpt, config_class in models:
    print("--------------------------------")
    print(" EXPORTING: ", model_ckpt)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_ckpt, torchscript=True
    )
    preprocessor = AutoTokenizer.from_pretrained(model_ckpt)

    coreml_config = config_class(
        base_model.config, 
        task=feature,
        use_past=False
    )
    mlmodel = export(
        preprocessor, base_model, coreml_config
    )
    validate_model_outputs(
                    coreml_config,
                    preprocessor,
                    base_model,
                    mlmodel,
                    coreml_config.atol_for_validation,
                )
    mlmodel.save(f"exported/{model_ckpt}.mlpackage")

    print("END --------------------------------")
