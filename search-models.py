from exporters.coreml import export, validate_model_outputs, CoreMLConfig
from exporters.coreml.models import *
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM

feature = "text2text-generation"

models = [
            # ("npc-engine/exported-bart-light-gail-chatbot", BartCoreMLConfig),  # not working
            # ("facebook/blenderbot_small-90M",BlenderbotSmallCoreMLConfig),  # Error computing NN outputs.
            # ("facebook/blenderbot-400M-distill",BlenderbotCoreMLConfig),  # not working
            # ("facebook/blenderbot-3B",BlenderbotCoreMLConfig),  # Too big
            # ("facebook/blenderbot-90M",BlenderbotCoreMLConfig),  # Error computing NN outputs.
            # ("facebook/blenderbot-1B-distill",BlenderbotCoreMLConfig),  # There appear to be 1 leaked semaphore objects to clean up at shutdown
            # ("hyunwoongko/blenderbot-9B",BlenderbotCoreMLConfig),  # Too big
            #("thu-coai/blenderbot-400M-esconv",BlenderbotCoreMLConfig),  #  # There appear to be 1 leaked semaphore objects to clean up at shutdown
            # ("thu-coai/blenderbot-1B-augesc",BlenderbotCoreMLConfig),  # Too big
            # ("Grendar/blenderbot-400M-distill-Shiro",BlenderbotCoreMLConfig),   # There appear to be 1 leaked semaphore objects to clean up at shutdown
    # ("shahules786/Safetybot-T5-base", T5CoreMLConfig),
    # ("osanseviero/t5-finetuned-test", T5CoreMLConfig),
            # ("BlackSamorez/rudialogpt3_medium_based_on_gpt2_2ch", GPT2CoreMLConfig),  # Error coniguracion
            # ("gorkemgoknar/gpt2chatbotenglish", GPT2CoreMLConfig),  # Error coniguracion
            # ("Alethea/GPT2-chitchat", GPT2CoreMLConfig),   # Error coniguracion
            # ("thu-coai/CDial-GPT2_LCCC-base", GPT2CoreMLConfig),  # Error modelo
            # ("robinhad/gpt2-uk-conversational", GPT2CoreMLConfig),    # Error coniguracion
            # ("AriakimTaiyo/gpt2-chat", GPT2CoreMLConfig),  # Error coniguracion
            # ("Vaibhav-rm/GPT2-Shri-v1", GPT2CoreMLConfig),    # Error coniguracion
            # ("LrxLcs/GPT2-V2", GPT2CoreMLConfig),    # Error coniguracion
            # ("LrxLcs/GPT2-Test", GPT2CoreMLConfig),  # Error coniguracion
            # ("huolongguo10/CDial-GPT2-LCCC-Base-copy", GPT2CoreMLConfig),   # Error coniguracion
            # ("h2oai/h2ogpt-oasst1-512-12b", GPT2CoreMLConfig),  # Error coniguracion
            # ("distilgpt2", GPT2CoreMLConfig),         # Error coniguracion
            # ("microsoft/DialoGPT-small", GPT2CoreMLConfig)  # Error coniguracion
            ("gorkemgoknar/gpt2chatbotenglish", GPT2CoreMLConfig)
    ]

for model_ckpt, config_class in models:
    print("--------------------------------")
    print(" EXPORTING: ", model_ckpt)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_ckpt, torchscript=True
    )
    preprocessor = AutoTokenizer.from_pretrained(model_ckpt)


    print(" ENCODER --------------------------------")
    coreml_config = config_class(
        base_model.config, 
        task="text2text-generation",
        use_past=False,
        seq2seq="encoder"
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
    mlmodel.save(f"exported/{model_ckpt}_encoder.mlpackage")

    print(" DECODER --------------------------------")
    coreml_config = config_class(
        base_model.config, 
        task="text2text-generation",
        use_past=False,
        seq2seq="decoder"
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
    mlmodel.save(f"exported/{model_ckpt}_decoder.mlpackage")
    print("END --------------------------------")














# # Encoder
# coreml_config = BlenderbotSmallCoreMLConfig(
#     base_model.config, 
#     task="text2text-generation",
#     use_past=False,
#     seq2seq="encoder"
# )
# mlmodel = export(
#     preprocessor, base_model, coreml_config
# )

# mlmodel.save(f"exported/{model_ckpt}_encoder.mlpackage")

# # Decoder
# coreml_config = BlenderbotSmallCoreMLConfig(
#     base_model.config, 
#     task="text2text-generation",
#     use_past=False,
#     seq2seq="decoder"
# )
# mlmodel = export(
#     preprocessor, base_model, coreml_config
# )

# mlmodel.save(f"exported/{model_ckpt}_decoder.mlpackage")