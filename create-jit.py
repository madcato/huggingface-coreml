import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, GPT2LMHeadModel, AutoModel


# Load "microsoft/DialoGPT-small"
# torch_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small", torchscript=True)
# torch_model = AutoModelForCausalLM.from_pretrained("facebook/blenderbot_small-90M", torchscript=True)
# torch_model = GPT2LMHeadModel.from_pretrained("distilgpt2", torchscript=True)
torch_model = AutoModel.from_pretrained("t5-small", torchscript=True)
# Set the model in evaluation mode.
torch_model.eval()

# Trace the model with random data.
# tensor of random ints
example_input = torch.randint(0, 100, (1,128), dtype=torch.int64)
print(example_input.shape)
print(example_input)
traced_model = torch.jit.trace(torch_model, example_input)
out = traced_model(example_input)
print(out[0].shape)



import coremltools as ct
import coremltools.models.datatypes as datatypes
sequence_length = 64
array_input = datatypes.Array(sequence_length)

# Define inputs and outputs for the Core ML model.
input_shape = ct.Shape(shape=(1,
                              3,
                              ct.RangeDim(lower_bound=25, upper_bound=100, default=45),
                              ct.RangeDim(lower_bound=25, upper_bound=100, default=45)))


# Using image_input in the inputs parameter:
# Convert to Core ML program using the Unified Conversion API.
model = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=example_input.shape, name="input")],
    # outputs=[ct.TensorType(name="output")],
    convert_to="mlprogram"
 )

# Save the converted model.
model.save("t5-small.mlpackage")