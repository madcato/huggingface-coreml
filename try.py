from transformers import BlenderbotTokenizer
from transformers import BlenderbotForConditionalGeneration

def get_models():
    #  pytorch keeps an internal state of the conversation
    model_name = "facebook/blenderbot-400M-distill"
    tokenizer = BlenderbotTokenizer.from_pretrained(model_name, cache_dir="./models")
    model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

def generate_answer():
    tokenizer, model = get_models()
    # Ask user for input from command line
    user_message = input(">> ")
    inputs = tokenizer(user_message, return_tensors="pt")
    result = model.generate(**inputs)
    message_bot = tokenizer.decode(
        result[0], skip_special_tokens=True
    )  # .replace("<s>", "").replace("</s>", "")

    print(message_bot)
    

print("Talk to the bot")

while True:
    generate_answer()