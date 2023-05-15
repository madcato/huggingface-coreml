from transformers import T5Tokenizer, AutoModelWithLMHead, T5ForConditionalGeneration

# model_name = "t5-small"
model_name = "shahules786/Safetybot-T5-base"
# model_name = "osanseviero/t5-finetuned-test"
tokenizer = T5Tokenizer.from_pretrained(model_name)
tokens = tokenizer("Hello world how are you?", return_tensors="pt")
print(tokens)

outputTokens = [55, 55, 19, 25, 58, 0,0,0,0,0,0]
words = tokenizer.convert_ids_to_tokens(outputTokens)
print(words)



tokens = tokenizer("My name is Daniel. What's your name?", return_tensors="pt").input_ids
print(tokens)

# outputTokens = [27, 19, 4173, 5, 363, 31, 7, 39, 564, 58, 10, 10]
# words = tokenizer.convert_ids_to_tokens(outputTokens)
# print(words)


model = AutoModelWithLMHead.from_pretrained(model_name)
outputs = model.generate(tokens)
words = tokenizer.convert_ids_to_tokens(outputs[0])
print(words)

