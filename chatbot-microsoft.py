from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

# result = [11896, 460, 12157, 5633, 11763, 26788, 5145, 5145, 5145, 5145, 5145, 5145, 5145, 5145, 5145, 5145, 5145, 5145, 5145, 5145, 5145, 5145, 5145, 5145]
result = [11896, 460, 12157, 5633, 11763, 26788, 50256]
result = [11896, 460, 12157, 5633, 11763, 26788]

result = [50256, 77, 2300, 12157, 5633, 50256]

result = [12157, 5633 ]


print(tokenizer.decode(result, skip_special_tokens=False))
# print( tokenizer.eos_token_id)

# Let's chat for 5 lines
for step in range(5):
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(input(">> User:") + tokenizer.eos_token, return_tensors='pt')

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids
    print(bot_input_ids)
    # generated a response while limiting the total chat history to 1000 tokens, 
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    print(chat_history_ids)
    # pretty print last ouput tokens from bot
    print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))