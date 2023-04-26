from llama_cpp import Llama 

max_total_ctx = 3000
llm = Llama(model_path="./models/ggml-vicuna-13b-1.1-q4_1.bin", n_ctx=max_total_ctx)

def read_chat(chat, ltm): 
   #llm = Llama(model_path="./convgpt4all.bin", n_ctx=max_total_ctx)
    prompt = f'''Goal: "As JFK-bot, decide whether or not to send a reply in chat. Err on the side of over-responding."
    Context: "You are JFK-bot, an AI chatbot version of former president John F. Kennedy (though people may refer to you as 'JFK' 'John' etc). You are in a Discord server, and is fairly chatty, tending to respond whenever possible."
    Long-Term-Memory: "{ltm}"
    Recent Chat History: "{chat}"
    Could you make a response now? (yes/no): "'''
    token_gen = 10

    output = llm(prompt, echo=False, max_tokens=token_gen, stop=["\""])
    response = output.get('choices')[0].get('text')
    print("Decision response: " + response)
    if 'yes' in response.lower():
        return True
    if 'no' in response.lower():
        return False
    else:
        return None
    
def generate_response(chat, ltm): 
    #llm = Llama(model_path="./ggml-alpaca-7b-q4.bin", n_ctx=max_total_ctx)
    prompt = f'''Goal: "Read the following chat history and generate your next message."
    Bio: "You are JFK-bot, an AI chatbot based off former president John F. Kennedy, you were known for your charismatic, charming, and intelligent personality. Your wit, sense of humor, and ability to connect with people were legendary, cementing you in history as one of the greatest leaders of all time."
    Context "You are in a Discord server. Prioritize humor in your response, and don't fear modern or dark humor. Avoid repeating your previous messages."
    Long-Term-Memory: "{ltm}"
    Recent Discord Chat History: "{chat}"

    JFK-bot Discord Chat Response: "'''
    token_check = prompt.encode("utf-8")
    input_tokens = llm.tokenize(token_check)
    return_tokens = max_total_ctx - len(input_tokens) - 1
    
    output = llm(prompt, echo=False, max_tokens=return_tokens, stop=["\""])
    response = output.get('choices')[0].get('text')

    return response
def compress_to_ltm(chat, ltm):
    prompt = f'''Goal: "As JFK-bot, compress & summarize the context within the following chat history/Old-Long-Term-Memory, and update your Long-Term-Memory."
    Discord Chat History: "{chat}"
    Old-Long-Term-Memory: "{ltm}"
    Compression Guidelines: "Combine & summarize your Old-Long-Term-Memory with the Discord Chat History in order to form ONE or TWO sentences of first person context, to remind yourself what is going on in the chat. Do not add additional information, only summarize and compress existing info. "
    New-Long-Term-Memory Response: "'''
    return_tokens = 55
    
    output = llm(prompt, echo=False, max_tokens=return_tokens, stop=["\""])
    response = output.get('choices')[0].get('text')

    print("Compressed context to memory: " + response)

    return response


