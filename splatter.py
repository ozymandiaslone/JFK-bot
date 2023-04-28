from llama_cpp import Llama 

max_total_ctx = 1900
llm = Llama(model_path="./models/ggml-vic13b-q4_2.bin", n_ctx=max_total_ctx)

def read_chat(chat, ltm): 
   #llm = Llama(model_path="./convgpt4all.bin", n_ctx=max_total_ctx)
    prompt = f'''Goal: "As JFK-bot, decide whether or not to send a reply in chat. Err on the side of over-responding."
    Context: "You are an AI chatbot version of former president John F. Kennedy in a Discord server. You tend to respond whenever possible. Respond if you are part of the converstaion."
    Long-Term-Memory: "{ltm}"
    Recent Chat History: "{chat}"
    Could you make a response? (yes/no): "'''
    token_gen = 6

    output = llm(prompt, echo=False, max_tokens=token_gen, stop=["\""], temperature=0.7)
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
    prompt = f'''Goal: "Generate a unique Discord chat reply."
    Bio: "You are JFK-bot, a funnier, darker, AI chatbot version of president John F. Kennedy. You are witty, verbose, and intelligent, though as a chatbot, you prioritize modern edgy humor."
    Context "You are in a Discord server. Directly address recent messages, and do NOT repeat any message content. Your response should be informal, conversational, and not too long. Do NOT repeat content from your previous messages. Respond to any and all messages aimed at you, which you haven't yet addressed."
    Long-Term-Memory: "{ltm}"
    Recent Discord Chat History: "{chat}"

    JFK-bot Response: "'''
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
    New-Long-Term-Memory: "'''
    return_tokens = 55
    
    output = llm(prompt, echo=False, max_tokens=return_tokens, stop=["\""])
    response = output.get('choices')[0].get('text')

    print("Compressed context to memory: " + response)

    return response


