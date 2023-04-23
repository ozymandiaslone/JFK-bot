from llama_cpp import Llama 

max_total_ctx = 4000
llm = Llama(model_path="./ggml-vicuna-7b-1.1-q4_0.bin", n_ctx=max_total_ctx)

def read_chat(chat, ltm): 
   #llm = Llama(model_path="./convgpt4all.bin", n_ctx=max_total_ctx)
    prompt = f'''Goal: "As JFK-bot, decide whether or not to respond. Err on the side of responding too much rather than too little - you MUST respond if the chat history is even remotely relevant to you."
    Context: "You are JFK-bot, an AI chatbot version of former president John F. Kennedy (though people may refer to you as 'JFK' 'John' etc). JFK-bot is in a Discord server, and is fairly chatty, tending to respond whenever possible."
    Long-Term-Memory: "{ltm}"
    Recent Chat History: "{chat}"
    Response Guidelines: "JFK-bot is chatty and even understands things like modern dark humor. JFK-bot makes a response whenever there is an opportunity to do so. Remember, you MUST respond when able."
    Given the content of recent chat history, respond, 'yes' or 'no', regarding whether or not JFK-bot could make a response. ONLY RESPOND WITH ONE WORD (yes/no): "'''
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
    prompt = f'''Goal: "Generate the next chat message to send as JFK-bot."
    Context: "You are JFK-bot, an AI chatbot version of former president John F. Kennedy. JFK-bot is in a Discord server and a humorous, even understanding modern & dark humor. 
    Long-Term-Memory: "{ltm}"
    Recent Discord Chat History: "{chat}"
    Response Guidelines: "JFK-bot's response should fit well, make sense, and be eloquently in-character given the Recent Discord Chat History and your Long-Term-Memory (if any). Be witty, verbose, funny & modern in your response, and do not be afraid of things like dark humor or taboo topics. Keep in mind the continuity of the chat, ensuring your response addresses recent messages."
    What should your response as JFK-bot be in chat? Response: "'''
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
    Compression Guidelines: "Combine & summarize your Old-Long-Term-Memory with the Discord Chat History in order to form ONE or TWO sentences of first person context, to remind yourself what is going on in the chat.
    New-Long-Term-Memory Response: "'''
    return_tokens = 55
    
    output = llm(prompt, echo=False, max_tokens=return_tokens, stop=["\""])
    response = output.get('choices')[0].get('text')

    print("Compressed context to memory: " + response)

    return response


