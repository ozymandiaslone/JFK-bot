from llama_cpp import Llama 

max_total_ctx = 4000
llm = Llama(model_path="./ggml-vicuna-7b-1.1-q4_0.bin", n_ctx=max_total_ctx)

def read_chat(chat): 
   #llm = Llama(model_path="./convgpt4all.bin", n_ctx=max_total_ctx)
    prompt = f'''Goal: "Decide whether or not to respond as JFK-bot"
    Context: "You are JFK-bot, an AI chatbot version of former president John F. Kennedy. JFK-bot is in a Discord server, and is fairly chatty, tending to respond whenever possible."
    Recent Chat History: "{chat}"
    Response Guidelines: "JFK-bot is chatty and even understands things like modern dark humor. JFK-bot makes a response whenever there is an opportunity to do so."
    Given the content of recent chat history, ### ANSWER WITH ONLY ONE WORD ###, 'yes' or 'no', regarding whether or not JFK-bot could potentially make a response (yes/no): "'''
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
    
def generate_response(chat): 
    #llm = Llama(model_path="./ggml-alpaca-7b-q4.bin", n_ctx=max_total_ctx)
    prompt = f'''Goal: "Generate the next chat message to send, as JFK-bot."
    Context: "JFK-bot is an AI chatbot version of former president John F. Kennedy. JFK-bot is in a Discord server, and is friendly and funny, even understanding modern dark humor. Format your response as the plain text of a Discord message. NO need for things like backticks, quotes, newline characters, etc.
    Recent Discord Chat History: "{chat}"
    Response Guidelines: JFK-bot's response should fit well into the recent chat history & flow directly with the conversation. JFK-bot is witty & humorous, though also incredibly thoughtful & verbose.
    What should JFK-bot's next message in chat be? Response: "'''
    token_check = prompt.encode("utf-8")
    input_tokens = llm.tokenize(token_check)
    return_tokens = max_total_ctx - len(input_tokens) - 1
    
    output = llm(prompt, echo=False, max_tokens=return_tokens, stop=["\""])
    response = output.get('choices')[0].get('text')

    return response
