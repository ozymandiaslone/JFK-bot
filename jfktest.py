from splatter import * 
chat = []
while True:
    msg = input("Chat: ")
    chat.append({'ozymandias':msg})
    if read_chat(msg):
        resp = generate_response(chat)
        chat.append({'JFK-bot':resp})
        print(resp)


