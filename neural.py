import discord
from discord.ext import tasks
from splatter import *
import time
import asyncio
from datetime import datetime
import constants

# Discord API token
TOKEN = constants.DISCORD_API_KEY

# new_msg bool 
new_msg = False
lm = None
ltm = "None"
changes = False
considering = False

# INTENT
intents = discord.Intents.default()
intents.message_content = True

# Initialize the Discord client
client = discord.Client(intents=intents)

# Global variables for storing chat history
messages = []


@client.event
async def on_ready():
    print(f'Logged in as {client.user}')
    thought_tick.start()

@client.event
async def on_message(message):
    global new_msg
    global lm
    global considering 
    msg = (message.author.name,message.content)
    messages.append(msg)
    
    while len(str(messages)) > 1500:
        messages.pop(0)

    if message.author == client.user:
        return

    new_msg = True
    if considering:
        return
    else:
        lm = message 
    
inactive_counter = 0
@tasks.loop(seconds=10)
async def thought_tick():
    global inactive_counter
    global lm
    global new_msg
    global messages
    global ltm
    global changes
    global considering
    inactive_counter = inactive_counter + 1
    if new_msg:
        considering = True
        memory_string = ''
        inactive_counter = 0
        changes = True
        for msg in messages:
            memory_string = memory_string + "'" + msg[0] + "': " + msg[1] + "\n" 
        print("Chat History: " + memory_string)
        async with lm.channel.typing():
            # Run read_chat in a separate thread using asyncio.to_thread
            read_result = await asyncio.to_thread(read_chat, memory_string, ltm)

            # Check the result of read_chat and run generate_response if it returns True
            if read_result:
                response_generated = await asyncio.to_thread(generate_response, memory_string, ltm)
                print("Attemping to send response: " + response_generated)
                await lm.channel.send(response_generated)
                lm = None
                considering = False

        new_msg = False
    if inactive_counter > 1000:
        if changes:
            ltm = await asyncio.to_thread(compress_to_ltm, str(messages), ltm)
            messages = []
            inactive_counter = 0
            changes = False


    
client.run(TOKEN)

