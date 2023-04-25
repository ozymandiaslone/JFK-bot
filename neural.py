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

# INTENT
intents = discord.Intents.default()
intents.message_content = True

# Initialize the Discord client
client = discord.Client(intents=intents)

# Global variables for storing chat history and the last message received
messages = []

@client.event
async def on_ready():
    print(f'Logged in as {client.user}')
    thought_tick.start()

@client.event
async def on_message(message):
    global new_msg
    global lm
    msg = {message.author.name:message.content}
    messages.append(msg)
    
    while len(str(messages)) > 4000:
        messages.pop(0)

    if message.author == client.user:
        return

    lm = message
    new_msg = True
    
inactive_counter = 0
@tasks.loop(seconds=10)
async def thought_tick():
    global inactive_counter
    global lm
    global new_msg
    global messages
    global ltm
    global changes
    inactive_counter = inactive_counter + 1
    if new_msg:
        inactive_counter = 0
        changes = True
        print("Chat History: " + str(messages))
        async with lm.channel.typing():
            # Run read_chat in a separate thread using asyncio.to_thread
            read_result = await asyncio.to_thread(read_chat, str(messages), ltm)

            # Check the result of read_chat and run generate_response if it returns True
            if read_result:
                generate_result = await asyncio.to_thread(generate_response, str(messages), ltm)
                print("Attemping to send response: " + generate_result)
                await lm.channel.send(generate_result)
                lm = None

        new_msg = False
    if inactive_counter > 90:
        if changes:
            ltm = await asyncio.to_thread(compress_to_ltm, str(messages), ltm)
            inactive_counter = 0
            changes = False


    
client.run(TOKEN)

