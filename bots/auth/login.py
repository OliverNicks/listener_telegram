from telethon import TelegramClient

# 1. Choose a name for your session file
session_name = "listener_session"

# 2. Your API credentials
api_id = 34360356  # replace with your api_id integer
api_hash = "814cadefc1a6c475c6b2cc2ff9723206"  # replace with your api_hash string

# 3. Create the client instance
client = TelegramClient(session_name, api_id, api_hash)


async def main():
    # client.start() prompts for phone number, code, and 2FA password if needed
    await client.start()
    print("Login successful! Session file created.")


with client:
    client.loop.run_until_complete(main())