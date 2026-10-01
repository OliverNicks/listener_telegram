import asyncio
import os
import random

from dotenv import load_dotenv
from question_generator import generate_parent_question
from telethon import TelegramClient
from telethon.errors import FloodWaitError

load_dotenv()

# Can use the same API_ID & API_HASH from my.telegram.org,
# but uses a separate session file name for Account A.
API_ID = int(os.getenv("TELEGRAM_API_ID"))
API_HASH = os.getenv("TELEGRAM_API_HASH")

# Target group username or invite link ID (e.g. '@chat_username' or entity ID)
TARGET_CHAT = os.getenv("TARGET_CHAT_USERNAME")

async def post_question():
    client = TelegramClient("account_a_session", API_ID, API_HASH)
    await client.start()

    me = await client.get_me()
    print(f"Logged in as Account A: {me.first_name} (@{me.username})")

    # 1. Generate text
    text = generate_parent_question()
    print(f"\nGenerated Question:\n\"{text}\"\n")

    try:
        # 2. Simulate human presence: send 'typing' action for a realistic duration
        print(f"Simulating typing in {TARGET_CHAT}...")
        async with client.action(TARGET_CHAT, 'typing'):
            # Realistic typing time based on message length (approx 4-8 seconds)
            await asyncio.sleep(random.uniform(4.5, 8.0))

        # 3. Send message
        sent_msg = await client.send_message(TARGET_CHAT, text)
        print(f"Sent successfully! Message ID: {sent_msg.id}")

    except FloodWaitError as e:
        print(f"Telegram FloodWait: need to wait {e.seconds} seconds.")
    except Exception as e:
        print(f"Error sending message: {e}")
    finally:
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(post_question())
