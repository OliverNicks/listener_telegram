import asyncio
import os
import random
import time
from dotenv import load_dotenv
from src.drafter import draft_peer_recommendation
from src.classifier import IntentClassifier
from src.prefilter import find_keywords
from telethon import TelegramClient, events
from telethon.errors import FloodWaitError

load_dotenv()

# Configuration & Secrets
session_name = "listener_session"
API_ID = int(os.getenv("TG_API_ID"))
API_HASH = os.getenv("TG_API_HASH")
TARGET_CHAT_ID = int(os.getenv("TARGET_CHAT_ID", -5511274738))
COOLDOWN_SECONDS = int(os.getenv("THROTTLE_COOLDOWN_SECONDS", 300))

last_sent_time = {}  # Maps chat_id -> timestamp
client = TelegramClient(session_name, API_ID, API_HASH)

EXCLUDE_KEYWORDS = ["обмін", "обміняю", "обмен", "меняю"]

SHOPPING_ANCHORS = [
    # Local store / District / Trying-on discovery
    "Де на районі або в магазині можна поміряти зимовий комбінезон на дитину?",
    "Підкажіть дитячий магазин або шоурум, де поміряти одяг і комбінезони.",
    "Де можна приміряти та купити дитячий комбінезон Reima чи Lenne?",
    "Де на Позняках або Осокорках поміряти зимовий комбінезон Reima чи Lenne?",
    "Підкажіть, де поміряти і купити дитячий комбінезон Рейма?",
    # Trying-on & Store discovery
    (
        "Хочу поміряти на дитину зимові штани або комбінезон, підкажіть, де є в"
        " наявності?"
    ),
    "Де можна приміряти дитячий одяг перед покупкою, хто знає магазини?",
    # Core buyer anchors
    (
        "Шукаю зимовий комбінезон на хлопчика або дівчинку, розмір 92-104,"
        " пропонуйте варіанти."
    ),
    "Чи є ще в наявності цей теплий комбінезон? Хочу купити.",
    "Хочу купити дитячий одяг або костюмчик, підкажіть ціну та заміри.",
    "Який дитячий комбінезон краще купити на зиму?",
    (
        "Підкажіть, чи підійде цей комбінезон на худеньку дитину, чи він"
        " великомірить?"
    ),
    # RU Buying
    "Куплю детский зимний комбинезон, размер 86-92, в идеальном состоянии.",
    "Где можно купить теплые детские вещи, готова забрать и оплатить.",
    "Где можно примерить и купить детский комбинезон или штаны?",
    "Подскажите, маломерит ли этот комбинезон и какой точный размер?",
]

SELLER_ANCHORS = [
    # UA Selling
    "Продам наш дитячий зимовий комбінезон, стан ідеальний, ціна договірна.",
    (
        "Новий зимовий комбінезон з бірками, проліт з розміром 92, віддам за"
        " 900 грн."
    ),
    "Продам дитячі речі та комбінезон, розмір 92, віддам за хорошу ціну.",
    "Розвантажую шафу: продаю теплий дитячий одяг та комбінезони пакетом.",
    # RU Selling
    "Продам теплый зимний комбинезон на овчине, после одного ребенка.",
    "Срочно продаю детские вещи и комбинезон, состояние отличное, размер 92.",
    "Пролет с размером, отдам абсолютно новый комбинезон с бирками.",
]

print("⏳ Initializing classifier...")
classifier = IntentClassifier(
    model_path=r"models\model.onnx",
    tokenizer_path=r"models\tokenizer.json",
    buyer_anchors=SHOPPING_ANCHORS,
    seller_anchors=SELLER_ANCHORS,
    threshold=0.55,
)
print("✅ Ready!")


@client.on(events.NewMessage(chats=[TARGET_CHAT_ID]))
async def message_handler(event):
  if event.out:
    return

  chat_id = event.chat_id
  now = time.time()

  # Check chat-level cooldown
  if (
      chat_id in last_sent_time
      and (now - last_sent_time[chat_id]) < COOLDOWN_SECONDS
  ):
    remaining = int(COOLDOWN_SECONDS - (now - last_sent_time[chat_id]))
    print(f"⏳ Cooldown active for chat {chat_id}. Skipping ({remaining}s left).")
    return

  raw_text = event.raw_text or ""
  normalized_text = raw_text.lower()

  if any(word in normalized_text for word in EXCLUDE_KEYWORDS):
    print("⛔ Trade/Exchange detected. Ignoring.")
    return

  matches = find_keywords(normalized_text)

  if matches:
    print(f"🎯 Match detected {matches}: {normalized_text}")
    (
        is_buyer,
        is_seller,
        buy_score,
        sell_score,
        best_buy_anchor,
        best_sell_anchor,
    ) = classifier.predict(normalized_text)

    if is_buyer:
      # Lock cooldown slot immediately to prevent duplicate triggers during sleep
      last_sent_time[chat_id] = time.time()

      print(
          f"🛍️ Shopping inquiry confirmed (buy score: {buy_score:.4f}, sell"
          f" score: {sell_score:.4f})"
      )
      print(f'👤 User: {event.sender_id} | Message: "{raw_text}"')

      # 1. Generate LLM draft using original casing
      peer_draft = draft_peer_recommendation(raw_text)
      reply_text = peer_draft.reply_draft

      # 2. Simulate human reading/typing delay
      delay = random.uniform(10, 25)
      print(f"⏳ Waiting {delay:.1f}s before sending reply:\n{reply_text}")
      await asyncio.sleep(delay)

      # 3. Dispatch reply with FloodWait handling
      try:
        await client.send_message(
            chat_id, reply_text, reply_to=event.message.id
        )
        last_sent_time[chat_id] = time.time()
        print("🚀 Message sent successfully!")

      except FloodWaitError as e:
        print(f"⚠️ Telegram FloodWait triggered! Sleep required: {e.seconds}s.")
        last_sent_time[chat_id] = time.time() + e.seconds

        if e.seconds <= 60:
          print(f"⏳ Sleeping {e.seconds}s before retrying...")
          await asyncio.sleep(e.seconds)
          try:
            await client.send_message(
                chat_id, reply_text, reply_to=event.message.id
            )
            last_sent_time[chat_id] = time.time()
            print("🚀 Message sent successfully on retry!")
          except Exception as retry_err:
            print(f"❌ Retry failed: {retry_err}")
        else:
          print("⏭️ Wait too long. Dropping to protect session.")

      except Exception as err:
        print(f"❌ Failed to send message: {err}")

    elif is_seller:
      print(
          f"🪝 Seller inquiry confirmed (buy: {buy_score:.4f}, sell:"
          f" {sell_score:.4f})"
      )
      print(
          f"Best buy anchor: {best_buy_anchor}\nBest sell anchor:"
          f" {best_sell_anchor}"
      )

    else:
      print(
          "❌ Neither meets threshold\n(buy score:"
          f" {buy_score:.4f}, sell score: {sell_score:.4f})"
      )
      print(
          f"Best buy anchor: {best_buy_anchor}\nBest sell anchor:"
          f" {best_sell_anchor}"
      )

  print("-" * 75)


with client:
  client.run_until_disconnected()