import asyncio
import os
import random
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Ensure project root is in sys.path so the bot can be run from any directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from telethon import TelegramClient, events
from telethon.errors import FloodWaitError

from src.catalog import CategoryCatalog
from src.classifier import IntentClassifier
from src.drafter import draft_peer_recommendation
from src.logger import logger
from src.prefilter import find_keywords

load_dotenv(PROJECT_ROOT / ".env")

# Configuration & Secrets
session_name = str(PROJECT_ROOT / "listener_session")
API_ID = int(os.getenv("TG_API_ID"))
API_HASH = os.getenv("TG_API_HASH")
TARGET_CHAT_ID = int(os.getenv("TARGET_CHAT_ID", "-5511274738"))
COOLDOWN_SECONDS = int(os.getenv("THROTTLE_COOLDOWN_SECONDS", "300"))

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


MODEL_PATH = PROJECT_ROOT / "models" / "model.onnx"
TOKENIZER_PATH = PROJECT_ROOT / "models" / "tokenizer.json"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_catalog_db"
if not CHROMA_DIR.exists():
    CHROMA_DIR = PROJECT_ROOT / "chroma_catalog_db"

logger.info("Initializing classifier...", model_path=str(MODEL_PATH))
classifier = IntentClassifier(
    model_path=str(MODEL_PATH),
    tokenizer_path=str(TOKENIZER_PATH),
    buyer_anchors=SHOPPING_ANCHORS,
    seller_anchors=SELLER_ANCHORS,
    threshold=0.55,
)
logger.info("Classifier ready")
logger.info("Initializing Chroma catalog...", collection="best4baby_categories", persist_dir=str(CHROMA_DIR))

# Initialize catalog
catalog = CategoryCatalog(
    persist_dir=str(CHROMA_DIR),
    collection_name="best4baby_categories",
)
logger.info("Catalog ready")


@client.on(events.NewMessage(chats=[TARGET_CHAT_ID]))
async def message_handler(event):
    if event.out:
        return

    chat_id = event.chat_id
    now = time.time()

    log = logger.bind(
        chat_id=chat_id,
        user_id=event.sender_id,
        msg_id=event.id
    )

    # Check chat-level cooldown
    if (
        chat_id in last_sent_time
        and (now - last_sent_time[chat_id]) < COOLDOWN_SECONDS
    ):
        remaining = int(COOLDOWN_SECONDS - (now - last_sent_time[chat_id]))
        log.warning(
            "Cooldown active, skipping message",
            remaining_seconds=remaining,
            cooldown_seconds=COOLDOWN_SECONDS
        )
        return

    raw_text = event.raw_text or ""
    normalized_text = raw_text.lower()

    if any(word in normalized_text for word in EXCLUDE_KEYWORDS):
        log.info("Message skipped: trade/exchange detected")
        return

    matches = find_keywords(normalized_text)

    t_start = time.perf_counter()

    if matches:
        (
            is_buyer,
            is_seller,
            buy_score,
            sell_score,
            best_buy_anchor,
            best_sell_anchor,
        ) = classifier.predict(normalized_text)

        classify_latency = round((time.perf_counter() - t_start) * 1000, 2)

        # Logging the intent-classification time
        log.info(
            "Intent classified",
            is_buyer=is_buyer,
            is_seller=is_seller,
            buy_score=round(buy_score, 4),
            sell_score=round(sell_score, 4),
            latency_ms=classify_latency,
            matched_keywords=matches,
        )

        if is_buyer:
            # Lock cooldown slot immediately to prevent duplicate triggers during sleep
            last_sent_time[chat_id] = time.time()

            log.info(
                "Buyer inquiry confirmed",
                buy_score=round(buy_score, 4),
                sell_score=round(sell_score, 4),
                raw_message=raw_text,
            )

            # Measuring catalogue resolution
            t_cat = time.perf_counter()
            target_url = catalog.resolve_url(raw_text)
            cat_latency = round((time.perf_counter() - t_cat) * 1000, 2)

            log.info(
                "Resolved product category URL",
                target_url=target_url,
                latency_ms=cat_latency,
            )

            # 1. Generate LLM draft using original casing
            t_llm = time.perf_counter()
            peer_draft = draft_peer_recommendation(raw_text, store_link=target_url)
            llm_latency = round((time.perf_counter() - t_llm) * 1000, 2)
            log.info("Draft generated successfully", latency_ms=llm_latency)

            reply_text = peer_draft.reply_draft

            # 2. Simulate human reading/typing delay
            wait_time = random.uniform(180, 360)
            log.info(
                "Simulating reading/typing delay before sending reply",
                delay_seconds=round(wait_time, 1),
                reply_preview=reply_text[:100] + "..." if len(reply_text) > 100 else reply_text,
            )
            await asyncio.sleep(wait_time)

            # 3. Dispatch reply with FloodWait handling
            try:
                await client.send_message(
                    chat_id, reply_text, reply_to=event.message.id
                )
                last_sent_time[chat_id] = time.time()
                log.info("Reply message sent successfully")

            except FloodWaitError as e:
                log.error("Telegram FloodWait hit", wait_seconds=e.seconds)
                last_sent_time[chat_id] = time.time() + e.seconds

                if e.seconds <= 60:
                    log.warning("Sleeping before retry", retry_delay_seconds=e.seconds)
                    await asyncio.sleep(e.seconds)
                    try:
                        await client.send_message(
                            chat_id, reply_text, reply_to=event.message.id
                        )
                        last_sent_time[chat_id] = time.time()
                        log.info("Reply message sent successfully on retry")
                    except Exception as retry_err:
                        log.error("Retry failed", error=str(retry_err))
                else:
                    log.warning(
                        "FloodWait duration too long. Dropping reply to protect session.",
                        wait_seconds=e.seconds,
                    )

            except Exception:
                log.exception("Unexpected error while processing buyer inquiry")

        elif is_seller:
            log.info(
                "Seller inquiry detected",
                buy_score=round(buy_score, 4),
                sell_score=round(sell_score, 4),
                best_buy_anchor=best_buy_anchor,
                best_sell_anchor=best_sell_anchor,
            )

        else:
            log.debug(
                "Threshold not met",
                buy_score=round(buy_score, 4),
                sell_score=round(sell_score, 4),
                best_buy_anchor=best_buy_anchor,
                best_sell_anchor=best_sell_anchor,
            )


with client:
  client.run_until_disconnected()
