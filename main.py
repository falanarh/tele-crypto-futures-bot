# telegram_crypto_bot.py
# Bot Telegram untuk analisis dan trading crypto futures

import os
import time
import logging
from datetime import datetime

from dotenv import load_dotenv

import numpy as _np
if not hasattr(_np, 'NaN'):
    _np.NaN = _np.nan

import pandas as pd
import pandas_ta as ta
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from binance.client import Client as BinanceClient

from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, ContextTypes

# ========== MUAT ENV ==========
load_dotenv()

BINANCE_API_KEY    = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')
TELEGRAM_TOKEN     = os.getenv('TELEGRAM_TOKEN')
SYMBOL_DEFAULT     = os.getenv('SYMBOL_DEFAULT', 'BTCUSDT')
INTERVAL_DEFAULT   = os.getenv('INTERVAL_DEFAULT', '1h')
LOOKBACK_SIGNALS   = int(os.getenv('LOOKBACK_SIGNALS', 100))

# ========== SETUP SESSION ==========
session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)

binance = BinanceClient(
    api_key=BINANCE_API_KEY,
    api_secret=BINANCE_API_SECRET,
    requests_params={'timeout': 20}
)

bot = Bot(token=TELEGRAM_TOKEN)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# ========== UTILITY FUNCTIONS ==========

def fetch_klines(symbol: str, interval: str = INTERVAL_DEFAULT, limit: int = 500) -> pd.DataFrame:
    for attempt in range(3):
        try:
            raw = binance.futures_klines(symbol=symbol, interval=interval, limit=limit)
            df = pd.DataFrame(raw, columns=[
                'open_time','open','high','low','close','volume',
                'close_time','quote_vol','trades','taker_buy_base','taker_buy_quote','ignore'
            ])
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
            return df[['open_time','open','high','low','close','volume']]
        except Exception as e:
            logging.error(f"Fetch klines gagal (attempt {attempt+1}): {e}")
            time.sleep(1)
    raise ConnectionError("Gagal mengambil data klines dari Binance setelah beberapa percobaan")


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    df['EMA_fast'] = ta.ema(df['close'], length=10)
    df['EMA_slow'] = ta.ema(df['close'], length=30)
    df['RSI']      = ta.rsi(df['close'], length=14)
    df['ATR']      = ta.atr(df['high'], df['low'], df['close'], length=14)

    df['signal'] = None
    for i in range(1, len(df)):
        prev, curr = df.iloc[i-1], df.iloc[i]
        if prev['EMA_fast'] < prev['EMA_slow'] and curr['EMA_fast'] > curr['EMA_slow'] and curr['RSI'] < 70:
            df.at[i, 'signal'] = 'LONG'
        elif prev['EMA_fast'] > prev['EMA_slow'] and curr['EMA_fast'] < curr['EMA_slow'] and curr['RSI'] > 30:
            df.at[i, 'signal'] = 'SHORT'
    return df


def calculate_win_rate(symbol: str, interval: str = INTERVAL_DEFAULT, lookback: int = LOOKBACK_SIGNALS) -> float:
    df = fetch_klines(symbol, interval, limit=lookback + 50)
    df = generate_signals(df)
    signals = df.dropna(subset=['signal']).tail(lookback)
    wins = 0
    for idx, row in signals.iterrows():
        next_price = df['close'].shift(-1).iloc[idx]
        if (row['signal'] == 'LONG' and next_price > row['close']) or \
           (row['signal'] == 'SHORT' and next_price < row['close']):
            wins += 1
    return round((wins / len(signals)) * 100, 2) if len(signals) else 0.0


def analyze_trend(df: pd.DataFrame) -> dict:
    df  = generate_signals(df)
    last = df.dropna(subset=['signal']).iloc[-1:]
    if last.empty:
        return {'signal': None}
    row       = last.iloc[0]
    entry     = row['close']
    atr       = row['ATR']
    direction = row['signal']
    sl        = entry - atr * (1 if direction == 'LONG' else -1)
    tp        = entry + atr * 2 * (1 if direction == 'LONG' else -1)
    return {
        'signal':      direction,
        'entry':       entry,
        'stop_loss':   sl,
        'take_profit': tp,
        'rsi':         row['RSI']
    }


def fetch_news(symbol: str) -> list:
    url = f"https://api.coingecko.com/api/v3/coins/{symbol.lower()}/status_updates"
    try:
        res = session.get(url, timeout=10)
        data = res.json().get('status_updates', [])[:3]
    except Exception as e:
        logging.error(f"Fetch news gagal: {e}")
        return []
    news = []
    for u in data:
        news.append({
            'date':  u['created_at'],
            'title': u['title'],
            'url':   u['user']['twitter_handle']
        })
    return news


def place_order(symbol: str, side: str, quantity: float):
    try:
        return binance.futures_create_order(
            symbol=symbol,
            side=side,
            type='MARKET',
            quantity=quantity
        )
    except Exception as e:
        logging.error(f"Place order error: {e}")
        raise

# ========== HANDLERS (ASYNC) ==========

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "Halo! Gunakan /signal SYMBOL untuk sinyal trading.\n"
        "Contoh: /signal BTCUSDT\n"
        "/order SYMBOL SIDE QTY untuk eksekusi."
    )
    await update.message.reply_text(text)

async def signal_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbol = context.args[0].upper() if context.args else SYMBOL_DEFAULT
    try:
        df       = fetch_klines(symbol)
        info     = analyze_trend(df)
        win_rate = calculate_win_rate(symbol)
        news     = fetch_news(symbol[:-4])

        text = f"📈 Sinyal {symbol}: {info.get('signal')}\n"
        if info.get('signal'):
            text += (
                f"Entry: {info['entry']:.2f}\n"
                f"SL: {info['stop_loss']:.2f}\n"
                f"TP: {info['take_profit']:.2f}\n"
                f"RSI: {info['rsi']:.1f}\n"
            )
        text += f"⚖️ Win rate historis ({LOOKBACK_SIGNALS}): {win_rate}%\n"
        text += "\n📰 Berita terbaru:\n"
        for n in news:
            text += f"- [{n['title']}]({n['url']}) {n['date'][:10]}\n"
    except Exception as e:
        logging.error(f"Signal handler error: {e}")
        text = f"Gagal mengambil data sinyal untuk {symbol}. Silakan coba lagi nanti."

    await update.message.reply_text(text, parse_mode='Markdown')

async def order_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        symbol, side, qty = context.args
        qty = float(qty)
        order = place_order(symbol.upper(), side.upper(), qty)
        await update.message.reply_text(f"✅ Order placed: {order['orderId']} {side} {qty} {symbol}")
    except Exception:
        await update.message.reply_text(
            "Format: /order SYMBOL SIDE(Q BUY/SELL) QTY. Contoh: /order BTCUSDT BUY 0.001"
        )

# ========== MAIN FUNCTION (ASYNC) ==========

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("signal", signal_handler))
    app.add_handler(CommandHandler("order", order_handler))

    logging.info("Bot starting...")
    app.run_polling()

if __name__ == "__main__":
    main()
