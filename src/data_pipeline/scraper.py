import asyncio
from telethon.sync import TelegramClient
from telethon.tl.types import MessageMediaPhoto
import yaml
import json
import os
import pandas as pd
from datetime import datetime

# Load configuration
def load_config(config_path='configs/scraping_config.yaml'):
    """Loads configuration from a YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Initialize Telegram client
async def initialize_client(api_id, api_hash, session_name='telegram_scraper_session'):
    """
    Initializes and connects the Telegram client.
    Handles interactive authorization if the session is not already authorized.
    """
    client = TelegramClient(session_name, api_id, api_hash)
    print("Connecting to Telegram...")
    await client.connect()

    if not await client.is_user_authorized():
        print("\n--- Telegram Authorization Required ---")
        print("Please authorize this Telegram client. You will be prompted for your phone number and verification code.")
        print("Telegram will send a code to your official Telegram app.")
        await client.start() # This line handles the interactive login process
        print("Client authorized successfully!")
    else:
        print("Client already authorized. Reusing existing session.")
    return client

# Fetch messages from a channel
async def fetch_channel_messages(client, channel_name, limit=None):
    """
    Fetches messages from a specified Telegram channel,
    including downloading images and saving their paths.
    """
    print(f"Fetching messages from channel: {channel_name}")
    messages_data = []
    try:
        # Get the channel entity by its username
        entity = await client.get_entity(channel_name)

        # Define directory for saving images for this channel
        image_download_dir = os.path.join('data', 'raw', 'telegram', channel_name.replace('@', ''), 'images')
        os.makedirs(image_download_dir, exist_ok=True)

        # Iterate through messages
        async for message in client.iter_messages(entity, limit=limit):
            image_path = None
            # Check if the message contains a photo
            if message.media and isinstance(message.media, MessageMediaPhoto):
                try:
                    # Construct a unique filename for the image
                    filename = f"{channel_name.replace('@', '')}_{message.id}.png"
                    file_path = os.path.join(image_download_dir, filename)
                    # Download the photo
                    downloaded_file = await client.download_media(message.media, file=file_path)
                    if downloaded_file:
                        image_path = os.path.abspath(downloaded_file) # Store absolute path
                        print(f"Downloaded image for message {message.id} to {image_path}")
                except Exception as e:
                    print(f"Error downloading image for message {message.id} from {channel_name}: {e}")

            message_info = {
                'channel_name': channel_name,
                'message_id': message.id,
                'sender_id': message.sender_id,
                'date': message.date.isoformat(),
                'text': message.text,
                'views': message.views,
                'media_type': message.media.to_dict()['_'] if message.media else None,
                'image_local_path': image_path, # New field to store the path to the downloaded image
                'is_forward': message.forward is not None,
                'reply_to_msg_id': message.reply_to_msg_id,
                'edit_date': message.edit_date.isoformat() if message.edit_date else None,
                'grouped_id': message.grouped_id,
                'reactions': message.reactions.to_dict()['results'] if message.reactions else None
            }
            messages_data.append(message_info)
        print(f"Successfully fetched {len(messages_data)} messages from {channel_name}.")
    except Exception as e:
        print(f"Error fetching messages from {channel_name}: {e}")
    return messages_data

async def main():
    config = load_config()
    api_id = config['telegram']['api_id']
    api_hash = config['telegram']['api_hash']
    channels_to_scrape = config['telegram']['channels']

    if api_id == "YOUR_API_ID" or api_hash == "YOUR_API_HASH":
        print("ERROR: Please update 'api_id' and 'api_hash' in configs/scraping_config.yaml with your actual credentials from my.telegram.org.")
        return

    client = await initialize_client(api_id, api_hash)

    all_scraped_data = []
    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for channel in channels_to_scrape:
        # Remove '@' from channel name for directory creation if present
        sanitized_channel_name = channel.replace('@', '')
        messages = await fetch_channel_messages(client, sanitized_channel_name, limit=500) # Fetching up to 500 messages per channel
        all_scraped_data.extend(messages)

        # Store raw data for each channel in its specific directory
        channel_dir = os.path.join('data', 'raw', 'telegram', sanitized_channel_name)
        os.makedirs(channel_dir, exist_ok=True)
        channel_file_path = os.path.join(channel_dir, f'{current_timestamp}_raw_messages.json')
        with open(channel_file_path, 'w', encoding='utf-8') as f:
            json.dump(messages, f, ensure_ascii=False, indent=4)
        print(f"Saved raw messages for {channel} to {channel_file_path}")

    await client.disconnect()
    print("Telegram client disconnected.")

    # Store all raw data in a single JSON for easier overview and subsequent processing
    raw_data_path = os.path.join('data', 'raw', 'all_raw_telegram_messages.json')
    with open(raw_data_path, 'w', encoding='utf-8') as f:
        json.dump(all_scraped_data, f, ensure_ascii=False, indent=4)
    print(f"All raw messages combined and saved to {raw_data_path}")

    print("\nScraping process complete (images downloaded). Next, run preprocessor.py to clean and structure the data, now including OCR.")

if __name__ == '__main__':
    asyncio.run(main())