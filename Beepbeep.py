import discord
from discord.ext import commands
import aiosqlite
import cv2
from ultralytics import YOLO
import os
import asyncio

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f"{bot.user} is working :-D")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if message.attachments:
        for attachment in message.attachments:
            if attachment.content_type.startswith("image/"):
                
                img_bytes = await attachment.read()
                filename = f"received_{attachment.filename}"

                with open(filename, "wb") as f:
                    f.write(img_bytes)

                await message.channel.send("Processing...")

                # Start predict
                model = YOLO('runs/detect/train/weights/best.pt')
                img = cv2.imread(filename)

                results = model(img, verbose=False)[0]
                print(results.boxes)

                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    label = f"{model.names[cls]} ({conf:.2f})"

                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f"Beep! Beep!: {label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.imwrite(f"result_{filename}.jpg", img) # SAVE OUTPUT

                await message.channel.send(file=discord.File(f"result_{filename}.jpg"))
                os.remove(f"result_{filename}.jpg")
                os.remove(filename)

                return

    await bot.process_commands(message)

bot.run(" CHANGE THIS TO YOUR BOT TOKEN :-D ")