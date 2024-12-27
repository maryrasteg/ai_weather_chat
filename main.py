import asyncio
import torch
import json
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, StoppingCriteria, StoppingCriteriaList
from weather import get_weather

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_ids: list):
        self.keywords = keywords_ids

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        if input_ids[0][-1] in self.keywords:
            return True
        return False

active_connections: Dict[str, WebSocket] = {}
chat_messages: Dict[tuple, List[dict]] = {}

def get_chat_key(u1: str, u2: str) -> tuple:
    return tuple(sorted([u1, u2]))

MODEL_NAME = "lambdalabs/pythia-2.8b-deduped-synthetic-instruct"

tokenizer = None
model = None
device = None
stop_criteria = None

stop_token = "<|stop|>"
max_new_tokens = 2048
CHATS_JSON_FILE = "chats.json"

def save_chats_to_json():
    """Сохраняем все диалоги в JSON-файл, преобразовав ключи-кортежи в строки."""
    data_to_save = {}
    for chat_key, messages in chat_messages.items():
        data_to_save[str(chat_key)] = messages  # ключ в виде строки

    with open(CHATS_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=4)

async def lifespan(app: FastAPI):
    """
    При запуске приложения загружаем токенайзер и модель из Hugging Face,
    которые поддерживают русский язык.
    """
    global tokenizer, model, device, stop_token, stop_criteria

    print(f"Загружаем модель {MODEL_NAME}...")

    tokenizer_local = AutoTokenizer.from_pretrained(MODEL_NAME)
    model_local = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    tokenizer_local.pad_token = tokenizer_local.eos_token
    tokenizer_local.add_tokens([stop_token])
    stop_ids = [tokenizer_local.encode(w)[0] for w in [stop_token]]
    stop_criteria = KeywordsStoppingCriteria(stop_ids)

    device_local = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_local.to(device_local)
    model_local.eval()

    tokenizer = tokenizer_local
    model = model_local
    device = device_local

    print(f"Модель {MODEL_NAME} загружена на {device}.")

    # Если при старте файла уже существует chats.json — загружаем
    if os.path.exists(CHATS_JSON_FILE):
        with open(CHATS_JSON_FILE, "r", encoding="utf-8") as f:
            old_data = json.load(f)
            # old_data — словарь, где ключи это строки, например "('Alice', 'Bob')"
            for key_str, msgs in old_data.items():
                try:
                    # Попробуем превратить строку обратно в tuple
                    tuple_key = eval(key_str)
                    chat_messages[tuple_key] = msgs
                except:
                    pass

    yield  # дальнейший код (если нужен) идёт после yield — но здесь пусто


app = FastAPI(lifespan=lifespan)

async def call_local_model(messages: List[dict]) -> str:
    """
    Генерируем ответ с помощью локальной HuggingFace-модели на последнее сообщение.
    На вход: список сообщений в формате [{'message': str, 'from_you': bool}, ...].
    Возвращает строку — ответ модели на русском языке.
    """
    global model, tokenizer, device, stop_criteria

    if not messages:
        prompt = "Привет! О чём поговорим?"
    else:
        # Берём последнее сообщение
        prompt = messages[-1]["message"]
        print(f"[AI] Последнее сообщение для генерации: {prompt}")

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=max_new_tokens,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        stopping_criteria=StoppingCriteriaList([stop_criteria]),
    )
    result = generator(prompt, num_return_sequences=1)

    # Отрезаем служебный токен, если он там есть (примерно)
    output = result[0]["generated_text"].replace(stop_token, "")
    if "Answer" in output:
        output = output.split('Answer')[1]
    print("generated_text:", output)

    if not output:
        output = "Честно говоря, я затрудняюсь с ответом."

    return output

@app.get("/")
async def get_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

async def generate_and_send_answer(
    websocket: WebSocket,
    messages: List[dict],
    from_user: str
):
    """
    Вызываем локальную модель, получаем ответ, и отправляем сообщение по вебсокету.
    Вызывается только при нажатии кнопки "Сгенерировать ответ".
    """
    ai_text = await call_local_model(messages)
    await websocket.send_json({
        "type": "ai_generated",
        "from_user": from_user,
        "message": ai_text
    })

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    init_data = await websocket.receive_json()
    user_name = init_data.get("user_name")
    if not user_name:
        await websocket.close()
        return

    active_connections[user_name] = websocket
    await broadcast_user_list()

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "request_chat_history":
                from_user = data["from_user"]
                to_user = data["to_user"]
                ckey = get_chat_key(from_user, to_user)
                history = chat_messages.get(ckey, [])
                await websocket.send_json({
                    "type": "chat_history",
                    "chat": history
                })

            elif msg_type == "send_message":
                # Отправка пользовательского сообщения
                from_user = data["from_user"]
                to_user = data["to_user"]
                text = data["message"]

                ckey = get_chat_key(from_user, to_user)
                if ckey not in chat_messages:
                    chat_messages[ckey] = []
                chat_messages[ckey].append({"message": text, "from_you": True})
                
                # Сохраняем сразу же
                save_chats_to_json()

                # Отправляем получателю, если он онлайн
                if to_user in active_connections:
                    to_ws = active_connections[to_user]
                    await to_ws.send_json({
                        "type": "receive_message",
                        "from_user": from_user,
                        "message": text
                    })

            elif msg_type == "generate_ai_answer":
                # Генерация ответа нейросетью только по нажатию специальной кнопки
                from_user = data["from_user"]
                to_user = data["to_user"]

                ckey = get_chat_key(from_user, to_user)
                messages = chat_messages.get(ckey, [])
                # Фоновая задача
                asyncio.create_task(generate_and_send_answer(
                    websocket,  # отвечаем только инициатору
                    messages,
                    from_user="AI"
                ))

            elif msg_type == "send_ai_to_chat":
                # Когда пользователь отправляет предложенный ответ в чат
                from_user = data["from_user"]
                to_user = data["to_user"]
                text = data["message"]

                ckey = get_chat_key(from_user, to_user)
                if ckey not in chat_messages:
                    chat_messages[ckey] = []

                from_you_flag = data.get("as_user", True)

                chat_messages[ckey].append({"message": text, "from_you": from_you_flag})
                save_chats_to_json()

                # Отправляем собеседнику
                if to_user in active_connections:
                    to_ws = active_connections[to_user]
                    await to_ws.send_json({
                        "type": "receive_message",
                        "from_user": ("ChatGPT" if not from_you_flag else from_user),
                        "message": text
                    })

            elif msg_type == "request_weather":
                # Пользователь нажал "Узнать погоду"
                from_user = data["from_user"]
                to_user = data["to_user"]

                ckey = get_chat_key(from_user, to_user)
                if ckey not in chat_messages:
                    chat_messages[ckey] = []

                weather_text = get_weather('perm')  # допустим, вернёт строку
                # Добавляем в историю чата
                chat_messages[ckey].append({"message": weather_text, "from_you": False})
                save_chats_to_json()

                # Показываем погоду обоим участникам
                # От кого "System" или "Погода" — как вам удобнее
                if to_user in active_connections:
                    to_ws = active_connections[to_user]
                    await to_ws.send_json({
                        "type": "receive_weather",
                        "from_user": "System",
                        "message": weather_text
                    })
                if from_user in active_connections:
                    from_ws = active_connections[from_user]
                    await from_ws.send_json({
                        "type": "receive_weather",
                        "from_user": "System",
                        "message": weather_text
                    })

            else:
                print("Неизвестный тип сообщения:", msg_type)

    except WebSocketDisconnect:
        if user_name in active_connections:
            del active_connections[user_name]
        await broadcast_user_list()

async def broadcast_user_list():
    users_list = list(active_connections.keys())
    for ws in active_connections.values():
        await ws.send_json({
            "type": "user_list",
            "users": users_list
        })
