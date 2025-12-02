import os
from openai import OpenAI
import numpy as np
from sentence_transformers import SentenceTransformer

# Загружаем векторизованное ТЗ
z = np.load("index.npz", allow_pickle=True)
embs = z["embs"]
texts = z["texts"].tolist()

# Модель для векторизации запросов
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Инициализация клиента OpenAI (DeepSeek)
client = OpenAI(
    api_key='sk-052d89cfe6be4d7d815c128ec700ba00',  # Замените на свой ключ
    base_url="https://api.deepseek.com"
)

def find_relevant_chunks(query, top_k=3):
    """Находит релевантные разделы ТЗ"""
    query_emb = model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
    sims = embs @ query_emb
    top_indices = sims.argsort()[-top_k:][::-1]
    
    relevant_chunks = []
    for idx in top_indices:
        relevant_chunks.append({
            'text': texts[idx],
            'score': float(sims[idx])
        })
    
    return relevant_chunks

def format_context(chunks):
    """Форматирует контекст для промпта"""
    context = "КОНТЕКСТ ИЗ ТЕХНИЧЕСКОГО ЗАДАНИЯ:\n\n"
    for i, chunk in enumerate(chunks, 1):
        context += f"[Документ {i}, сходство: {chunk['score']:.3f}]\n"
        context += f"{chunk['text'][:500]}...\n\n"
    return context

while True:
    user_text = input("> ").strip()
    
    if user_text.lower() in ['выход', 'exit', 'quit']:
        print("Выход из программы")
        break
    
    # 1. Находим релевантные разделы
    print("\n🔍 Поиск релевантных разделов...")
    relevant = find_relevant_chunks(user_text, top_k=2)
    
    print(f"Найдено {len(relevant)} релевантных разделов")
    for i, chunk in enumerate(relevant, 1):
        title = chunk['text'].split('\n')[0]
        print(f"{i}. {title[:50]}... (сходство: {chunk['score']:.3f})")
    
    # 2. Формируем контекст
    context = format_context(relevant)
    
    # 3. Отправляем запрос к нейросети
    print("\n🤖 Генерация ответа...")
    
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Ты помощник, который отвечает на вопросы о техническом задании Telegram-бота для кофейни. Используй только информацию из предоставленного контекста."},
                {"role": "user", "content": f"{context}\n\nВопрос: {user_text}"},
            ],
            stream=False
        )
        print("\n💡 Ответ:", resp.choices[0].message.content, "\n")
    except Exception as e:
        print(f"\n❌ Ошибка при запросе к нейросети: {e}")
        print("Пример ответа (заглушка):")
        print("Согласно ТЗ, Telegram-бот для кофейни должен иметь...\n")
