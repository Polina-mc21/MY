"""
RAG-система для технического задания Telegram-бота кофейни
Объединяет векторизацию, поиск и генерацию ответов
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os

class TZ_RAG_System:
    def __init__(self, index_file="index.npz", use_api=False):
        """
        Инициализация RAG-системы
        
        Args:
            index_file: файл с векторизованным ТЗ
            use_api: использовать ли реальный API (False для демо)
        """
        self.use_api = use_api
        
        # Загружаем векторизованное ТЗ
        z = np.load(index_file, allow_pickle=True)
        self.embs = z["embs"]
        self.texts = z["texts"].tolist()
        
        # Модель для векторизации
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        # Инициализация API (если используется)
        if use_api:
            self.client = OpenAI(
                api_key=os.getenv('OPENAI_API_KEY', 'ваш_ключ_здесь'),
                base_url="https://api.deepseek.com"
            )
        
        print(f"✅ RAG-система загружена. Разделов: {len(self.texts)}")
    
    def find_similar(self, query, top_k=3):
        """
        Находит наиболее похожие разделы ТЗ по запросу
        
        Args:
            query: текст запроса
            top_k: количество возвращаемых результатов
            
        Returns:
            Список словарей с информацией о найденных разделах
        """
        # Векторизуем запрос
        query_emb = self.model.encode(
            query, 
            normalize_embeddings=True, 
            convert_to_numpy=True
        )
        
        # Вычисляем косинусное сходство
        similarities = self.embs @ query_emb
        
        # Получаем топ-K наиболее похожих
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            chunk_text = self.texts[idx]
            lines = chunk_text.split('\n')
            title = lines[0] if lines else "Без заголовка"
            
            results.append({
                'index': int(idx),
                'similarity': float(similarities[idx]),
                'title': title,
                'content': chunk_text,
                'preview': chunk_text[:150] + "..."
            })
        
        return results
    
    def generate_answer(self, query, relevant_chunks):
        """
        Генерирует ответ на основе найденных разделов
        
        Args:
            query: исходный запрос
            relevant_chunks: найденные релевантные разделы
            
        Returns:
            Текст ответа
        """
        # Формируем контекст
        context = "ИНФОРМАЦИЯ ИЗ ТЕХНИЧЕСКОГО ЗАДАНИЯ:\n\n"
        for i, chunk in enumerate(relevant_chunks, 1):
            context += f"=== РАЗДЕЛ {i} ===\n"
            context += f"Заголовок: {chunk['title']}\n"
            context += f"Сходство: {chunk['similarity']:.3f}\n"
            context += f"Содержание:\n{chunk['content'][:500]}...\n\n"
        
        # Если используем реальный API
        if self.use_api and hasattr(self, 'client'):
            try:
                response = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "Ты помощник по техническому заданию Telegram-бота кофейни. Отвечай ТОЛЬКО на основе предоставленного контекста."},
                        {"role": "user", "content": f"{context}\n\nВопрос: {query}"}
                    ],
                    max_tokens=500,
                    temperature=0.3
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"Ошибка API: {str(e)}\n\n{self._generate_local_answer(query, relevant_chunks)}"
        else:
            # Локальная генерация (демо-режим)
            return self._generate_local_answer(query, relevant_chunks)
    
    def _generate_local_answer(self, query, relevant_chunks):
        """Генерация ответа без использования API"""
        answer = f"ОТВЕТ на вопрос: '{query}'\n\n"
        answer += "На основе технического задания:\n\n"
        
        for chunk in relevant_chunks:
            title = chunk['title']
            similarity = chunk['similarity']
            
            if similarity > 0.3:  # Только достаточно релевантные
                answer += f"📄 {title}\n"
                
                # Простая логика для демонстрации
                content = chunk['content'].lower()
                
                if "интерфейс" in query.lower() or "интерфейс" in content:
                    answer += "• Пользовательский интерфейс с кнопками: 'Меню', 'Корзина', 'История заказов', 'Акции'\n"
                    answer += "• Административный интерфейс для сотрудников\n"
                    answer += "• Интуитивно понятная навигация\n\n"
                
                elif "оплат" in query.lower() or "оплат" in content:
                    answer += "• Генерация QR-кода для оплаты\n"
                    answer += "• Уведомление об успешной оплате\n"
                    answer += "• Интеграция с платежной системой\n\n"
                
                elif "функци" in query.lower() or "функци" in content:
                    answer += "• Меню товаров с ценами\n"
                    answer += "• Корзина для выбора товаров\n"
                    answer += "• История заказов и статусы\n\n"
                
                else:
                    # Общий ответ
                    answer += f"• Содержится в разделе '{title}'\n"
                    answer += f"• Релевантность: {similarity:.3f}\n\n"
        
        if len(answer.split('\n')) < 10:  # Если ответ слишком короткий
            answer += "\n💡 Для более точного ответа уточните вопрос или используйте реальный API ключ."
        
        return answer
    
    def ask(self, query, top_k=3):
        """
        Основной метод для вопросов к системе
        
        Args:
            query: вопрос пользователя
            top_k: количество релевантных разделов для поиска
            
        Returns:
            Словарь с результатами
        """
        print(f"\n{'='*60}")
        print(f"❓ ВОПРОС: {query}")
        print('='*60)
        
        # 1. Поиск релевантных разделов
        print("\n🔍 Поиск в ТЗ...")
        relevant_chunks = self.find_similar(query, top_k)
        
        # 2. Вывод найденного
        print(f"\n📚 Найдено {len(relevant_chunks)} релевантных разделов:")
        for i, chunk in enumerate(relevant_chunks, 1):
            print(f"{i}. [{chunk['index']}] {chunk['title'][:50]}... (сходство: {chunk['similarity']:.3f})")
        
        # 3. Генерация ответа
        print("\n🤖 Генерация ответа...")
        answer = self.generate_answer(query, relevant_chunks)
        
        # 4. Вывод результата
        print(f"\n{'='*60}")
        print("💡 ОТВЕТ:")
        print('='*60)
        print(answer)
        
        return {
            'query': query,
            'relevant_chunks': relevant_chunks,
            'answer': answer
        }

def demo():
    """Демонстрация работы системы"""
    print("🚀 ДЕМОНСТРАЦИЯ RAG-СИСТЕМЫ")
    print("="*60)
    
    # Проверяем наличие файла с векторами
    if not os.path.exists("index.npz"):
        print("❌ Файл index.npz не найден!")
        print("Сначала запустите: python md_to_vectors.py TZ.md")
        return
    
    # Создаем систему
    rag = TZ_RAG_System("index.npz", use_api=False)
    
    # Тестовые запросы
    test_queries = [
        "Какие требования к интерфейсу бота?",
        "Как происходит оплата заказа?",
        "Какой функционал должен быть у бота?",
        "Какие интеграции нужны для работы бота?"
    ]
    
    for query in test_queries[:2]:  # Тестируем первые 2 запроса
        result = rag.ask(query, top_k=2)
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    demo()
