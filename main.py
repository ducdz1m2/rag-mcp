import ollama
from database import load_db
import time
import numpy as np

class MCPDispatcher:
    def __init__(self):
     
        self.tools = {
            "rag_search": {
                "keywords": ["về", "là gì", "tài liệu", "thông tin", "quy định", "máy móc", "hướng dẫn"],
                "description": "Truy xuất kiến thức từ database nội bộ"
            },
            "general_chat": {
                "keywords": ["chào", "hi", "hello", "tạm biệt", "cảm ơn", "bạn là ai"],
                "description": "Tán gẫu hoặc chào hỏi"
            }
        }

    def fast_route(self, query):
        query_lower = query.lower()
        # Ưu tiên RAG nếu câu hỏi dài hoặc có từ khóa chuyên môn
        if len(query_lower.split()) > 6:
            return "rag_search"
            
        for tool_name, config in self.tools.items():
            if any(k in query_lower for k in config["keywords"]):
                return tool_name
        
        return "rag_search"

# --- KHỞI TẠO HỆ THỐNG ---
print("Khởi tạo hệ thống MCP...")
dispatcher = MCPDispatcher()
try:
    vectorstore = load_db()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3, "fetch_k": 10})
except:
    retriever = None

def ask_bot():
    while True:
        user_query = input("\n👉 Câu hỏi: ").strip()
        if not user_query: continue
        if user_query.lower() in ['exit', 'quit']: break

        t_start = time.perf_counter()

        
        selected_tool = dispatcher.fast_route(user_query)
        
        context_text = ""
        if selected_tool == "rag_search" and retriever:
            print("🔍 [Internal Search]...", end="\r")
            docs = retriever.invoke(user_query)
            context_text = "\n\n".join(d.page_content[:150] for d in docs)
            prompt = f"Ngữ cảnh: {context_text}\n\nCâu hỏi: {user_query}\nTrả lời ngắn gọn:"
        else:
            prompt = user_query

        t_prep = time.perf_counter() - t_start

        print(f"Bot ({selected_tool} | prep: {t_prep:.3f}s): ", end="", flush=True)

        try:
            stream = ollama.chat(
                model="qwen2.5:0.5b",
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                options={"temperature": 0.1}
            )

            for chunk in stream:
                print(chunk['message']['content'], end="", flush=True)
            print()

        except Exception as e:
            print(f"\nLỗi: {e}")

if __name__ == "__main__":
    ask_bot()