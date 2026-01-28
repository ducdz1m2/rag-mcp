import ollama
from database import load_db
import sys
import time

# 1. LOAD DB RA NGOÀI VÒNG LẶP: Chỉ load 1 lần duy nhất khi khởi động script
print("🚀 Đang khởi tạo hệ thống và load dữ liệu...")
try:
    vectorstore = load_db()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3} # Giảm k xuống 3 để nhanh hơn nếu tài liệu chất lượng
    )
    print("✅ Hệ thống sẵn sàng!")
except Exception as e:
    print(f"❌ Lỗi: {e}")
    sys.exit(1)

def ask_bot():
    while True:
        user_query = input("\n👉 Câu hỏi: ").strip()
        if user_query.lower() in ['exit', 'quit', '']:
            break

        # Bắt đầu tính giờ
        t_start = time.perf_counter()
        
        # Hiển thị trạng thái ngay lập tức
        print("🔍 Đang tìm tài liệu...", end="\r", flush=True)
      
        docs = retriever.invoke(user_query)
   
        t_retrieve = time.perf_counter() - t_start

        context_text = "\n\n".join(
            d.page_content[:400].strip()
            for d in docs if d.page_content.strip()
        )

        full_prompt = f"""Trả lời ngắn gọn dựa trên tài liệu:
{context_text}
Câu hỏi: {user_query}"""

        print(f"🤖 Trả lời ({t_retrieve:.2f}s): ", end="", flush=True)

        try:
            stream = ollama.chat(
                model="qwen2.5:0.5b",
                messages=[{"role": "user", "content": full_prompt}],
                stream=True,
                options={
                    "temperature": 0.4, 
                    "keep_alive": "-1",
                }
            )

            for chunk in stream:
                content = chunk['message']['content']
                sys.stdout.write(content)
                sys.stdout.flush()
            print()

        except Exception as e:
            print(f"\nLỗi: {e}")

if __name__ == "__main__":
    ask_bot()