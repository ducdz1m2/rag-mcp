import ollama
from database import load_db
import time
import numpy as np
from functools import lru_cache
import json
import asyncio
from typing import Dict, List, Any, Optional

class MCPDispatcher:
    def __init__(self):
        self.tools = {
            "rag_search": {
                "keywords": ["về", "là gì", "tài liệu", "thông tin", "quy định", "máy móc", "hướng dẫn", "ct", "đại học", "logo", "thương hiệu", "e-newsletter", "newsletter", "brand"],
                "description": "Truy xuất kiến thức từ database nội bộ",
                "handler": self._handle_rag_search
            },
            "sensor_read": {
                "keywords": ["đọc sensor", "đọc dữ liệu", "sensor", "nhiệt độ", "độ ẩm", "ánh sáng", "đọc", "nhiệt", "ẩm", "sáng"],
                "description": "Đọc dữ liệu từ các cảm biến",
                "handler": self._handle_sensor_read
            },
            "device_control": {
                "keywords": ["bật", "tắt", "điều khiển", "mở", "đóng", "thiết bị", "quạt", "đèn", "relay"],
                "description": "Điều khiển các thiết bị",
                "handler": self._handle_device_control
            },
            "general_chat": {
                "keywords": ["chào", "hi", "hello", "tạm biệt", "cảm ơn", "bạn là ai", "bạn tên", "ai"],
                "description": "Tán gẫu hoặc chào hỏi",
                "handler": self._handle_general_chat
            }
        }
        
        # MCP server registry cho mở rộng
        self.mcp_servers = {}
        self.register_builtin_servers()

    def register_builtin_servers(self):
        """Đăng ký các MCP server builtin"""
        # Placeholder cho các MCP server tương lai
        pass
    
    def register_mcp_server(self, name: str, server_config: Dict[str, Any]):
        """Đăng ký MCP server mới"""
        self.mcp_servers[name] = server_config
        print(f"✅ Đã đăng ký MCP server: {name}")
    
    def route_to_mcp_server(self, tool_name: str, query: str) -> Optional[str]:
        """Route request đến MCP server tương ứng"""
        if tool_name in self.mcp_servers:
            # Gọi MCP server (mock implementation)
            return f"[MCP Response from {tool_name}] Đã xử lý: {query}"
        return None
    
    def _handle_rag_search(self, query: str, retriever) -> str:
        """Handler cho RAG search với hybrid search (vector + keyword)"""
        if not retriever:
            return "Xin lỗi, database tìm kiếm chưa được tải. Vui lòng kiểm tra lại file FAISS index."
        
        try:
            query_lower = query.lower()
            relevant_docs = []
            
            # First try vector search
            docs = retriever.invoke(query)
            
            for d in docs:
                content = d.page_content.strip().lower()
                relevance_score = 0
                
                # Exact phrase matching
                if query_lower in content:
                    relevance_score += 10
                
                # Word matching
                query_words = query_lower.split()
                for word in query_words:
                    if len(word) > 2 and word in content:
                        relevance_score += 2
                
                if relevance_score >= 2:
                    relevant_docs.append((d, relevance_score))
            
            # If no good results, do keyword search across more documents
            if not relevant_docs:
                # Get more documents and filter by keywords
                all_docs = retriever.vectorstore.similarity_search('đại học', k=50)
                
                for d in all_docs:
                    content = d.page_content.strip().lower()
                    relevance_score = 0
                    
                    # More lenient keyword matching
                    query_words = query_lower.split()
                    for word in query_words:
                        if len(word) > 2 and word in content:
                            relevance_score += 1
                    
                    if relevance_score >= 1:
                        relevant_docs.append((d, relevance_score))
            
            # Sort by relevance score
            relevant_docs.sort(key=lambda x: x[1], reverse=True)
            
            if relevant_docs:
                context_parts = []
                for doc, score in relevant_docs[:3]:
                    content = doc.page_content.strip()
                    if len(content) > 50:
                        context_parts.append(content[:500])
                
                context_text = "\n\n".join(context_parts)
                
                return f"Dựa trên thông tin trong database, đây là câu trả lời cho câu hỏi '{query}':\n\n{context_text}"
            else:
                return f"Xin lỗi, tôi không tìm thấy thông tin liên quan đến '{query}' trong database. Database hiện có thông tin về Đại học Cần Thơ, thương hiệu, logo, và các tài liệu liên quan. Bạn có thể thử hỏi về các chủ đề này."
                
        except Exception as e:
            return f"Xin lỗi, có lỗi xảy ra khi tìm kiếm thông tin: {str(e)}"
    
    def _handle_general_chat(self, query: str) -> str:
        """Handler cho general chat"""
        query_lower = query.lower()
        
        if any(greeting in query_lower for greeting in ['chào', 'hello', 'hi']):
            return f"Chào bạn! Tôi là AI assistant của DTHub. Tôi có thể giúp bạn tìm kiếm thông tin, đọc sensor data, hoặc điều khiển thiết bị. Bạn cần giúp gì không?"
        
        if any(who in query_lower for who in ['bạn là ai', 'who are you', 'bạn là gì']):
            return "Tôi là AI assistant được tích hợp vào hệ thống DTHub. Tôi có khả năng xử lý các yêu cầu về tìm kiếm tài liệu, đọc dữ liệu cảm biến, và điều khiển thiết bị IoT thông qua RAG-MCP system."
        
        if any(thanks in query_lower for thanks in ['cảm ơn', 'thank', 'tks']):
            return "Rất vui được giúp đỡ bạn! Nếu có câu hỏi nào khác, đừng ngần ngại hỏi nhé."
        
        if any(bye in query_lower for bye in ['tạm biệt', 'bye', 'goodbye']):
            return "Tạm biệt! Hẹn gặp lại bạn sớm."
        
        # Default response
        return f"Tôi hiểu bạn nói: '{query}'. Tôi có thể giúp bạn tìm kiếm thông tin, đọc sensor, hoặc điều khiển thiết bị. Bạn muốn làm gì cụ thể?"
    
    def _handle_sensor_read(self, query: str) -> str:
        """Handler cho sensor reading (placeholder)"""
        # Extract sensor type from query
        query_lower = query.lower()
        
        if 'nhiệt độ' in query_lower:
            return f"🌡️ Nhiệt độ hiện tại: 25.5°C. Đọc từ sensor DHT22 trong phòng khách."
        elif 'độ ẩm' in query_lower:
            return f"💧 Độ ẩm hiện tại: 60%. Đọc từ sensor DHT22 trong phòng khách."
        elif 'ánh sáng' in query_lower:
            return f"💡 Cường độ ánh sáng: 800 lux. Đọc từ photoresistor gần cửa sổ."
        elif 'soil' in query_lower or 'đất' in query_lower:
            return f"🌱 Độ ẩm đất: 45%. Đọc từ soil moisture sensor trong chậu cây."
        else:
            # Default sensor reading
            return f"📊 Dữ liệu sensors hiện tại:\n- Nhiệt độ: 25.5°C\n- Độ ẩm: 60%\n- Ánh sáng: 800 lux\n- Độ ẩm đất: 45%\n\nThời gian đọc: {time.strftime('%H:%M:%S')}"
    
    def _handle_device_control(self, query: str) -> str:
        """Handler cho device control (placeholder)"""
        # Mock response cho device control
        return f"Điều khiển thiết bị: {query}. Đã thực hiện thành công."
    
    def smart_route(self, query: str) -> tuple[str, str]:
        """Smart routing với confidence scoring"""
        query_lower = query.lower()
        scores = {}
        
        for tool_name, config in self.tools.items():
            score = 0
            for keyword in config["keywords"]:
                if keyword in query_lower:
                    score += 1
            
            # Bonus cho câu hỏi dài (RAG)
            if tool_name == "rag_search" and len(query_lower.split()) > 6:
                score += 2
                
            scores[tool_name] = score
        
        # Chọn tool có score cao nhất
        best_tool = max(scores, key=scores.get)
        confidence = scores[best_tool] / max(scores.values()) if max(scores.values()) > 0 else 0
        
        return best_tool, confidence

# --- KHỞI TẠO HỆ THỐNG ---
print("Khởi tạo hệ thống MCP...")
dispatcher = MCPDispatcher()

try:
    vectorstore = load_db()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3, "fetch_k": 8})  # Tăng lại k lên 3
except:
    retriever = None

def ask_bot():
    while True:
        user_query = input("\n👉 Câu hỏi: ").strip()
        if not user_query: continue
        if user_query.lower() in ['exit', 'quit']: break

        t_start = time.perf_counter()

        
        selected_tool, confidence = dispatcher.smart_route(user_query)
        
        # Xử lý với handler tương ứng
        if selected_tool in dispatcher.tools:
            handler = dispatcher.tools[selected_tool]["handler"]
            
            if selected_tool == "rag_search":
                prompt = handler(user_query, retriever)
                t_search = 0  # Sẽ được tính trong handler
            else:
                prompt = handler(user_query)
                t_search = 0
                
            # Thử route đến MCP server nếu có
            mcp_response = dispatcher.route_to_mcp_server(selected_tool, user_query)
            if mcp_response:
                prompt = f"{prompt}\n\nAdditional MCP Response: {mcp_response}"
        else:
            prompt = user_query
            t_search = 0

        t_prep = time.perf_counter() - t_start

        print(f"Bot ({selected_tool} | conf: {confidence:.2f} | prep: {t_prep:.3f}s): ", end="", flush=True)

        try:
            stream = ollama.chat(
                model="qwen2.5:1.5b",
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                options={
                    "temperature": 0.1,
                    "num_predict": 250  # Tăng lên 250 để trả lời chi tiết hơn
                }
            )

            for chunk in stream:
                print(chunk['message']['content'], end="", flush=True)
            print()

        except Exception as e:
            print(f"\nLỗi: {e}")

if __name__ == "__main__":
    ask_bot()