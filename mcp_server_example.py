"""
Ví dụ về MCP Server cho Sensor Reading và Device Control
Đây là template để bạn phát triển MCP server thực tế
"""

import asyncio
import json
from typing import Dict, Any, Optional
import random
import time

class MCPServer:
    """Base class cho MCP Server"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.tools = {}
        
    async def handle_request(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP request"""
        if tool_name not in self.tools:
            return {"error": f"Tool {tool_name} not found"}
            
        tool_handler = self.tools[tool_name]
        try:
            result = await tool_handler(params)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

class SensorMCPServer(MCPServer):
    """MCP Server cho Sensor Reading"""
    
    def __init__(self):
        super().__init__("sensor_server", "Server for reading sensor data")
        self.tools = {
            "read_temperature": self._read_temperature,
            "read_humidity": self._read_humidity,
            "read_light": self._read_light,
            "read_all_sensors": self._read_all_sensors
        }
    
    async def _read_temperature(self, params: Dict[str, Any]) -> str:
        """Đọc nhiệt độ (mock)"""
        # Mock: Giả lập đọc sensor DHT22
        temp = round(random.uniform(20, 35), 1)
        return f"Nhiệt độ hiện tại: {temp}°C"
    
    async def _read_humidity(self, params: Dict[str, Any]) -> str:
        """Đọc độ ẩm (mock)"""
        # Mock: Giả lập đọc sensor DHT22
        humidity = round(random.uniform(40, 80), 1)
        return f"Độ ẩm hiện tại: {humidity}%"
    
    async def _read_light(self, params: Dict[str, Any]) -> str:
        """Đọc ánh sáng (mock)"""
        # Mock: Giả lập đọc photoresistor
        light = random.randint(100, 1000)
        return f"Cường độ ánh sáng: {light} lux"
    
    async def _read_all_sensors(self, params: Dict[str, Any]) -> str:
        """Đọc tất cả sensors"""
        temp = round(random.uniform(20, 35), 1)
        humidity = round(random.uniform(40, 80), 1)
        light = random.randint(100, 1000)
        
        return f"""Dữ liệu sensors:
- Nhiệt độ: {temp}°C
- Độ ẩm: {humidity}%
- Ánh sáng: {light} lux
- Thời gian: {time.strftime('%H:%M:%S')}"""

class DeviceMCPServer(MCPServer):
    """MCP Server cho Device Control"""
    
    def __init__(self):
        super().__init__("device_server", "Server for controlling devices")
        self.devices = {
            "led": {"state": "off", "pin": 18},
            "fan": {"state": "off", "pin": 19},
            "pump": {"state": "off", "pin": 20}
        }
        self.tools = {
            "turn_on": self._turn_on,
            "turn_off": self._turn_off,
            "toggle": self._toggle,
            "get_status": self._get_status
        }
    
    async def _turn_on(self, params: Dict[str, Any]) -> str:
        """Bật thiết bị"""
        device = params.get("device")
        if device not in self.devices:
            return f"Thiết bị {device} không tồn tại"
        
        self.devices[device]["state"] = "on"
        return f"Đã bật {device} (PIN {self.devices[device]['pin']})"
    
    async def _turn_off(self, params: Dict[str, Any]) -> str:
        """Tắt thiết bị"""
        device = params.get("device")
        if device not in self.devices:
            return f"Thiết bị {device} không tồn tại"
        
        self.devices[device]["state"] = "off"
        return f"Đã tắt {device} (PIN {self.devices[device]['pin']})"
    
    async def _toggle(self, params: Dict[str, Any]) -> str:
        """Chuyển đổi trạng thái thiết bị"""
        device = params.get("device")
        if device not in self.devices:
            return f"Thiết bị {device} không tồn tại"
        
        current_state = self.devices[device]["state"]
        new_state = "on" if current_state == "off" else "off"
        self.devices[device]["state"] = new_state
        
        action = "bật" if new_state == "on" else "tắt"
        return f"Đã {action} {device} (PIN {self.devices[device]['pin']})"
    
    async def _get_status(self, params: Dict[str, Any]) -> str:
        """Lấy trạng thái tất cả thiết bị"""
        status_list = []
        for device, info in self.devices.items():
            state_emoji = "🟢" if info["state"] == "on" else "🔴"
            status_list.append(f"{state_emoji} {device}: {info['state']} (PIN {info['pin']})")
        
        return "Trạng thái thiết bị:\n" + "\n".join(status_list)

# Example usage
async def main():
    # Tạo servers
    sensor_server = SensorMCPServer()
    device_server = DeviceMCPServer()
    
    # Test sensor server
    print("=== Test Sensor Server ===")
    temp_result = await sensor_server.handle_request("read_temperature", {})
    print(temp_result)
    
    humidity_result = await sensor_server.handle_request("read_humidity", {})
    print(humidity_result)
    
    all_sensors_result = await sensor_server.handle_request("read_all_sensors", {})
    print(all_sensors_result)
    
    print("\n=== Test Device Server ===")
    # Bật LED
    led_on = await device_server.handle_request("turn_on", {"device": "led"})
    print(led_on)
    
    # Bật quạt
    fan_on = await device_server.handle_request("turn_on", {"device": "fan"})
    print(fan_on)
    
    # Xem trạng thái
    status = await device_server.handle_request("get_status", {})
    print(status)
    
    # Tắt LED
    led_off = await device_server.handle_request("turn_off", {"device": "led"})
    print(led_off)

if __name__ == "__main__":
    asyncio.run(main())
