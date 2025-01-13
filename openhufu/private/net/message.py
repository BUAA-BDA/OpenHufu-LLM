from typing import Dict, Any


class Message(Dict[str, Any]):
    def __init__(self, headers: Dict[str, Any], data: Any):
        super().__init__()
        self["headers"] = headers
        self["data"] = data
        
    
    @classmethod
    def from_dict(cls, message_dict: Dict[str, Any]) -> 'Message':
        headers = message_dict.get('headers', {})
        data = message_dict.get('data', None)
        return cls(headers, data)
    
    
    def get_headers(self) -> Dict[str, any]:
        return self["headers"]
    
    
    def get_data(self) -> any:
        return self["data"]
    
    
    def get_from_headers(self, key: str) -> any:
        return self["headers"].get(key) 
    
    
    def set_header(self, key: str, value: any):
        self["headers"][key] = value
    
    
    def set_headers(self, headers: Dict[str, any]):
        self["headers"] = headers
    
    
    def update_headers(self, headers: Dict[str, any]):
        self["headers"].update(headers)
    
    
    def set_data(self, data: any):
        self["data"] = data
    
    
    def __getitem__(self, key):
        return super().__getitem__(key)
    
    
    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        
    def __str__(self):
        return f"Message({self.get_headers()}, {self.get_data()})"
    
    
    
        