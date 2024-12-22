import json
from typing import Any, Dict
from datetime import datetime

def format_response(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'data': data,
        'timestamp': datetime.now().isoformat(),
        'status': 'success'
    }

def format_error(message: str, code: int = 500) -> Dict[str, Any]:
    return {
        'error': message,
        'code': code,
        'timestamp': datetime.now().isoformat(),
        'status': 'error'
    }