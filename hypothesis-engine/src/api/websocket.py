"""
WebSocket Handler

Real-time communication for streaming hypothesis generation progress.
"""

from typing import List, Dict, Callable
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import json
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class WSMessage:
    """WebSocket message structure"""
    type: str  # 'progress', 'result', 'error', 'status'
    data: Dict
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class WebSocketManager:
    """Manages WebSocket connections and broadcasts"""
    
    def __init__(self):
        self.active_connections: List = []
        self.message_handlers: Dict[str, Callable] = {}
    
    async def connect(self, websocket):
        """Accept and store new connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Active: {len(self.active_connections)}")
    
    def disconnect(self, websocket):
        """Remove connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket disconnected. Active: {len(self.active_connections)}")
    
    async def broadcast(self, message: WSMessage):
        """Send message to all connections"""
        msg_json = json.dumps({
            'type': message.type,
            'data': message.data,
            'timestamp': message.timestamp
        })
        
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(msg_json)
            except Exception as e:
                logger.warning(f"Failed to send to connection: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected
        for conn in disconnected:
            self.disconnect(conn)
    
    async def send_progress(self, step: str, progress: float, details: str = ""):
        """Send progress update"""
        await self.broadcast(WSMessage(
            type='progress',
            data={
                'step': step,
                'progress': progress,
                'details': details
            }
        ))
    
    async def send_result(self, result: Dict):
        """Send final result"""
        await self.broadcast(WSMessage(
            type='result',
            data=result
        ))
    
    async def send_error(self, error: str, details: Dict = None):
        """Send error message"""
        await self.broadcast(WSMessage(
            type='error',
            data={
                'error': error,
                'details': details or {}
            }
        ))


class ProgressTracker:
    """Tracks and reports progress for long-running operations"""
    
    def __init__(self, ws_manager: WebSocketManager = None, total_steps: int = 5):
        self.ws_manager = ws_manager
        self.total_steps = total_steps
        self.current_step = 0
        self.step_names = [
            "Analyzing research question",
            "Searching primary domain",
            "Discovering cross-domain connections",
            "Generating hypotheses",
            "Validating and ranking"
        ]
    
    async def update(self, step_index: int, details: str = ""):
        """Update progress"""
        self.current_step = step_index
        progress = (step_index + 1) / self.total_steps
        step_name = self.step_names[step_index] if step_index < len(self.step_names) else f"Step {step_index + 1}"
        
        logger.info(f"Progress: {step_name} ({progress:.0%})")
        
        if self.ws_manager:
            await self.ws_manager.send_progress(step_name, progress, details)
    
    async def complete(self, result: Dict):
        """Mark operation complete"""
        logger.info("Operation complete")
        
        if self.ws_manager:
            await self.ws_manager.send_result(result)
    
    async def error(self, error: str):
        """Report error"""
        logger.error(f"Operation error: {error}")
        
        if self.ws_manager:
            await self.ws_manager.send_error(error)


# Global WebSocket manager
_ws_manager = WebSocketManager()


def get_ws_manager() -> WebSocketManager:
    return _ws_manager
