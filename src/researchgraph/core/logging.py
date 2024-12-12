"""Logging implementation for research graph nodes."""
import os
import json
import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .types import NodeState, NodeResult, NodeInput, NodeOutput


class NodeLogger:
    """Logger for tracking node execution progress."""

    def __init__(self, log_dir: str):
        """Initialize NodeLogger.

        Args:
            log_dir: Directory to store log files
        """
        self.log_dir = log_dir
        self.logger = logging.getLogger(self.__class__.__name__)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            self.logger.info(f"Created log directory: {log_dir}")

    def _get_log_path(self, node_name: str) -> str:
        """Get log file path for a node.

        Args:
            node_name: Name of the node

        Returns:
            Path to log file
        """
        return os.path.join(self.log_dir, f"{node_name}.json")

    def log_start(
        self,
        node_name: str,
        input_state: Dict[str, NodeInput],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log node execution start.

        Args:
            node_name: Name of the node
            input_state: Input state dictionary
            metadata: Optional metadata to log
        """
        log_entry = {
            "node": node_name,
            "status": "started",
            "timestamp": datetime.now().isoformat(),
            "input_state": input_state,
            "metadata": metadata or {}
        }

        try:
            log_path = self._get_log_path(node_name)
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(log_entry, f, indent=2)
            self.logger.debug(f"Logged start for node: {node_name}")
        except Exception as e:
            self.logger.error(f"Failed to log start for node {node_name}: {str(e)}")

    def log_complete(
        self,
        node_name: str,
        output_state: Dict[str, NodeOutput],
        execution_time: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> NodeResult:
        """Log node execution completion.

        Args:
            node_name: Name of the node
            output_state: Output state dictionary
            execution_time: Execution time in seconds
            metadata: Optional metadata to log

        Returns:
            NodeResult containing execution details
        """
        log_entry = {
            "node": node_name,
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "output_state": output_state,
            "execution_time": execution_time,
            "metadata": metadata or {}
        }

        try:
            log_path = self._get_log_path(node_name)
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(log_entry, f, indent=2)
            self.logger.debug(f"Logged completion for node: {node_name}")
        except Exception as e:
            self.logger.error(f"Failed to log completion for node {node_name}: {str(e)}")

    def log_error(
        self,
        node_name: str,
        error: Exception,
        state: NodeState,
        metadata: Optional[Dict[str, Any]] = None
    ) -> NodeResult:
        """Log node execution error.

        Args:
            node_name: Name of the node
            error: Exception that occurred
            state: Current state dictionary
            metadata: Optional metadata to log

        Returns:
            NodeResult containing error details
        """
        log_entry = {
            "node": node_name,
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": {
                "type": type(error).__name__,
                "message": str(error)
            },
            "state": state,
            "metadata": metadata or {}
        }

        try:
            log_path = self._get_log_path(node_name)
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(log_entry, f, indent=2)
            self.logger.debug(f"Logged error for node: {node_name}")
        except Exception as e:
            self.logger.error(f"Failed to log error for node {node_name}: {str(e)}")
