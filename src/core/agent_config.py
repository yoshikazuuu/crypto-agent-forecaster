"""
Agent-specific configuration management.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class AgentConfig:
    """Configuration for LLM agents."""
    
    # LLM-specific configurations for different agents
    LLM_AGENT_CONFIGS: Dict[str, Dict[str, Any]] = {
        "market_data": {
            "temperature": 0.0, 
            "max_tokens": 2000,
            "preferred_provider": "openai",
            "preferred_model": "gpt-4o-mini"
        },
        "sentiment": {
            "temperature": 0.05, 
            "max_tokens": 3000,
            "preferred_provider": "openai", 
            "preferred_model": "gpt-4o"
        },
        "technical": {
            "temperature": 0.0, 
            "max_tokens": 2500,
            "preferred_provider": "openai",
            "preferred_model": "gpt-4o"
        },
        "forecasting": {
            "temperature": 0.05, 
            "max_tokens": 4000,
            "preferred_provider": "google",
            "preferred_model": "gemini-1.5-pro"
        }
    }
    
    # CrewAI specific settings
    CREW_SETTINGS: Dict[str, Any] = {
        "verbose": True,
        "memory": False,
        "cache": True,
        "max_iter": 3,
        "max_execution_time": 300,  # 5 minutes max per forecast
    }
    
    @classmethod
    def get_agent_llm_config(cls, agent_type: str) -> Dict[str, Any]:
        """
        Get LLM configuration for a specific agent type.
        
        Args:
            agent_type: Type of agent (market_data, sentiment, technical, forecasting)
            
        Returns:
            Dict containing LLM configuration for the agent
        """
        if agent_type not in cls.LLM_AGENT_CONFIGS:
            logger.warning(f"Unknown agent type: {agent_type}, using default config")
            return cls.LLM_AGENT_CONFIGS["market_data"]
        
        config = cls.LLM_AGENT_CONFIGS[agent_type].copy()
        logger.debug(f"Retrieved config for {agent_type}: {config}")
        return config
    
    @classmethod
    def validate_llm_config(cls, agent_type: str) -> bool:
        """
        Validate LLM configuration for a specific agent.
        
        Args:
            agent_type: Type of agent to validate
            
        Returns:
            bool: True if configuration is valid
        """
        try:
            config = cls.get_agent_llm_config(agent_type)
            
            # Check required fields
            required_fields = ["temperature", "max_tokens", "preferred_provider", "preferred_model"]
            for field in required_fields:
                if field not in config:
                    logger.error(f"Missing required field '{field}' for agent {agent_type}")
                    return False
            
            # Validate temperature range
            temp = config["temperature"]
            if not (0.0 <= temp <= 2.0):
                logger.error(f"Invalid temperature {temp} for agent {agent_type}")
                return False
            
            # Validate max_tokens
            max_tokens = config["max_tokens"]
            if not isinstance(max_tokens, int) or max_tokens <= 0:
                logger.error(f"Invalid max_tokens {max_tokens} for agent {agent_type}")
                return False
            
            logger.debug(f"Agent {agent_type} configuration validated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to validate config for agent {agent_type}: {str(e)}")
            return False
    
    @classmethod
    def get_all_agent_types(cls) -> list:
        """Get list of all configured agent types."""
        return list(cls.LLM_AGENT_CONFIGS.keys())
    
    @classmethod
    def update_agent_config(cls, agent_type: str, updates: Dict[str, Any]) -> bool:
        """
        Update configuration for a specific agent.
        
        Args:
            agent_type: Type of agent to update
            updates: Dictionary of configuration updates
            
        Returns:
            bool: True if update was successful
        """
        try:
            if agent_type not in cls.LLM_AGENT_CONFIGS:
                logger.error(f"Cannot update unknown agent type: {agent_type}")
                return False
            
            cls.LLM_AGENT_CONFIGS[agent_type].update(updates)
            logger.info(f"Updated configuration for agent {agent_type}: {updates}")
            
            # Validate the updated configuration
            return cls.validate_llm_config(agent_type)
            
        except Exception as e:
            logger.error(f"Failed to update config for agent {agent_type}: {str(e)}")
            return False 