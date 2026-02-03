"""
Validation utilities for the CLI application.
"""

import logging
from typing import Optional, Dict, Any, List, Tuple
from rich.prompt import Confirm, Prompt

from .constants import ERROR_MESSAGES, SUCCESS_MESSAGES
from .output import OutputManager
from ..config import Config
from ..llm_factory import LLMFactory

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class Validator:
    """Handles all validation logic for the CLI application."""
    
    def __init__(self, output_manager: OutputManager):
        self.output = output_manager
    
    def validate_configuration(self) -> bool:
        """
        Validate system configuration.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Check if at least one LLM provider is configured
            providers = LLMFactory.get_available_providers()
            if not providers:
                self.output.display_error(ERROR_MESSAGES["no_config"])
                self.output.print(ERROR_MESSAGES["config_create_hint"])
                return False
            
            # Validate LLM configuration
            LLMFactory.validate_configuration()
            logger.info(f"Configuration validated: {len(providers)} providers available")
            return True
            
        except Exception as e:
            self.output.display_error(f"Configuration validation failed: {str(e)}")
            logger.error(f"Configuration validation error: {str(e)}")
            return False
    
    def validate_provider(self, provider: Optional[str]) -> bool:
        """
        Validate LLM provider.
        
        Args:
            provider: Provider name to validate
            
        Returns:
            bool: True if provider is valid, False otherwise
        """
        if not provider:
            return True  # No provider specified is OK (use default)
        
        available_providers = LLMFactory.get_available_providers()
        if provider not in available_providers:
            self.output.display_error(ERROR_MESSAGES["provider_unavailable"].format(provider))
            return False
        
        logger.info(f"Provider {provider} validated successfully")
        return True
    
    def validate_crypto_name(self, crypto_name: str) -> bool:
        """
        Validate cryptocurrency name.
        
        Args:
            crypto_name: Cryptocurrency name to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not crypto_name or not crypto_name.strip():
            self.output.display_error("Cryptocurrency name cannot be empty")
            return False
        
        # Basic validation - cryptocurrency names should be alphanumeric with hyphens
        if not all(c.isalnum() or c in '-_' for c in crypto_name):
            self.output.display_error("Invalid cryptocurrency name format")
            return False
        
        logger.info(f"Cryptocurrency name {crypto_name} validated")
        return True
    
    def validate_forecast_horizon(self, horizon: str) -> bool:
        """
        Validate forecast horizon format.
        
        Args:
            horizon: Time horizon string to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not horizon or not horizon.strip():
            self.output.display_error("Forecast horizon cannot be empty")
            return False
        
        # Basic validation - should contain numbers and time units
        valid_units = ['hour', 'hours', 'day', 'days', 'week', 'weeks', 'month', 'months']
        horizon_lower = horizon.lower()
        
        has_number = any(c.isdigit() for c in horizon)
        has_valid_unit = any(unit in horizon_lower for unit in valid_units)
        
        if not (has_number and has_valid_unit):
            self.output.display_error("Invalid forecast horizon format. Use formats like '24 hours', '3 days', '1 week'")
            return False
        
        logger.info(f"Forecast horizon {horizon} validated")
        return True
    
    def validate_model_name(self, provider: str, model: Optional[str]) -> bool:
        """
        Validate model name for the given provider.
        
        Args:
            provider: LLM provider name
            model: Model name to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not model:
            return True  # No model specified is OK (use default)
        
        model_specs = LLMFactory.MODEL_SPECS.get(provider, {})
        if model not in model_specs:
            self.output.display_warning(f"Model {model} not in known specifications for {provider}")
            # Don't fail validation - unknown models might still work
        
        logger.info(f"Model {model} for provider {provider} validated")
        return True
    
    def get_user_confirmation(self, message: str, default: bool = True) -> bool:
        """
        Get user confirmation for an action.
        
        Args:
            message: Confirmation message to display
            default: Default response if user just presses Enter
            
        Returns:
            bool: True if user confirms, False otherwise
        """
        try:
            return Confirm.ask(message, default=default)
        except KeyboardInterrupt:
            self.output.print("\nOperation cancelled by user.")
            return False
    
    def prompt_model_selection(self) -> Tuple[str, str]:
        """
        Prompt user to select a model from the default choices.
        
        Returns:
            Tuple of (provider, model) selected by user
        """
        choices = LLMFactory.DEFAULT_MODEL_CHOICES
        
        self.output.print("\nSelect a model to use:", style="bold")
        self.output.print("")
        
        for i, choice in enumerate(choices, 1):
            provider = choice["provider"]
            model = choice["model"]
            description = choice["description"]
            self.output.print(f"  [{i}] {model} ({provider}) - {description}")
        
        self.output.print("")
        
        try:
            selection = Prompt.ask(
                "Enter choice",
                choices=[str(i) for i in range(1, len(choices) + 1)],
                default="1"
            )
            
            selected_choice = choices[int(selection) - 1]
            provider = selected_choice["provider"]
            model = selected_choice["model"]
            
            self.output.print(f"\nSelected: {model} ({provider})", style="bold green")
            logger.info(f"User selected model: {provider}/{model}")
            
            return provider, model
            
        except KeyboardInterrupt:
            self.output.print("\nOperation cancelled by user.")
            # Default to first choice
            return choices[0]["provider"], choices[0]["model"]
    
    def validate_debug_environment(self) -> Dict[str, Any]:
        """
        Validate environment for debugging purposes.
        
        Returns:
            Dict containing debug information
        """
        try:
            debug_info = Config.debug_environment()
            logger.info("Environment debug information collected")
            return debug_info
        except Exception as e:
            logger.error(f"Environment debug failed: {str(e)}")
            return {
                "error": str(e),
                "environment_loaded": False,
                "api_keys_configured": {},
                "env_vars_present": {},
                "default_provider": None,
                "default_model": None
            }
    
    def validate_agent_configurations(self) -> List[Tuple[str, bool]]:
        """
        Validate configuration for all agents.
        
        Returns:
            List of tuples containing (agent_type, is_valid)
        """
        from .constants import AGENT_TYPES
        
        results = []
        for agent_type in AGENT_TYPES:
            try:
                is_valid = Config.validate_llm_config(agent_type)
                results.append((agent_type, is_valid))
                logger.debug(f"Agent {agent_type} validation: {is_valid}")
            except Exception as e:
                logger.error(f"Agent {agent_type} validation error: {str(e)}")
                results.append((agent_type, False))
        
        return results
    
    def validate_test_parameters(self, crypto: str, quick: bool) -> Tuple[str, int]:
        """
        Validate and normalize test parameters.
        
        Args:
            crypto: Cryptocurrency name for testing
            quick: Whether to run quick test
            
        Returns:
            Tuple of (validated_crypto, days_for_test)
        """
        from .constants import DEFAULT_CRYPTO, DEFAULT_QUICK_TEST_DAYS, DEFAULT_FULL_TEST_DAYS
        
        # Validate crypto name
        if not self.validate_crypto_name(crypto):
            self.output.display_warning(f"Invalid crypto name '{crypto}', using default: {DEFAULT_CRYPTO}")
            crypto = DEFAULT_CRYPTO
        
        # Determine test duration
        days = DEFAULT_QUICK_TEST_DAYS if quick else DEFAULT_FULL_TEST_DAYS
        
        logger.info(f"Test parameters validated: crypto={crypto}, days={days}")
        return crypto, days
    
    def check_critical_dependencies(self) -> bool:
        """
        Check for critical system dependencies.
        
        Returns:
            bool: True if all critical dependencies are available
        """
        try:
            # Check for required modules
            import crewai
            import pandas
            import matplotlib
            
            logger.info("Critical dependencies check passed")
            return True
            
        except ImportError as e:
            self.output.display_error(f"Critical dependency missing: {str(e)}")
            logger.error(f"Dependency check failed: {str(e)}")
            return False
    
    def validate_forecast_inputs(self, crypto: str, horizon: str, provider: Optional[str], 
                                model: Optional[str]) -> bool:
        """
        Validate all inputs for a forecast operation.
        
        Args:
            crypto: Cryptocurrency name
            horizon: Forecast time horizon
            provider: LLM provider (optional)
            model: Model name (optional)
            
        Returns:
            bool: True if all inputs are valid
        """
        validations = [
            self.validate_crypto_name(crypto),
            self.validate_forecast_horizon(horizon),
            self.validate_provider(provider)
        ]
        
        if provider and model:
            validations.append(self.validate_model_name(provider, model))
        
        is_valid = all(validations)
        
        if is_valid:
            logger.info(f"Forecast inputs validated: {crypto}, {horizon}, {provider}, {model}")
        else:
            logger.warning(f"Forecast input validation failed")
        
        return is_valid 