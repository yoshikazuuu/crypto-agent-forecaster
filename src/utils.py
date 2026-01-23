"""
Utility functions and classes for the crypto forecasting system.
"""

import time
import logging
from collections import deque
from typing import Optional, Dict, Any
import json
from pathlib import Path
import os
import re
import base64
from datetime import datetime

logger = logging.getLogger(__name__)


class APIRateLimiter:
    """
    Rate limiter for API calls to respect rate limits.
    
    Designed for CoinGecko API which has 30 requests/minute on Demo plan.
    """
    
    def __init__(self, max_requests_per_minute: int = 30, safety_buffer: float = 0.1):
        """
        Initialize rate limiter.
        
        Args:
            max_requests_per_minute: Maximum requests allowed per minute
            safety_buffer: Safety buffer (0.1 = 10% slower to be safe)
        """
        self.max_requests = max_requests_per_minute
        self.safety_buffer = safety_buffer
        
        # Calculate minimum delay between requests (with safety buffer)
        base_delay = 60.0 / max_requests_per_minute  # Base delay in seconds
        self.min_delay = base_delay * (1 + safety_buffer)
        
        # Track request timestamps
        self.request_times = deque()
        self.last_request_time = 0
        
        logger.info(f"Initialized rate limiter: {max_requests_per_minute} req/min, "
                   f"min delay: {self.min_delay:.2f}s")
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limits."""
        current_time = time.time()
        
        # Remove requests older than 1 minute
        cutoff_time = current_time - 60
        while self.request_times and self.request_times[0] < cutoff_time:
            self.request_times.popleft()
        
        # Check if we're at the rate limit
        if len(self.request_times) >= self.max_requests:
            # Wait until the oldest request is over a minute old
            wait_time = 60 - (current_time - self.request_times[0]) + 0.1  # Extra 0.1s buffer
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
        
        # Ensure minimum delay between requests
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_delay:
            sleep_time = self.min_delay - time_since_last
            logger.debug(f"Enforcing minimum delay, waiting {sleep_time:.2f}s...")
            time.sleep(sleep_time)
        
        # Record this request
        self.request_times.append(time.time())
        self.last_request_time = time.time()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current rate limiter status."""
        current_time = time.time()
        
        # Count recent requests
        cutoff_time = current_time - 60
        recent_requests = sum(1 for t in self.request_times if t > cutoff_time)
        
        return {
            "max_requests_per_minute": self.max_requests,
            "recent_requests": recent_requests,
            "remaining_requests": max(0, self.max_requests - recent_requests),
            "min_delay_seconds": self.min_delay,
            "time_until_reset": max(0, 60 - (current_time - (self.request_times[0] if self.request_times else current_time)))
        }


class APICache:
    """Simple cache for API responses to reduce redundant calls."""
    
    def __init__(self, cache_file: str = "api_cache.json", max_age_hours: int = 24):
        """
        Initialize API cache.
        
        Args:
            cache_file: File to store cache data
            max_age_hours: Maximum age of cached data in hours
        """
        self.cache_file = Path(cache_file)
        self.max_age_seconds = max_age_hours * 3600
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict[str, Any]:
        """Load cache from file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return {}
    
    def _save_cache(self):
        """Save cache to file."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired."""
        if key in self.cache:
            entry = self.cache[key]
            age = time.time() - entry.get('timestamp', 0)
            if age < self.max_age_seconds:
                logger.debug(f"Cache hit for {key} (age: {age/3600:.1f}h)")
                return entry.get('data')
            else:
                logger.debug(f"Cache expired for {key} (age: {age/3600:.1f}h)")
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        """Set cached value."""
        self.cache[key] = {
            'data': value,
            'timestamp': time.time()
        }
        logger.debug(f"Cached {key}")
        self._save_cache()
    
    def clear_expired(self):
        """Clear expired cache entries."""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self.cache.items():
            age = current_time - entry.get('timestamp', 0)
            if age >= self.max_age_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            logger.info(f"Cleared {len(expired_keys)} expired cache entries")
            self._save_cache()


# Legacy utility functions for compatibility
def truncate_json_for_logging(data: Any, max_length: int = None) -> str:
    """
    Convert data to JSON string without any truncation.
    
    Args:
        data: Data to be JSON serialized
        max_length: Ignored - kept for compatibility
        
    Returns:
        Full JSON string
    """
    try:
        return json.dumps(data, indent=2)
    except (TypeError, ValueError):
        return str(data)


def hide_base64_from_logs(text: str) -> str:
    """
    Replace base64 strings in text with placeholders for logging.
    
    Args:
        text: Text that may contain base64 strings
        
    Returns:
        Text with base64 strings replaced with placeholders
    """
    # Pattern for data URI base64 images
    data_uri_pattern = r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+'
    
    # Replace data URI with placeholder
    cleaned_text = re.sub(data_uri_pattern, '[BASE64_IMAGE_DATA_HIDDEN]', text)
    
    # Also handle standalone base64 strings that are long (likely images or large data)
    standalone_base64_pattern = r'[A-Za-z0-9+/]{50,}={0,2}'
    cleaned_text = re.sub(standalone_base64_pattern, '[LONG_BASE64_DATA_HIDDEN]', cleaned_text)
    
    return cleaned_text


def truncate_text_for_display(text: str, max_length: int = 500) -> str:
    """
    Truncate long text for display in logs/terminal while preserving readability.
    
    Args:
        text: Text to potentially truncate
        max_length: Maximum length before truncation (default 500 chars)
        
    Returns:
        Truncated text with indicator if shortened
    """
    if not text or len(text) <= max_length:
        return text
    
    # Truncate and add indicator
    truncated = text[:max_length].rstrip()
    remaining = len(text) - max_length
    return f"{truncated}... [+{remaining} chars truncated]"


def truncate_crew_output(output: str, max_prompt_length: int = 500, max_result_length: int = 500) -> str:
    """
    Truncate CrewAI verbose output to make workflow easier to follow.
    Truncates long prompts and results while keeping status/flow information intact.
    
    Args:
        output: The full crew verbose output
        max_prompt_length: Max chars for prompt sections
        max_result_length: Max chars for result sections
        
    Returns:
        Output with long sections truncated
    """
    lines = output.split('\n')
    result_lines = []
    in_prompt = False
    in_result = False
    current_section = []
    section_type = None
    
    for line in lines:
        # Detect start of prompt section
        if 'Prompt:' in line or 'prompt:' in line.lower():
            # Flush previous section if any
            if current_section and section_type:
                section_text = '\n'.join(current_section)
                if section_type == 'prompt':
                    result_lines.append(truncate_text_for_display(section_text, max_prompt_length))
                elif section_type == 'result':
                    result_lines.append(truncate_text_for_display(section_text, max_result_length))
                current_section = []
            
            in_prompt = True
            in_result = False
            section_type = 'prompt'
            current_section = [line]
            continue
        
        # Detect start of result section
        if 'Result:' in line or 'Output:' in line or 'result:' in line.lower():
            # Flush previous section
            if current_section and section_type:
                section_text = '\n'.join(current_section)
                if section_type == 'prompt':
                    result_lines.append(truncate_text_for_display(section_text, max_prompt_length))
                elif section_type == 'result':
                    result_lines.append(truncate_text_for_display(section_text, max_result_length))
                current_section = []
            
            in_result = True
            in_prompt = False
            section_type = 'result'
            current_section = [line]
            continue
        
        # Detect end of sections (new task, agent info, etc.)
        if any(marker in line for marker in ['Task:', 'Agent:', '===', '---', 'Starting', 'Completed', 'Executing']):
            # Flush current section
            if current_section and section_type:
                section_text = '\n'.join(current_section)
                if section_type == 'prompt':
                    result_lines.append(truncate_text_for_display(section_text, max_prompt_length))
                elif section_type == 'result':
                    result_lines.append(truncate_text_for_display(section_text, max_result_length))
                current_section = []
                section_type = None
            
            in_prompt = False
            in_result = False
            result_lines.append(line)
            continue
        
        # Accumulate content in current section
        if in_prompt or in_result:
            current_section.append(line)
        else:
            result_lines.append(line)
    
    # Flush any remaining section
    if current_section and section_type:
        section_text = '\n'.join(current_section)
        if section_type == 'prompt':
            result_lines.append(truncate_text_for_display(section_text, max_prompt_length))
        elif section_type == 'result':
            result_lines.append(truncate_text_for_display(section_text, max_result_length))
    
    return '\n'.join(result_lines)


def sanitize_for_logging(data: Any, max_json_length: int = None) -> str:
    """
    Sanitize data for logging by hiding base64 only (no truncation).
    
    Args:
        data: Data to sanitize
        max_json_length: Ignored - kept for compatibility
        
    Returns:
        Sanitized string suitable for logging
    """
    # First convert to string/JSON
    if isinstance(data, (dict, list)):
        text = truncate_json_for_logging(data)
    else:
        text = str(data)
    
    # Then hide base64 data
    return hide_base64_from_logs(text)


def create_run_directory(crypto_name: str, timestamp: datetime = None) -> Path:
    """
    Create a directory for storing results of a specific run.
    
    Args:
        crypto_name: Name of the cryptocurrency being analyzed
        timestamp: Timestamp for the run (defaults to current time)
        
    Returns:
        Path to the created directory
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    # Create directory name with timestamp
    dir_name = f"{crypto_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path("results") / dir_name
    
    # Create directory if it doesn't exist
    run_dir.mkdir(parents=True, exist_ok=True)
    
    return run_dir


def save_run_results(results: Dict[str, Any], charts: Dict[str, bytes] = None, logs: str = None, verbose: bool = False, existing_dir: Path = None) -> Path:
    """
    Save complete run results including charts and logs to a dedicated folder.
    
    Args:
        results: Dictionary containing forecast results
        charts: Dictionary mapping chart names to image bytes
        logs: String containing run logs
        verbose: Whether to include verbose information in README
        existing_dir: Optional existing directory to use instead of creating new one
        
    Returns:
        Path to the directory where results were saved
    """
    # Use existing directory or create new one
    if existing_dir:
        run_dir = Path(existing_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        # Extract crypto_name from results for use in README
        crypto_name = results.get('crypto_name', 'unknown')
    else:
        # Create run directory
        crypto_name = results.get('crypto_name', 'unknown')
        timestamp = datetime.fromisoformat(results.get('timestamp', datetime.now().isoformat()))
        run_dir = create_run_directory(crypto_name, timestamp)
    
    # Save main results
    results_file = run_dir / "forecast_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save charts if provided
    chart_paths = {}
    if charts:
        charts_dir = run_dir / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)
        
        for chart_name, chart_data in charts.items():
            chart_file = charts_dir / f"{chart_name}.png"
            if isinstance(chart_data, str):
                # Assume it's base64 encoded
                try:
                    chart_bytes = base64.b64decode(chart_data)
                    with open(chart_file, 'wb') as f:
                        f.write(chart_bytes)
                    chart_paths[chart_name] = chart_file
                except Exception as e:
                    print(f"Warning: Could not save chart {chart_name}: {e}")
            elif isinstance(chart_data, bytes):
                with open(chart_file, 'wb') as f:
                    f.write(chart_data)
                chart_paths[chart_name] = chart_file
    
    # Save logs if provided
    if logs:
        logs_file = run_dir / "run_logs.txt"
        with open(logs_file, 'w') as f:
            f.write(logs)
    
    # Create enhanced summary file
    summary_file = run_dir / "README.md"
    with open(summary_file, 'w') as f:
        # Header
        f.write(f"# 🔮 Forecast Results for {crypto_name.upper()}\n\n")
        
        # Basic information
        f.write("## 📊 Forecast Summary\n\n")
        f.write(f"- **Timestamp:** {results.get('timestamp', 'Unknown')}\n")
        f.write(f"- **Forecast Horizon:** {results.get('forecast_horizon', 'Unknown')}\n")
        f.write(f"- **Direction:** {results.get('direction', 'Unknown')}\n")
        f.write(f"- **Confidence:** {results.get('confidence', 'Unknown')}\n")
        
        # Execution summary if available
        if 'execution_summary' in results:
            summary = results['execution_summary']
            f.write(f"- **Verbose Mode:** {'✅ Enabled' if summary.get('verbose_mode') else '❌ Disabled'}\n")
            
            if verbose:
                f.write(f"\n## 🤖 Execution Details\n\n")
                f.write(f"### Agents Used\n")
                for agent in summary.get('agents_used', []):
                    f.write(f"- {agent}\n")
                
                f.write(f"\n### Tools Used\n")
                for tool in summary.get('tools_used', []):
                    f.write(f"- {tool}\n")
        
        f.write("\n")
        
        # Charts section with embedded images
        if charts:
            f.write("## 📈 Generated Charts\n\n")
            for chart_name, chart_path in chart_paths.items():
                if chart_path.exists():
                    # Create relative path for the README
                    relative_path = f"charts/{chart_name}.png"
                    f.write(f"### {chart_name.replace('_', ' ').title()}\n")
                    f.write(f"![{chart_name}]({relative_path})\n\n")
                    
                    if verbose:
                        # Add more details about the chart
                        chart_size = chart_path.stat().st_size
                        f.write(f"- **File:** `{relative_path}`\n")
                        f.write(f"- **Size:** {chart_size:,} bytes\n\n")
        
        # Forecast explanation
        if 'explanation' in results and results['explanation']:
            f.write("## 🧠 Analysis & Reasoning\n\n")
            explanation = results['explanation']
            # Clean the explanation for markdown
            explanation = hide_base64_from_logs(explanation)
            f.write(f"{explanation}\n\n")
        
        # Error information if present
        if 'error' in results:
            f.write("## ❌ Error Information\n\n")
            f.write(f"```\n{results['error']}\n```\n\n")
        
        # Files section
        f.write("## 📁 Files in this Run\n\n")
        f.write("- `forecast_results.json`: Complete forecast data in JSON format\n")
        if charts:
            f.write("- `charts/`: Directory containing generated technical analysis charts\n")
            for chart_name in charts.keys():
                f.write(f"  - `{chart_name}.png`: Technical analysis chart\n")
        if logs:
            f.write("- `run_logs.txt`: Complete execution logs\n")
        f.write("- `README.md`: This summary file\n")
        
        if verbose:
            f.write(f"\n## 🔧 Technical Information\n\n")
            f.write(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- **Crypto:** {crypto_name}\n")
            f.write(f"- **System:** CryptoAgentForecaster v1.0\n")
            f.write(f"- **Mode:** {'Verbose' if verbose else 'Standard'}\n")
    
    return run_dir


class LogCapture:
    """Context manager to capture logs during execution."""
    
    def __init__(self):
        self.logs = []
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def log(self, message: str):
        """Add a log message."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # Only hide base64 data, don't truncate anything else
        clean_message = hide_base64_from_logs(str(message))
        log_entry = f"[{timestamp}] {clean_message}"
        self.logs.append(log_entry)
    
    def get_logs(self) -> str:
        """Get all captured logs as a single string."""
        return "\n".join(self.logs) 