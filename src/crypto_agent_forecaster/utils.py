"""
Utility functions for the CryptoAgentForecaster system.
"""

import json
import os
import re
import base64
from typing import Any, Dict
from datetime import datetime
from pathlib import Path


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
        charts_dir.mkdir(exist_ok=True)
        
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