#!/usr/bin/env python3
"""
VPS Deployment and Automation Script

Automates the deployment and continuous running of the crypto forecasting validation
on a VPS server. Includes monitoring, error recovery, and automated reporting.
"""

import os
import sys
import time
import logging
import asyncio
import schedule
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path
import subprocess
import json
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import configparser
from dataclasses import dataclass

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from validation.validator import CryptoValidator
from validation.analytics import ValidationAnalytics


@dataclass
class DeploymentConfig:
    """Configuration for VPS deployment"""
    validation_interval_hours: int = 1
    report_interval_hours: int = 24
    max_memory_usage_mb: int = 2048
    max_cpu_usage_percent: float = 80.0
    email_notifications: bool = True
    email_smtp_server: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    email_recipients: List[str] = None
    test_coins: List[str] = None
    log_retention_days: int = 30
    auto_restart_on_error: bool = True
    max_restart_attempts: int = 3


class VPSDeploymentManager:
    """Manages continuous validation deployment on VPS"""
    
    def __init__(self, config_file: str = "vps_config.ini"):
        self.config_file = Path(config_file)
        self.config = self._load_config()
        
        # Setup directories
        self.deployment_dir = Path("vps_deployment")
        self.deployment_dir.mkdir(exist_ok=True)
        
        self.logs_dir = self.deployment_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize components
        self.validator = CryptoValidator(str(self.deployment_dir / "validation_results"))
        self.analytics = ValidationAnalytics(str(self.deployment_dir / "validation_results"))
        
        # Runtime tracking
        self.start_time = datetime.now()
        self.restart_attempts = 0
        self.last_successful_run = None
        
        self.logger.info("VPS Deployment Manager initialized")
    
    def _load_config(self) -> DeploymentConfig:
        """Load configuration from file or create default"""
        config = configparser.ConfigParser()
        
        if self.config_file.exists():
            config.read(self.config_file)
            
            return DeploymentConfig(
                validation_interval_hours=config.getint('validation', 'interval_hours', fallback=1),
                report_interval_hours=config.getint('reporting', 'interval_hours', fallback=24),
                max_memory_usage_mb=config.getint('resources', 'max_memory_mb', fallback=2048),
                max_cpu_usage_percent=config.getfloat('resources', 'max_cpu_percent', fallback=80.0),
                email_notifications=config.getboolean('email', 'enabled', fallback=True),
                email_smtp_server=config.get('email', 'smtp_server', fallback='smtp.gmail.com'),
                email_smtp_port=config.getint('email', 'smtp_port', fallback=587),
                email_username=config.get('email', 'username', fallback=''),
                email_password=config.get('email', 'password', fallback=''),
                email_recipients=config.get('email', 'recipients', fallback='').split(','),
                test_coins=config.get('validation', 'test_coins', fallback='bitcoin,ethereum,solana').split(','),
                log_retention_days=config.getint('maintenance', 'log_retention_days', fallback=30),
                auto_restart_on_error=config.getboolean('maintenance', 'auto_restart', fallback=True),
                max_restart_attempts=config.getint('maintenance', 'max_restart_attempts', fallback=3)
            )
        else:
            # Create default config
            self._create_default_config()
            return DeploymentConfig()
    
    def _create_default_config(self):
        """Create default configuration file"""
        config = configparser.ConfigParser()
        
        config['validation'] = {
            'interval_hours': '1',
            'test_coins': 'bitcoin,ethereum,solana,cardano,polygon'
        }
        
        config['reporting'] = {
            'interval_hours': '24'
        }
        
        config['resources'] = {
            'max_memory_mb': '2048',
            'max_cpu_percent': '80.0'
        }
        
        config['email'] = {
            'enabled': 'true',
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': '587',
            'username': 'your_email@gmail.com',
            'password': 'your_app_password',
            'recipients': 'recipient1@email.com,recipient2@email.com'
        }
        
        config['maintenance'] = {
            'log_retention_days': '30',
            'auto_restart': 'true',
            'max_restart_attempts': '3'
        }
        
        with open(self.config_file, 'w') as f:
            config.write(f)
        
        self.logger.info(f"Created default config file: {self.config_file}")
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_file = self.logs_dir / f"deployment_{datetime.now().strftime('%Y%m%d')}.log"
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Setup logger
        self.logger = logging.getLogger('VPSDeployment')
        self.logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    async def start_continuous_validation(self):
        """Start continuous validation process"""
        self.logger.info("Starting continuous validation process")
        
        # Schedule validation runs
        schedule.every(self.config.validation_interval_hours).hours.do(
            self._run_validation_job
        )
        
        # Schedule reports
        schedule.every(self.config.report_interval_hours).hours.do(
            self._generate_report_job
        )
        
        # Schedule maintenance
        schedule.every().day.at("02:00").do(self._maintenance_job)
        
        # Send startup notification
        await self._send_notification("🚀 VPS Validation Started", 
                                     f"Continuous validation started at {datetime.now()}")
        
        # Main loop
        while True:
            try:
                # Run scheduled jobs
                schedule.run_pending()
                
                # System monitoring
                await self._monitor_system_resources()
                
                # Wait before next check
                await asyncio.sleep(60)  # Check every minute
                
            except KeyboardInterrupt:
                self.logger.info("Shutdown requested by user")
                await self._send_notification("🛑 VPS Validation Stopped", 
                                             "Validation stopped by user request")
                break
                
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                
                if self.config.auto_restart_on_error and self.restart_attempts < self.config.max_restart_attempts:
                    self.restart_attempts += 1
                    self.logger.info(f"Attempting restart {self.restart_attempts}/{self.config.max_restart_attempts}")
                    await self._send_notification("⚠️ VPS Validation Error", 
                                                 f"Error occurred, attempting restart {self.restart_attempts}")
                    await asyncio.sleep(300)  # Wait 5 minutes before restart
                else:
                    self.logger.error("Max restart attempts reached, stopping")
                    await self._send_notification("🚨 VPS Validation Failed", 
                                                 "Maximum restart attempts reached, manual intervention required")
                    break
    
    def _run_validation_job(self):
        """Run validation job (synchronous wrapper)"""
        asyncio.create_task(self._async_run_validation())
    
    async def _async_run_validation(self):
        """Run validation asynchronously"""
        try:
            self.logger.info("Starting validation run")
            
            # Run live validation for the configured interval
            metrics = await self.validator.run_live_validation(
                duration_hours=self.config.validation_interval_hours,
                interval_hours=1,  # Make forecasts every hour
                coins=self.config.test_coins
            )
            
            self.last_successful_run = datetime.now()
            self.restart_attempts = 0  # Reset on successful run
            
            self.logger.info(f"Validation completed successfully. Accuracy: {metrics.get('accuracy_percentage', 0):.1f}%")
            
        except Exception as e:
            self.logger.error(f"Error in validation run: {e}")
            await self._send_notification("❌ Validation Run Failed", 
                                         f"Error in validation: {str(e)}")
    
    def _generate_report_job(self):
        """Generate report job (synchronous wrapper)"""
        asyncio.create_task(self._async_generate_report())
    
    async def _async_generate_report(self):
        """Generate and send report asynchronously"""
        try:
            self.logger.info("Generating validation report")
            
            # Generate report
            report_file = self.analytics.generate_comprehensive_report()
            csv_file = self.analytics.export_metrics_csv()
            
            # Send report via email
            if self.config.email_notifications and report_file:
                await self._send_report_email(report_file, csv_file)
            
            self.logger.info("Report generated and sent successfully")
            
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
    
    def _maintenance_job(self):
        """Run maintenance tasks"""
        asyncio.create_task(self._async_maintenance())
    
    async def _async_maintenance(self):
        """Run maintenance tasks asynchronously"""
        try:
            self.logger.info("Running maintenance tasks")
            
            # Clean old logs
            self._clean_old_logs()
            
            # System health check
            health_report = self._generate_health_report()
            
            # Send health report
            await self._send_notification("🔧 Daily Health Report", health_report)
            
            self.logger.info("Maintenance completed")
            
        except Exception as e:
            self.logger.error(f"Error in maintenance: {e}")
    
    async def _monitor_system_resources(self):
        """Monitor system resources and alert if thresholds exceeded"""
        try:
            # Get memory usage
            memory = psutil.virtual_memory()
            memory_usage_mb = memory.used / 1024 / 1024
            
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Check thresholds
            if memory_usage_mb > self.config.max_memory_usage_mb:
                await self._send_notification("⚠️ High Memory Usage", 
                                             f"Memory usage: {memory_usage_mb:.0f}MB (limit: {self.config.max_memory_usage_mb}MB)")
            
            if cpu_percent > self.config.max_cpu_usage_percent:
                await self._send_notification("⚠️ High CPU Usage", 
                                             f"CPU usage: {cpu_percent:.1f}% (limit: {self.config.max_cpu_usage_percent}%)")
            
        except Exception as e:
            self.logger.error(f"Error monitoring resources: {e}")
    
    def _clean_old_logs(self):
        """Clean log files older than retention period"""
        cutoff_date = datetime.now() - timedelta(days=self.config.log_retention_days)
        
        for log_file in self.logs_dir.glob("*.log"):
            if datetime.fromtimestamp(log_file.stat().st_mtime) < cutoff_date:
                log_file.unlink()
                self.logger.info(f"Deleted old log file: {log_file}")
    
    def _generate_health_report(self) -> str:
        """Generate system health report"""
        uptime = datetime.now() - self.start_time
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        report = f"""
        🖥️ System Health Report
        
        Uptime: {uptime.days} days, {uptime.seconds // 3600} hours
        Last Successful Run: {self.last_successful_run or 'Never'}
        Restart Attempts: {self.restart_attempts}
        
        💾 Memory Usage: {memory.percent:.1f}% ({memory.used / 1024**3:.1f}GB / {memory.total / 1024**3:.1f}GB)
        💿 Disk Usage: {disk.percent:.1f}% ({disk.used / 1024**3:.1f}GB / {disk.total / 1024**3:.1f}GB)
        🔥 CPU Usage: {psutil.cpu_percent()}%
        
        📊 Validation Status: {'✅ Running' if self.last_successful_run and (datetime.now() - self.last_successful_run).seconds < 7200 else '❌ Issues detected'}
        """
        
        return report
    
    async def _send_notification(self, subject: str, message: str):
        """Send email notification"""
        if not self.config.email_notifications or not self.config.email_username:
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config.email_username
            msg['To'] = ', '.join(self.config.email_recipients)
            msg['Subject'] = f"[Crypto Validator] {subject}"
            
            body = f"""
            {message}
            
            ---
            Sent from Crypto Agent Forecaster VPS Deployment
            Server Time: {datetime.now()}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.config.email_smtp_server, self.config.email_smtp_port)
            server.starttls()
            server.login(self.config.email_username, self.config.email_password)
            
            text = msg.as_string()
            server.sendmail(self.config.email_username, self.config.email_recipients, text)
            server.quit()
            
            self.logger.info(f"Notification sent: {subject}")
            
        except Exception as e:
            self.logger.error(f"Failed to send notification: {e}")
    
    async def _send_report_email(self, report_file: str, csv_file: str):
        """Send validation report via email"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config.email_username
            msg['To'] = ', '.join(self.config.email_recipients)
            msg['Subject'] = f"[Crypto Validator] Daily Validation Report - {datetime.now().strftime('%Y-%m-%d')}"
            
            body = f"""
            📊 Daily Crypto Forecasting Validation Report
            
            Please find attached the comprehensive validation report and metrics.
            
            Report generated at: {datetime.now()}
            
            ---
            Crypto Agent Forecaster VPS Deployment
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach HTML report
            if os.path.exists(report_file):
                with open(report_file, "rb") as attachment:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment.read())
                    encoders.encode_base64(part)
                    part.add_header(
                        'Content-Disposition',
                        f'attachment; filename= {os.path.basename(report_file)}'
                    )
                    msg.attach(part)
            
            # Attach CSV file
            if os.path.exists(csv_file):
                with open(csv_file, "rb") as attachment:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment.read())
                    encoders.encode_base64(part)
                    part.add_header(
                        'Content-Disposition',
                        f'attachment; filename= {os.path.basename(csv_file)}'
                    )
                    msg.attach(part)
            
            server = smtplib.SMTP(self.config.email_smtp_server, self.config.email_smtp_port)
            server.starttls()
            server.login(self.config.email_username, self.config.email_password)
            
            text = msg.as_string()
            server.sendmail(self.config.email_username, self.config.email_recipients, text)
            server.quit()
            
            self.logger.info("Report sent via email")
            
        except Exception as e:
            self.logger.error(f"Failed to send report email: {e}")


def create_systemd_service():
    """Create systemd service file for automatic startup"""
    service_content = f"""
[Unit]
Description=Crypto Agent Forecaster Validation
After=network.target
Wants=network-online.target

[Service]
Type=simple
User={os.getenv('USER', 'root')}
WorkingDirectory={os.getcwd()}
ExecStart=/usr/local/bin/uv run python {os.path.abspath(__file__)}
Restart=always
RestartSec=300
Environment=PYTHONPATH={os.getcwd()}
Environment=PATH=/usr/local/bin:/usr/bin:/bin

[Install]
WantedBy=multi-user.target
"""
    
    service_file = Path('/etc/systemd/system/crypto-validator.service')
    
    try:
        with open(service_file, 'w') as f:
            f.write(service_content)
        
        print(f"Systemd service created: {service_file}")
        print("To enable and start the service:")
        print("sudo systemctl daemon-reload")
        print("sudo systemctl enable crypto-validator.service")
        print("sudo systemctl start crypto-validator.service")
        print("\nNote: Make sure uv is installed system-wide for the service to work:")
        print("curl -LsSf https://astral.sh/uv/install.sh | sh")
        
    except PermissionError:
        print("Permission denied. Run with sudo to create systemd service.")
        return False
    
    return True


def install_dependencies():
    """Install required dependencies using uv"""
    try:
        # Check if uv is installed
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
        print("✅ uv is already installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("📦 Installing uv...")
        try:
            # Install uv using pip as fallback
            subprocess.run([sys.executable, "-m", "pip", "install", "uv"], check=True)
            print("✅ uv installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install uv")
            return False
    
    try:
        # Install validation framework dependencies
        print("📦 Installing validation framework dependencies...")
        
        # Use uv to install the validation framework in development mode
        validation_dir = Path(__file__).parent
        subprocess.run([
            "uv", "pip", "install", "-e", str(validation_dir)
        ], check=True)
        
        print("✅ All dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="VPS Deployment Manager for Crypto Validator")
    parser.add_argument("--install-deps", action="store_true", help="Install required dependencies")
    parser.add_argument("--create-service", action="store_true", help="Create systemd service")
    parser.add_argument("--config", default="vps_config.ini", help="Configuration file path")
    
    args = parser.parse_args()
    
    if args.install_deps:
        install_dependencies()
        sys.exit(0)
    
    if args.create_service:
        create_systemd_service()
        sys.exit(0)
    
    # Start deployment manager
    manager = VPSDeploymentManager(args.config)
    
    try:
        asyncio.run(manager.start_continuous_validation())
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1) 