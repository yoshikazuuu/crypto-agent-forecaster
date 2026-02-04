"""
Warosu archive tool for fetching historical 4chan /biz/ data.
"""

import time
import requests
from bs4 import BeautifulSoup
import re
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from crewai.tools import tool
import json

from ..config import Config


@tool("warosu_archive_tool")
def warosu_archive_tool(keywords: List[str], date_from: Optional[str] = None, date_to: Optional[str] = None, max_posts: int = 50, historical_date: Optional[str] = None) -> str:
    """
    Fetches historical cryptocurrency discussions from Warosu 4chan /biz/ archive.
    
    Args:
        keywords: Keywords to search for in posts (e.g., ['btc', 'bitcoin'])
        date_from: Start date in YYYY-MM-DD format (optional if historical_date provided)
        date_to: End date in YYYY-MM-DD format (optional if historical_date provided)
        max_posts: Maximum number of posts to fetch (default: 50)
        historical_date: Optional date in YYYY-MM-DD format for backtesting mode
    
    Returns:
        JSON string containing historical posts and metadata
    """
    
    def _clean_html(raw_html: str) -> str:
        """Clean HTML and extract text content."""
        if not raw_html:
            return ""
        
        # Parse HTML
        soup = BeautifulSoup(raw_html, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text and clean it
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Clean up common 4chan artifacts
        text = re.sub(r'>>\d+', '', text)  # Remove reply links
        text = re.sub(r'Anonymous', '', text)  # Remove Anonymous
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        
        return text.strip()
    
    def _extract_post_data(post_element) -> Optional[Dict[str, Any]]:
        """Extract post data from Warosu HTML element."""
        try:
            # Extract post content from blockquote
            blockquote = post_element.find('blockquote')
            if not blockquote:
                return None
            
            text = _clean_html(str(blockquote))
            if not text or len(text) < 10:  # Skip very short posts
                return None
            
            # Extract post ID from the element's id attribute (e.g., "p59522611")
            post_no = None
            post_id = post_element.get('id')
            if post_id and post_id.startswith('p'):
                try:
                    post_no = int(post_id[1:])  # Remove 'p' prefix
                except ValueError:
                    pass
            
            # Extract timestamp from span.posttime
            timestamp = None
            time_elem = post_element.find('span', class_='posttime')
            if time_elem:
                title_attr = time_elem.get('title')
                if title_attr:
                    try:
                        # Convert timestamp milliseconds to datetime
                        timestamp_ms = int(title_attr)
                        timestamp = timestamp_ms / 1000  # Convert to seconds
                    except (ValueError, TypeError):
                        pass
                
                # If no title, try to parse the text content
                if not timestamp:
                    time_text = time_elem.get_text().strip()
                    if time_text:
                        try:
                            # Try parsing different date formats
                            time_formats = [
                                '%a, %b %d, %Y %H:%M:%S',
                                '%a, %b %d, %Y %H:%M:%S',
                                '%Y-%m-%d %H:%M:%S'
                            ]
                            for fmt in time_formats:
                                try:
                                    dt = datetime.strptime(time_text, fmt)
                                    timestamp = dt.timestamp()
                                    break
                                except ValueError:
                                    continue
                        except:
                            pass
            
            # Extract thread number from post link
            thread_no = None
            post_link = post_element.find('a', href=re.compile(r'/biz/thread/\d+'))
            if post_link:
                href = post_link.get('href', '')
                thread_match = re.search(r'/biz/thread/(\d+)', href)
                if thread_match:
                    thread_no = int(thread_match.group(1))
            
            # Extract username (usually "Anonymous" on 4chan)
            username = "Anonymous"
            username_elem = post_element.find('span', class_='postername')
            if username_elem:
                username = username_elem.get_text().strip()
            
            return {
                'post_no': post_no,
                'thread_no': thread_no,
                'timestamp': timestamp,
                'username': username,
                'text': text,
                'source': 'warosu_archive'
            }
            
        except Exception as e:
            print(f"Error extracting post data: {e}")
            return None
    
    def _search_warosu(keywords: List[str], date_from: str, date_to: str, max_posts: int) -> List[Dict[str, Any]]:
        """Search Warosu archive for posts using real search API."""
        
        print(f"Searching Warosu for {keywords} from {date_from} to {date_to}")
        
        try:
            search_url = "https://warosu.org/biz/"
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            })
            
            # Prepare search parameters for the advanced search
            search_params = {
                'task': 'search2',
                'ghost': 'false',
                'search_text': ' '.join(keywords), 
                'search_datefrom': date_from,
                'search_dateto': date_to,
                'search_op': 'all',
                'search_del': 'dontcare',
                'search_int': 'dontcare',
                'search_ord': 'new', 
                'search_capcode': 'all',
                'search_res': 'post' 
            }
            
            print(f"Search parameters: {search_params}")
            
            # Submit the search
            response = session.post(search_url, data=search_params, timeout=15)
            response.raise_for_status()
            
            if response.status_code != 200:
                print(f"Search failed with status {response.status_code}")
                return []
            
            # Parse the results
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Check for errors
            error_div = soup.find('div', class_='error')
            if error_div:
                print(f"Search error: {error_div.get_text().strip()}")
                return []
            
            # Look for posts - they have class "comment reply" and ID starting with "p"
            post_elements = soup.find_all('td', class_=lambda x: x and 'comment' in x)
            
            posts = []
            for post_elem in post_elements:
                if len(posts) >= max_posts:
                    break
                    
                post_data = _extract_post_data(post_elem)
                if post_data:
                    posts.append(post_data)
            
            print(f"Successfully extracted {len(posts)} posts from Warosu search")
            return posts
            
        except requests.exceptions.RequestException as e:
            print(f"Failed to access Warosu: {e}")
            return []
        except Exception as e:
            print(f"Error searching Warosu: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    # Main execution
    try:
        # Handle historical backtesting mode
        if historical_date and historical_date.strip():
            from datetime import datetime, timedelta
            current_date = datetime.strptime(historical_date, '%Y-%m-%d')
            prev_date = current_date - timedelta(days=1)
            date_from = prev_date.strftime('%Y-%m-%d')
            date_to = historical_date
            print(f"Historical mode: Searching from {date_from} to {date_to}")
        
        # Auto-generate date range if not provided (for live mode)
        if not date_from or not date_to:
            from datetime import datetime, timedelta
            current_date = datetime.now()
            date_to = current_date.strftime('%Y-%m-%d')
            date_from = (current_date - timedelta(days=7)).strftime('%Y-%m-%d')  # Default to last 7 days
            print(f"Auto-generated date range: {date_from} to {date_to}")
        
        # Validate date format if provided
        
        try:
            datetime.strptime(date_from, '%Y-%m-%d')
            datetime.strptime(date_to, '%Y-%m-%d')
        except ValueError:
            return json.dumps({"error": "Invalid date format. Use YYYY-MM-DD"})
        
        if not keywords:
            return json.dumps({"error": "Keywords list cannot be empty"})
        
        posts = _search_warosu(keywords, date_from, date_to, max_posts)
        
        result = {
            "total_posts": len(posts),
            "date_range": f"{date_from} to {date_to}",
            "keywords": keywords,
            "posts": posts,
            "metadata": {
                "source": "warosu_archive",
                "search_type": "historical",
                "collection_timestamp": time.time(),
                "max_posts_requested": max_posts
            }
        }
        
        print(f"Successfully collected {len(posts)} historical posts from Warosu archive")
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Warosu archive error: {str(e)}"})


# Legacy wrapper for backward compatibility
class WarosuArchiveTool:
    """Legacy wrapper for the warosu_archive_tool function."""
    
    def __init__(self):
        self.name = "warosu_archive_tool"
        self.description = """
        Fetches historical cryptocurrency discussions from Warosu 4chan /biz/ archive.
        Use this tool to gather historical sentiment data for backtesting and analysis.
        Supports date range queries and keyword filtering.
        
        Args:
            keywords: List of keywords to search for (REQUIRED)
            date_from: Start date in YYYY-MM-DD format (optional)
            date_to: End date in YYYY-MM-DD format (optional)  
            max_posts: Maximum number of posts to fetch (default: 50)
            historical_date: Historical date for backtesting mode (optional)
        """
    
    def _run(self, keywords: List[str], date_from: Optional[str] = None, date_to: Optional[str] = None, max_posts: int = 50, historical_date: Optional[str] = None) -> str:
        """Legacy interface for the tool."""
        return warosu_archive_tool.func(keywords, date_from, date_to, max_posts, historical_date)
    
    def __call__(self, keywords: List[str], date_from: Optional[str] = None, date_to: Optional[str] = None, max_posts: int = 50, historical_date: Optional[str] = None) -> str:
        """Allow the tool to be called directly."""
        return self._run(keywords, date_from, date_to, max_posts, historical_date)


def create_warosu_tool():
    """Create and return a Warosu archive tool instance."""
    return WarosuArchiveTool() 