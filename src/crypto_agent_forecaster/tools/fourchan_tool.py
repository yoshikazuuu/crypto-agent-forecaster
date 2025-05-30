"""
4chan /biz/ board data fetching tool.
"""

import time
import json
import requests
import re
from typing import List, Dict, Optional, Any
from crewai.tools import tool

from ..config import Config


@tool("fourchan_biz_tool")
def fourchan_biz_tool(keywords: List[str], max_threads: int = 5, max_posts_per_thread: int = 20) -> str:
    """
    Fetches cryptocurrency discussions from 4chan's /biz/ board.
    
    Args:
        keywords: Keywords to search for in thread titles and posts
        max_threads: Maximum number of threads to fetch (default: 5)
        max_posts_per_thread: Maximum posts per thread (default: 20)
    
    Returns:
        JSON string containing filtered posts and metadata
    """
    
    def _rate_limit_wait(last_request_time: float) -> float:
        """Ensure we don't exceed 4chan's rate limit of 1 request per second."""
        current_time = time.time()
        time_since_last = current_time - last_request_time
        if time_since_last < Config.FOURCHAN_RATE_LIMIT:
            time.sleep(Config.FOURCHAN_RATE_LIMIT - time_since_last)
        return time.time()
    
    def _make_request(url: str, last_request_time: float) -> tuple[Optional[Dict[str, Any]], float]:
        """Make a rate-limited request to 4chan API."""
        last_request_time = _rate_limit_wait(last_request_time)
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json(), last_request_time
        except requests.exceptions.RequestException as e:
            print(f"Error fetching from 4chan: {e}")
            return None, last_request_time
    
    def _clean_post_text(text: str) -> str:
        """Clean and normalize 4chan post text."""
        if not text:
            return ""
        
        # Remove HTML tags and entities
        text = re.sub(r'<[^>]+>', '', text)
        text = text.replace('&gt;', '>')
        text = text.replace('&lt;', '<')
        text = text.replace('&amp;', '&')
        text = text.replace('&quot;', '"')
        text = text.replace('&#039;', "'")
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _matches_keywords(text: str, keywords: List[str]) -> bool:
        """Check if text contains any of the keywords (case-insensitive)."""
        if not text or not keywords:
            return False
        
        text_lower = text.lower()
        return any(keyword.lower() in text_lower for keyword in keywords)
    
    def _get_catalog(last_request_time: float) -> tuple[Optional[List[Dict[str, Any]]], float]:
        """Fetch the /biz/ board catalog."""
        catalog_url = f"{Config.FOURCHAN_BASE_URL}/biz/catalog.json"
        catalog_data, last_request_time = _make_request(catalog_url, last_request_time)
        
        if not catalog_data:
            return None, last_request_time
        
        threads = []
        for page in catalog_data:
            if 'threads' in page:
                threads.extend(page['threads'])
        
        return threads, last_request_time
    
    def _get_thread_posts(thread_no: int, last_request_time: float) -> tuple[Optional[List[Dict[str, Any]]], float]:
        """Fetch all posts from a specific thread."""
        thread_url = f"{Config.FOURCHAN_BASE_URL}/biz/thread/{thread_no}.json"
        thread_data, last_request_time = _make_request(thread_url, last_request_time)
        
        if not thread_data or 'posts' not in thread_data:
            return None, last_request_time
        
        return thread_data['posts'], last_request_time
    
    # Main execution
    print(f"Fetching 4chan /biz/ data for keywords: {keywords}")
    
    last_request_time = 0.0
    
    # Get catalog
    threads, last_request_time = _get_catalog(last_request_time)
    if not threads:
        return json.dumps({"error": "Failed to fetch 4chan catalog", "posts": []})
    
    # Filter threads by keywords in subject or comment
    relevant_threads = []
    for thread in threads:
        subject = thread.get('sub', '')
        comment = thread.get('com', '')
        
        if (_matches_keywords(subject, keywords) or 
            _matches_keywords(comment, keywords)):
            relevant_threads.append(thread)
    
    # Sort by reply count (engagement) and take top threads
    relevant_threads.sort(key=lambda x: x.get('replies', 0), reverse=True)
    relevant_threads = relevant_threads[:max_threads]
    
    collected_posts = []
    threads_processed = 0
    
    for thread in relevant_threads:
        if threads_processed >= max_threads:
            break
            
        thread_no = thread.get('no')
        if not thread_no:
            continue
        
        print(f"Processing thread {thread_no}...")
        
        # Get all posts in thread
        posts, last_request_time = _get_thread_posts(thread_no, last_request_time)
        if not posts:
            continue
        
        posts_in_thread = 0
        for post in posts:
            if posts_in_thread >= max_posts_per_thread:
                break
            
            comment = post.get('com', '')
            if not comment:
                continue
            
            cleaned_text = _clean_post_text(comment)
            
            # Only include posts that match keywords or are in relevant threads
            if (_matches_keywords(cleaned_text, keywords) or 
                thread_no in [t.get('no') for t in relevant_threads]):
                
                collected_posts.append({
                    'thread_no': thread_no,
                    'post_no': post.get('no'),
                    'timestamp': post.get('time'),
                    'text': cleaned_text,
                    'subject': thread.get('sub', ''),
                    'replies': thread.get('replies', 0)
                })
                posts_in_thread += 1
        
        threads_processed += 1
    
    result = {
        "total_posts": len(collected_posts),
        "threads_processed": threads_processed,
        "keywords": keywords,
        "posts": collected_posts,
        "metadata": {
            "4chan_api_used": True,
            "rate_limit_respected": True,
            "collection_timestamp": time.time()
        }
    }
    
    print(f"Collected {len(collected_posts)} posts from {threads_processed} threads")
    return json.dumps(result, indent=2)


# Legacy wrapper for backward compatibility
class FourChanBizTool:
    """Legacy wrapper for the fourchan_biz_tool function."""
    
    def __init__(self):
        self.name = "fourchan_biz_tool"
        self.description = """
        Fetches cryptocurrency discussions from 4chan's /biz/ board.
        Use this tool to gather sentiment data from anonymous cryptocurrency discussions.
        Input should include keywords related to cryptocurrencies.
        """
    
    def _run(self, keywords: List[str], max_threads: int = 5, max_posts_per_thread: int = 20) -> str:
        """Legacy interface for the tool."""
        return fourchan_biz_tool.func(keywords, max_threads, max_posts_per_thread)


def create_fourchan_tool():
    """Create and return a 4chan /biz/ tool instance."""
    return fourchan_biz_tool 