"""
4chan /biz/ board data fetching tool.
"""

import time
import json
import requests
from typing import List, Dict, Optional, Any
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from ..config import Config


class FourChanBizInput(BaseModel):
    """Input for 4chan /biz/ tool."""
    keywords: List[str] = Field(description="Keywords to search for in thread titles and posts")
    max_threads: int = Field(default=5, description="Maximum number of threads to fetch")
    max_posts_per_thread: int = Field(default=20, description="Maximum posts per thread")


class FourChanBizTool(BaseTool):
    """Tool for fetching cryptocurrency discussions from 4chan /biz/ board."""
    
    name: str = "fourchan_biz_tool"
    description: str = """
    Fetches cryptocurrency discussions from 4chan's /biz/ board.
    Use this tool to gather sentiment data from anonymous cryptocurrency discussions.
    Input should include keywords related to cryptocurrencies.
    """
    args_schema: type[BaseModel] = FourChanBizInput
    
    def __init__(self):
        super().__init__()
        self.base_url = Config.FOURCHAN_BASE_URL
        self.rate_limit = Config.FOURCHAN_RATE_LIMIT
        self.last_request_time = 0
    
    def _rate_limit_wait(self):
        """Ensure we don't exceed 4chan's rate limit of 1 request per second."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit:
            time.sleep(self.rate_limit - time_since_last)
        self.last_request_time = time.time()
    
    def _make_request(self, url: str) -> Optional[Dict[str, Any]]:
        """Make a rate-limited request to 4chan API."""
        self._rate_limit_wait()
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching from 4chan: {e}")
            return None
    
    def _clean_post_text(self, text: str) -> str:
        """Clean and normalize 4chan post text."""
        if not text:
            return ""
        
        # Remove HTML tags and entities
        import re
        text = re.sub(r'<[^>]+>', '', text)
        text = text.replace('&gt;', '>')
        text = text.replace('&lt;', '<')
        text = text.replace('&amp;', '&')
        text = text.replace('&quot;', '"')
        text = text.replace('&#039;', "'")
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _matches_keywords(self, text: str, keywords: List[str]) -> bool:
        """Check if text contains any of the keywords (case-insensitive)."""
        if not text or not keywords:
            return False
        
        text_lower = text.lower()
        return any(keyword.lower() in text_lower for keyword in keywords)
    
    def _get_catalog(self) -> Optional[List[Dict[str, Any]]]:
        """Fetch the /biz/ board catalog."""
        catalog_url = f"{self.base_url}/biz/catalog.json"
        catalog_data = self._make_request(catalog_url)
        
        if not catalog_data:
            return None
        
        threads = []
        for page in catalog_data:
            if 'threads' in page:
                threads.extend(page['threads'])
        
        return threads
    
    def _get_thread_posts(self, thread_no: int) -> Optional[List[Dict[str, Any]]]:
        """Fetch all posts from a specific thread."""
        thread_url = f"{self.base_url}/biz/thread/{thread_no}.json"
        thread_data = self._make_request(thread_url)
        
        if not thread_data or 'posts' not in thread_data:
            return None
        
        return thread_data['posts']
    
    def _run(
        self,
        keywords: List[str],
        max_threads: int = 5,
        max_posts_per_thread: int = 20,
    ) -> str:
        """
        Fetch and filter posts from 4chan /biz/ board.
        
        Args:
            keywords: List of keywords to search for
            max_threads: Maximum number of threads to process
            max_posts_per_thread: Maximum posts to collect per thread
            
        Returns:
            JSON string containing filtered posts
        """
        print(f"Fetching 4chan /biz/ data for keywords: {keywords}")
        
        # Get catalog
        threads = self._get_catalog()
        if not threads:
            return json.dumps({"error": "Failed to fetch 4chan catalog", "posts": []})
        
        # Filter threads by keywords in subject or comment
        relevant_threads = []
        for thread in threads:
            subject = thread.get('sub', '')
            comment = thread.get('com', '')
            
            if (self._matches_keywords(subject, keywords) or 
                self._matches_keywords(comment, keywords)):
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
            posts = self._get_thread_posts(thread_no)
            if not posts:
                continue
            
            posts_in_thread = 0
            for post in posts:
                if posts_in_thread >= max_posts_per_thread:
                    break
                
                comment = post.get('com', '')
                if not comment:
                    continue
                
                cleaned_text = self._clean_post_text(comment)
                
                # Only include posts that match keywords or are in relevant threads
                if (self._matches_keywords(cleaned_text, keywords) or 
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
            "posts": collected_posts
        }
        
        print(f"Collected {len(collected_posts)} posts from {threads_processed} threads")
        return json.dumps(result, indent=2)


def create_fourchan_tool() -> FourChanBizTool:
    """Create and return a 4chan /biz/ tool instance."""
    return FourChanBizTool() 