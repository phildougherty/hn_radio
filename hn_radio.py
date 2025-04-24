#!/usr/bin/env python3
"""
Hacker News Radio Show Generator
--------------------------------
Scrapes Hacker News, summarizes articles and comments, and creates a radio show audio file.
"""

import argparse
import json
import os
import re
import socket  # Added missing socket import
import sys
import time
from datetime import datetime
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
import html2text
import trafilatura


# Constants
HN_BASE_URL = "https://news.ycombinator.com"
HN_ITEM_API = "https://hacker-news.firebaseio.com/v0/item/{}.json"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
TTS_SERVER_URL = "http://192.168.86.201:8000/v1/audio/speech"

# Available voices based on the actual server capabilities
AVAILABLE_VOICES = [
    "af_bella", "af_alloy", "af_aoede", "af_heart", "af_jadzia", 
    "af_jessica", "af_kore", "af_nicole", "af_river",
    "bm_george", "bm_adam", "bm_brian", "bm_chris", "bm_dave", "bm_james", 
    "bm_jason", "bm_john", "bm_mike", "bm_paul"
]
DEFAULT_VOICE = "af_bella"  # Default voice

# Request timeouts (in seconds)
SCRAPE_TIMEOUT = 10
API_TIMEOUT = 30


class HackerNewsRadio:
    """Main class for the Hacker News Radio Show Generator"""
    
    def __init__(self, openrouter_api_key, model_name, num_articles=5, max_comments=10, 
                 output_file=None, tts_voice=DEFAULT_VOICE, verbose=False):
        """Initialize the HackerNewsRadio with configuration parameters"""
        self.openrouter_api_key = openrouter_api_key
        self.model_name = model_name
        self.num_articles = num_articles
        self.max_comments = max_comments
        self.verbose = verbose
        
        # Validate TTS voice
        if tts_voice not in AVAILABLE_VOICES:
            self.log(f"Voice '{tts_voice}' not available. Using {DEFAULT_VOICE} instead.")
            tts_voice = DEFAULT_VOICE
            
        self.tts_voice = tts_voice
        
        # Set up output file name with timestamp if not provided
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_file = f"hn_radio_{timestamp}.mp3"
        else:
            self.output_file = output_file
        
        # Initialize HTML to text converter
        self.text_converter = html2text.HTML2Text()
        self.text_converter.ignore_links = True
        self.text_converter.ignore_images = True
        self.text_converter.ignore_tables = True
        
        # Configure request session with headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'HackerNewsRadioBot/1.0 (Educational Project)'
        })
    
    def log(self, message):
        """Print message if verbose mode is enabled"""
        if self.verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
    
    def scrape_top_stories(self):
        """Scrape the top stories from the Hacker News front page"""
        self.log("Scraping Hacker News top stories...")
        try:
            response = self.session.get(HN_BASE_URL, timeout=SCRAPE_TIMEOUT)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            stories = []
            story_rows = soup.select('tr.athing')
            
            for i, row in enumerate(story_rows[:self.num_articles]):
                # Get the title and link
                title_cell = row.select_one('.titleline a')
                if not title_cell:
                    title_cell = row.select_one('.title a')  # Alternative selector
                
                if not title_cell:
                    continue
                    
                title = title_cell.get_text().strip()
                url = title_cell.get('href')
                
                # Add the HN base URL if it's a relative link
                if url and not url.startswith('http'):
                    url = f"{HN_BASE_URL}/{url}"
                
                # Get the story ID for fetching comments
                story_id = row.get('id')
                
                # Get the score and other metadata from the next row
                subtext_row = row.find_next_sibling('tr')
                score_text = subtext_row.select_one('.score').get_text() if subtext_row.select_one('.score') else "0 points"
                
                # Extract comment link to get comment count
                comment_link = None
                for link in subtext_row.select('a'):
                    if 'comment' in link.text.lower() or 'discuss' in link.text.lower():
                        comment_link = link.get('href')
                        break
                
                if comment_link and not comment_link.startswith('http'):
                    comment_link = f"{HN_BASE_URL}/{comment_link}"
                
                stories.append({
                    'id': story_id,
                    'title': title,
                    'url': url,
                    'score': score_text,
                    'comment_link': comment_link
                })
                
                self.log(f"Found story: {title}")
            
            return stories
        
        except requests.Timeout:
            self.log("Timeout while scraping Hacker News top stories")
            return []
        except Exception as e:
            self.log(f"Error scraping top stories: {str(e)}")
            return []
    
    def fetch_article_content(self, url):
        """Fetch and extract main content from the article URL"""
        self.log(f"Fetching content from {url}")
        
        try:
            # Handle special cases or non-article URLs
            parsed_url = urlparse(url)
            if parsed_url.netloc == 'news.ycombinator.com':
                return "This is a Hacker News discussion thread without an external article."
            
            # Use trafilatura for content extraction with fixed timeout handling
            try:
                # Set a socket timeout
                old_timeout = socket.getdefaulttimeout()
                socket.setdefaulttimeout(SCRAPE_TIMEOUT)
                
                try:
                    downloaded = trafilatura.fetch_url(url)
                    if downloaded:
                        content = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
                        if content:
                            # Trim and clean the content
                            content = content.strip()
                            # Remove excessively long content
                            if len(content) > 8000:
                                content = content[:8000] + "..."
                            return content
                finally:
                    # Restore original timeout
                    socket.setdefaulttimeout(old_timeout)
                    
            except Exception as e:
                self.log(f"Error with trafilatura: {str(e)}")
            
            # Fallback to basic request if trafilatura fails
            try:
                response = self.session.get(url, timeout=SCRAPE_TIMEOUT)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Remove script, style elements and comments
                for element in soup(['script', 'style', 'header', 'footer', 'nav']):
                    element.decompose()
                
                # Convert to plain text
                text = self.text_converter.handle(str(soup.find('body')))
                return text[:8000] + ("..." if len(text) > 8000 else "")
            except requests.Timeout:
                self.log(f"Timeout fetching content via direct request from {url}")
                return f"Unable to fetch article content due to timeout: {url}"
            except Exception as e:
                self.log(f"Error with direct request: {str(e)}")
                return f"Unable to fetch article content: {url}"
        
        except Exception as e:
            self.log(f"Error fetching article content: {str(e)}")
            return f"Unable to fetch article content due to: {str(e)}"
    
    def fetch_comments(self, story_id):
        """Fetch top comments for a given story"""
        self.log(f"Fetching comments for story {story_id}")
        
        try:
            # First get the item details to get the list of comment IDs
            if not story_id.isdigit():
                # Extract the numeric ID from the URL if needed
                parts = story_id.split('=')
                if len(parts) > 1:
                    story_id = parts[1]
                else:
                    return []
            
            story_url = HN_ITEM_API.format(story_id)
            response = self.session.get(story_url, timeout=SCRAPE_TIMEOUT)
            response.raise_for_status()
            story_data = response.json()
            
            if not story_data or 'kids' not in story_data:
                return []
            
            # Get only the top comments based on max_comments setting
            top_comment_ids = story_data['kids'][:self.max_comments]
            comments = []
            
            for comment_id in top_comment_ids:
                comment_data = self.fetch_comment_and_replies(comment_id)
                if comment_data:
                    comments.append(comment_data)
                    
                    # Add a small delay to avoid hitting rate limits
                    time.sleep(0.1)
            
            return comments
        
        except requests.Timeout:
            self.log(f"Timeout fetching comments for story {story_id}")
            return []
        except Exception as e:
            self.log(f"Error fetching comments: {str(e)}")
            return []
    
    def fetch_comment_and_replies(self, comment_id, depth=0, max_depth=2):
        """Recursively fetch a comment and its replies up to max_depth"""
        if depth > max_depth:
            return None
        
        try:
            comment_url = HN_ITEM_API.format(comment_id)
            response = self.session.get(comment_url, timeout=SCRAPE_TIMEOUT)
            response.raise_for_status()
            comment_data = response.json()
            
            if not comment_data or comment_data.get('deleted', False) or comment_data.get('dead', False):
                return None
            
            # Extract the text, handle HTML entities
            comment_text = comment_data.get('text', '')
            if comment_text:
                comment_text = BeautifulSoup(comment_text, 'html.parser').get_text()
            
            # Build the comment object
            comment = {
                'author': comment_data.get('by', 'anonymous'),
                'text': comment_text,
                'replies': []
            }
            
            # Fetch replies if they exist and we haven't reached max depth
            if 'kids' in comment_data and depth < max_depth:
                for kid_id in comment_data['kids'][:3]:  # Limit to 3 replies per comment
                    reply = self.fetch_comment_and_replies(kid_id, depth + 1, max_depth)
                    if reply:
                        comment['replies'].append(reply)
                        
                        # Add a small delay to avoid hitting rate limits
                        time.sleep(0.1)
            
            return comment
        
        except requests.Timeout:
            self.log(f"Timeout fetching comment {comment_id}")
            return None
        except Exception as e:
            self.log(f"Error fetching comment {comment_id}: {str(e)}")
            return None
    
    def clean_text_for_tts(self, text):
        """Clean text to make it more suitable for TTS processing"""
        if not text:
            return ""
        
        # Replace markdown-style links and formatting
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # Replace [text](url) with just text
        text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)  # Remove bullet points
        text = re.sub(r'`([^`]+)`', r'\1', text)  # Remove code ticks
        text = re.sub(r'_{1,2}([^_]+)_{1,2}', r'\1', text)  # Remove underscores for italics/bold
        text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)  # Remove asterisks for italics/bold
        
        # Fix spacing issues
        text = re.sub(r'\n{3,}', '\n\n', text)  # Replace multiple newlines with just two
        text = re.sub(r'\s{2,}', ' ', text)  # Replace multiple spaces with single space
        
        # Replace special characters that might cause TTS issues
        text = text.replace('&', 'and')
        text = text.replace('<', 'less than')
        text = text.replace('>', 'greater than')
        text = re.sub(r'https?://\S+', 'link', text)  # Replace URLs with "link"
        
        # Remove any remaining HTML or XML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Handle parentheses more naturally for speech
        text = re.sub(r'\(([^)]+)\)', r', \1,', text)
        
        return text.strip()
    
    def generate_radio_script(self, stories_with_content):
        """Generate a radio show script using the OpenRouter API"""
        self.log(f"Generating radio script using model: {self.model_name}")
        
        # Create a structured prompt for the model
        current_date = datetime.now().strftime("%A, %B %d, %Y")
        
        prompt = f"""
        You are the host of "Hacker News Radio", a daily tech news show. Today is {current_date}.
        
        Create a complete radio show script that introduces, summarizes, and discusses the following tech news stories.
        The format should include:
        
        1. An enthusiastic introduction for the show
        2. For each story:
           - Introduction of the topic
           - Detailed summary of the article
           - Discussion of key points from the comments
           - Your analysis or opinion as the host
        3. A conclusion that wraps up the show
        
        Make it engaging, conversational, and suitable for audio consumption. Include appropriate transitions between segments.
        Keep in mind that this will be converted to speech, so avoid elements that don't work well in audio format.
        
        Here are today's stories:
        """
        
        # Add each story's information to the prompt
        for i, story in enumerate(stories_with_content, 1):
            prompt += f"\n\nSTORY {i}: {story['title']}\n"
            prompt += f"URL: {story['url']}\n"
            prompt += f"Score: {story['score']}\n\n"
            
            # Add article summary
            prompt += f"ARTICLE CONTENT:\n{story['content'][:3000]}\n\n"  # Limit content length
            
            # Add comments
            prompt += "TOP COMMENTS:\n"
            if story['comments']:
                for j, comment in enumerate(story['comments'], 1):
                    prompt += f"Comment {j} by {comment['author']}:\n{comment['text'][:500]}\n"
                    
                    # Add some replies if available
                    if comment['replies']:
                        for k, reply in enumerate(comment['replies'], 1):
                            prompt += f"- Reply by {reply['author']}:\n{reply['text'][:200]}\n"
            else:
                prompt += "No comments available for this story.\n"
        
        # Send the prompt to OpenRouter API
        try:
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "You are an engaging radio show host specializing in technology news."},
                    {"role": "user", "content": prompt}
                ]
            }
            
            response = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=API_TIMEOUT
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract the generated script
            script = result['choices'][0]['message']['content']
            self.log(f"Successfully generated script ({len(script)} chars)")
            clean_script = self.clean_text_for_tts(script)
            return clean_script
        
        except requests.Timeout:
            self.log("Timeout generating radio script from OpenRouter")
            return "We're experiencing technical difficulties with our Hacker News Radio service due to a timeout. Please try again later."
        except Exception as e:
            self.log(f"Error generating radio script: {str(e)}")
            return f"We apologize, but we're experiencing technical difficulties with our Hacker News Radio service. The error was: {str(e)}. Please try again later."
    
    def text_to_speech(self, script):
        """Convert the script to speech using the Kokoro TTS server"""
        self.log("Sending script to Kokoro TTS server...")
        
        try:
            headers = {
                "Content-Type": "application/json"
            }
            
            # Break the script into smaller chunks to avoid TTS limitations
            # Maximum safe length for TTS input (characters)
            max_chunk_size = 4000
            
            # Split by paragraphs first
            paragraphs = re.split(r'\n\s*\n', script)
            chunks = []
            current_chunk = ""
            
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                
                # If adding this paragraph exceeds chunk size, start a new chunk
                if len(current_chunk) + len(paragraph) > max_chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = paragraph
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + paragraph
                    else:
                        current_chunk = paragraph
            
            # Add any remaining content as the last chunk
            if current_chunk:
                chunks.append(current_chunk)
                
            self.log(f"Split script into {len(chunks)} chunks for TTS processing")
            
            # Process each chunk and combine the audio
            complete_audio = bytearray()
            
            for i, chunk in enumerate(chunks):
                self.log(f"Processing TTS chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
                
                payload = {
                    "model": "tts-1",
                    "input": chunk,
                    "voice": self.tts_voice
                }
                
                response = requests.post(TTS_SERVER_URL, 
                                       headers=headers, 
                                       json=payload, 
                                       timeout=API_TIMEOUT)
                
                if response.status_code != 200:
                    self.log(f"TTS error: Status {response.status_code}, Response: {response.text[:200]}")
                    raise Exception(f"TTS API failed with status {response.status_code}")
                
                # Append this chunk's audio to the complete audio
                complete_audio.extend(response.content)
                
                # Short delay between requests
                time.sleep(0.5)
            
            self.log(f"Successfully received complete audio ({len(complete_audio)} bytes)")
            return bytes(complete_audio)
            
        except requests.Timeout:
            self.log("Timeout connecting to TTS server")
            return None
        except Exception as e:
            self.log(f"Error with TTS server: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            return None
    
    def verify_audio_file(self, file_path):
        """Verify that the generated MP3 file is valid"""
        self.log(f"Verifying audio file: {file_path}")
        
        try:
            # Check file exists and has content
            if not os.path.exists(file_path):
                self.log("File does not exist")
                return False
                
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                self.log("File exists but is empty")
                return False
                
            self.log(f"File exists with size: {file_size} bytes")
            
            # Check first few bytes for MP3 header signature
            with open(file_path, 'rb') as f:
                header = f.read(4)
                
            # MP3 files typically start with ID3 or with an MP3 frame header
            if header.startswith(b'ID3') or (header[0] == 0xFF and (header[1] & 0xE0) == 0xE0):
                self.log("File appears to be a valid MP3 (has proper header)")
                return True
            else:
                self.log(f"File does not have a valid MP3 header. First bytes: {header.hex()}")
                return False
                
        except Exception as e:
            self.log(f"Error verifying audio file: {str(e)}")
            return False
    
    def save_audio_file(self, audio_data):
        """Save the audio data as an MP3 file"""
        self.log(f"Saving audio to {self.output_file}")
        
        try:
            with open(self.output_file, 'wb') as f:
                f.write(audio_data)
            self.log(f"Successfully saved audio file ({len(audio_data)} bytes)")
            
            # Verify the audio file
            is_valid = self.verify_audio_file(self.output_file)
            if not is_valid:
                self.log("WARNING: The generated audio file may not be valid")
            
            return True
        
        except Exception as e:
            self.log(f"Error saving audio file: {str(e)}")
            return False
    
    def run(self):
        """Main workflow to generate the radio show"""
        self.log("Starting Hacker News Radio Show generation process")
        
        # 1. Scrape top stories
        stories = self.scrape_top_stories()
        if not stories:
            self.log("No stories found. Exiting.")
            return False
        
        # 2. For each story, fetch article content and comments
        stories_with_content = []
        for story in stories:
            content = self.fetch_article_content(story['url'])
            comments = []
            
            # Extract numeric ID for comment fetching if available
            if 'id' in story and story['id']:
                comments = self.fetch_comments(story['id'])
            elif 'comment_link' in story and story['comment_link']:
                # Try to extract ID from comment link
                parts = story['comment_link'].split('id=')
                if len(parts) > 1:
                    comments = self.fetch_comments(parts[1])
            
            stories_with_content.append({
                'title': story['title'],
                'url': story['url'],
                'score': story['score'],
                'content': content,
                'comments': comments
            })
        
        # 3. Generate radio script
        script = self.generate_radio_script(stories_with_content)
        
        # 4. Convert script to speech
        audio_data = self.text_to_speech(script)
        if not audio_data:
            self.log("Failed to get audio data from TTS server. Exiting.")
            return False
        
        # 5. Save the audio file
        success = self.save_audio_file(audio_data)
        
        # 6. Output the script to file as well (for reference)
        script_file = self.output_file.replace('.mp3', '.txt')
        try:
            with open(script_file, 'w', encoding='utf-8') as f:
                f.write(script)
            self.log(f"Script saved to {script_file}")
        except Exception as e:
            self.log(f"Error saving script file: {str(e)}")
        
        return success


def main():
    """Parse command line arguments and run the HackerNewsRadio"""
    parser = argparse.ArgumentParser(description='Generate a Hacker News radio show')
    
    parser.add_argument('--api-key', required=True, help='OpenRouter API Key')
    parser.add_argument('--model', default='anthropic/claude-3-sonnet-20240229', 
                      help='Model to use for script generation (default: anthropic/claude-3-sonnet-20240229)')
    parser.add_argument('--articles', type=int, default=5, 
                      help='Number of articles to include (default: 5)')
    parser.add_argument('--comments', type=int, default=10,
                      help='Maximum number of comments per article (default: 10)')
    parser.add_argument('--output', '-o', help='Output MP3 file name')
    parser.add_argument('--voice', default=DEFAULT_VOICE,
                      help=f'TTS voice to use (default: {DEFAULT_VOICE}). Available voices: {", ".join(AVAILABLE_VOICES)}')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show verbose output')
    
    args = parser.parse_args()
    
    radio = HackerNewsRadio(
        openrouter_api_key=args.api_key,
        model_name=args.model,
        num_articles=args.articles,
        max_comments=args.comments,
        output_file=args.output,
        tts_voice=args.voice,
        verbose=args.verbose
    )
    
    success = radio.run()
    
    if success:
        print(f"Radio show successfully generated as {radio.output_file}")
        return 0
    else:
        print("Failed to generate radio show")
        return 1


if __name__ == "__main__":
    sys.exit(main())
