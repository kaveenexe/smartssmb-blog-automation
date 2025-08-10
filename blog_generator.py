"""
Automated Blog Content Generation Script
Replicates n8n workflow for AI-powered blog creation with Supabase, Perplexity, OpenRouter, Pexels, and Telegram
"""

import os
import re
import json
import random
import requests
import schedule
import time
import logging
from datetime import datetime
from urllib.parse import quote
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('blog_generator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration from environment variables
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
PEXELS_API_KEY = os.getenv('PEXELS_API_KEY')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Supabase headers
SUPABASE_HEADERS = {
    'apikey': SUPABASE_KEY,
    'Authorization': f'Bearer {SUPABASE_KEY}',
    'Content-Type': 'application/json'
}

class BlogGenerationError(Exception):
    """Custom exception for blog generation errors"""
    pass

def slugify(text: str) -> str:
    """
    Generate unique slug from text with timestamp and random suffix
    Based on the provided slugify function
    """
    if not text:
        text = 'post'
    
    slug = text.strip().lower()
    # Normalize Unicode characters
    slug = slug.encode('ascii', 'ignore').decode('ascii')
    # Replace spaces with hyphens
    slug = re.sub(r'\s+', '-', slug)
    # Remove non-word characters except hyphens
    slug = re.sub(r'[^\w\-]+', '', slug)
    # Replace multiple hyphens with single hyphen
    slug = re.sub(r'\-\-+', '-', slug)
    # Remove leading/trailing hyphens
    slug = slug.strip('-')
    
    # Add uniqueness with timestamp and random number
    timestamp = int(time.time() * 1000)
    random_suffix = random.randint(1000, 9999)
    
    return f"{slug}-{timestamp}-{random_suffix}"

def estimate_reading_time(text: str, words_per_minute: int = 200) -> int:
    """
    Estimate reading time based on word count
    Based on the provided reading time function
    """
    if not isinstance(text, str):
        return 0
    
    words = text.strip().split()
    word_count = len([word for word in words if word])
    
    return max(1, round(word_count / words_per_minute))

def supabase_get_all(table: str) -> List[Dict[str, Any]]:
    """Fetch all records from a Supabase table"""
    logger.info(f"Fetching all records from {table} table")
    
    url = f"{SUPABASE_URL}/rest/v1/{table}?select=*"
    
    try:
        response = requests.get(url, headers=SUPABASE_HEADERS)
        response.raise_for_status()
        data = response.json()
        logger.info(f"Successfully fetched {len(data)} records from {table}")
        return data
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data from {table}: {str(e)}")
        raise BlogGenerationError(f"Failed to fetch {table} data")

def build_perplexity_prompt(posts: List[Dict], tags: List[Dict], categories: List[Dict]) -> str:
    """
    Build optimized prompt for Perplexity to minimize token usage
    """
    logger.info("Building Perplexity prompt")
    
    # Extract only titles to avoid duplicates
    all_titles = ' | '.join([post.get('title', '') for post in posts[-50:]])  # Limit to recent posts
    
    # Build compact tag and category lists
    tags_list = [{'name': tag.get('name', ''), 'slug': tag.get('slug', '')} for tag in tags]
    categories_list = [{'name': cat.get('name', ''), 'slug': cat.get('slug', '')} for cat in categories]
    
    prompt = f"""Suggest a unique trending blog post topic about AI tools or automation for small businesses in 2025. Avoid topics similar to these titles: {all_titles}.

You have these tags and categories available:

Tags: {json.dumps(tags_list)}

Categories: {json.dumps(categories_list)}

Choose exactly THREE tags and ONE category from the above lists for this blog post topic.

Return JSON like:
{{
  "topic": "...",
  "tags": ["tag-slug-1", "tag-slug-2", "tag-slug-3"],
  "category": "category-slug"
}}

RESEARCH MISSION: Find the most trending, viral, and newsworthy AI topic for small businesses in 2025 that will:
- Get massive social media shares (LinkedIn, Facebook, Twitter)
- Address urgent pain points small business owners face RIGHT NOW
- Showcase game-changing AI tools or breakthroughs from the last 30 days
- Include surprising statistics, case studies, or "David vs Goliath" stories
- Focus on ROI, cost savings, or competitive advantages

5 major points to cover:
1. Current market trends and opportunities
2. Practical implementation strategies
3. Case studies and real-world examples
4. Integration with existing business systems
5. ROI measurement and optimization techniques"""

    logger.info(f"Perplexity prompt built (length: {len(prompt)} chars)")
    return prompt

def call_perplexity_api(prompt: str) -> str:
    """Call Perplexity API for content research"""
    logger.info("Calling Perplexity API")
    
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {PERPLEXITY_API_KEY}'
    }
    
    payload = {
        'model': 'sonar',
        'messages': [{'role': 'user', 'content': prompt}],
        'stream': False
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        content = response.json()['choices'][0]['message']['content']
        logger.info("Perplexity API call successful")
        return content
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Perplexity API error: {str(e)}")
        raise BlogGenerationError(f"Perplexity API failed: {str(e)}")

def parse_perplexity_output(content: str) -> Dict[str, Any]:
    """
    Parse Perplexity response to extract JSON and major points
    Based on the provided parsing function
    """
    logger.info("Parsing Perplexity output")
    
    # Parse JSON safely
    parsed = {}
    try:
        # Extract JSON part
        json_match = re.search(r'{[\s\S]*}', content)
        if json_match:
            parsed = json.loads(json_match.group(0))
        else:
            parsed = json.loads(content)
    except Exception as e:
        logger.error(f"Failed to parse Perplexity JSON: {str(e)}")
        raise BlogGenerationError(f"Failed to parse Perplexity JSON content: {str(e)}")

    # Extract major points
    points_match = re.search(r'major points to cover:\n([\s\S]*)', content, re.IGNORECASE)
    major_points = []
    if points_match:
        major_points = [
            re.sub(r'^\d+\.\s*', '', line.strip()).strip()
            for line in points_match.group(1).split('\n')
            if line.strip()
        ]

    result = {
        'topic': parsed.get('topic', ''),
        'tags': parsed.get('tags', []),
        'category': parsed.get('category', ''),
        'major_points': major_points
    }
    
    logger.info(f"Parsed topic: {result['topic']}")
    return result

def build_openrouter_messages(perplexity_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Build messages for OpenRouter chat completion"""
    logger.info("Building OpenRouter messages")
    
    topic = perplexity_data['topic']
    category = perplexity_data['category']
    tags = perplexity_data['tags']
    major_points = perplexity_data['major_points']

    content = f"""Write a single, comprehensive 1,500–2,200 word blog post in clean markdown format on the following topic:

{topic}

Use these SEO tags and category exactly as provided:

Category: {category}

Tags: {', '.join(tags)}


Focus your content on these major points:
"""
    
    for i, point in enumerate(major_points, 1):
        content += f"{i}. {point}\n"

    content += f"""
Instructions:
- Use H2 and H3 markdown headers for structure.
- Include lists, code blocks, and examples where relevant.
- Write for 8th-grade reading level
- Start with an engaging introduction highlighting practical value for small business owners.
- Provide specific actionable steps and tool recommendations.
- At the end of the post, output the following JSON metadata inside a markdown code block labeled json:

{{
  "meta_title": "...",
  "meta_description": "...",
  "excerpt": "...",
  "reading_time": "...",
  "category": "{category}",
  "tags": {json.dumps(tags)},
  "keyword_clusters": ["..."],
  "long_tail_keywords": ["..."],
  "focus_keywords": ["..."],
  "suggested_image_query": "..."
}}

- Do NOT return any HTML; only markdown and the final JSON metadata block.
- Output only ONE blog post article.
"""

    return [
        {
            "role": "system",
            "content": "You are an expert blog writer specializing in AI, automation, and small business. Write comprehensive, SEO-optimized markdown blog posts suited for small business owners."
        },
        {
            "role": "user",
            "content": content
        }
    ]

def call_openrouter_chat(messages: List[Dict[str, str]]) -> str:
    """Call OpenRouter API for blog content generation"""
    logger.info("Calling OpenRouter API")
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        'Authorization': f'Bearer {OPENROUTER_API_KEY}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        "model": "mistralai/mistral-small-3.1-24b-instruct:free",
        "messages": messages,
        "max_tokens": 2500,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        content = response.json()['choices'][0]['message']['content']
        logger.info("OpenRouter API call successful")
        return content
        
    except requests.exceptions.RequestException as e:
        logger.error(f"OpenRouter API error: {str(e)}")
        raise BlogGenerationError(f"OpenRouter API failed: {str(e)}")

def parse_openrouter_output(text: str) -> Dict[str, Any]:
    """
    Parse OpenRouter output to extract markdown and metadata
    """
    logger.info("Parsing OpenRouter output")
    
    # Ensure text is not None
    if not text:
        logger.warning("Empty or None text received from OpenRouter")
        text = ""
    
    markdown = ''
    meta = {}

    # Find JSON metadata block
    json_block_start = text.find('```json')
    if json_block_start != -1:
        # Everything before JSON block is markdown
        markdown = text[:json_block_start].strip()

        # Find end of JSON block
        json_block_end = text.find('```', json_block_start + 7)
        if json_block_end != -1:
            raw_json = text[json_block_start + 7:json_block_end].strip()
            try:
                meta = json.loads(raw_json)
                logger.info(f"Successfully parsed JSON metadata with keys: {list(meta.keys())}")
            except Exception as e:
                logger.warning(f"Failed to parse JSON metadata: {str(e)}")
                logger.warning(f"Raw JSON content: {raw_json[:200]}...")
                meta = {}
    else:
        # No JSON block found; assume all is markdown
        markdown = text.strip()
        logger.warning("No JSON metadata block found in OpenRouter response")
        
        # Try to extract title from markdown headers
        title_match = re.search(r'^#\s+(.+)$', markdown, re.MULTILINE)
        if title_match:
            meta['meta_title'] = title_match.group(1).strip()

    # Ensure markdown is never None or empty
    if not markdown:
        markdown = "AI automation content for small businesses"
        logger.warning("Empty markdown content, using default")

    # Default meta fields with safe initial values
    default_meta = {
        'meta_title': 'AI Tools for Small Business',
        'meta_description': 'Discover AI automation tools for small businesses',
        'excerpt': 'Learn about AI automation tools to boost your small business',
        'reading_time': 0,
        'category': '',
        'tags': [],
        'keyword_clusters': ['AI tools', 'small business', 'automation'],
        'long_tail_keywords': ['AI tools for small business'],
        'focus_keywords': ['AI tools'],
        'suggested_image_query': 'AI business automation tools'
    }

    # Merge defaults with parsed meta
    meta = {**default_meta, **meta}

    # Normalize reading_time to integer
    if meta['reading_time']:
        if isinstance(meta['reading_time'], str):
            match = re.search(r'\d+', str(meta['reading_time']))
            parsed_reading_time = int(match.group()) if match else 0
            meta['reading_time'] = parsed_reading_time if parsed_reading_time else 0
        elif isinstance(meta['reading_time'], (int, float)):
            meta['reading_time'] = int(meta['reading_time'])
        else:
            meta['reading_time'] = 0
    else:
        meta['reading_time'] = 0

    # Ensure arrays are properly formatted
    for field in ['tags', 'keyword_clusters', 'long_tail_keywords', 'focus_keywords']:
        if not isinstance(meta[field], list):
            meta[field] = []

    result = {"markdown": markdown, **meta}
    logger.info(f"Parsed blog with title: {result.get('meta_title', 'No title')}")
    return result


def get_pexels_image(query: str) -> Dict[str, Any]:
    """
    Fetch image from Pexels API and map to required fields
    Based on the provided image fetching function
    """
    if not query:
        logger.warning("No image query provided")
        return {}
    
    logger.info(f"Fetching image for query: {query}")
    
    url = f'https://api.pexels.com/v1/search?query={quote(query)}&per_page=1'
    headers = {'Authorization': PEXELS_API_KEY}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        photo = data.get('photos', [None])[0]
        
        if not photo:
            logger.warning("No photos found for query")
            return {}
        
        # Map image URLs to required fields
        image_data = {
            'image_url': photo['src'].get('landscape'),
            'meta_image': photo['src'].get('large2x'),
            'og_image': photo['src'].get('landscape'),
            'twitter_image': photo['src'].get('medium')
        }
        
        logger.info("Successfully fetched image data")
        return image_data
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Pexels API error: {str(e)}")
        return {}

def assign_featured_or_trending() -> Dict[str, bool]:
    """
    Assign featured or trending status (mutually exclusive)
    Based on the provided assignment function
    """
    roll = random.random()
    
    if roll < 0.8:
        # 80% neither
        is_featured, is_trending = False, False
    elif roll < 0.9:
        # 10% featured
        is_featured, is_trending = True, False
    else:
        # 10% trending
        is_featured, is_trending = False, True
    
    logger.info(f"Assigned featured: {is_featured}, trending: {is_trending}")
    return {"is_featured": is_featured, "is_trending": is_trending}

def get_category_uuid(category_slug: str, categories: List[Dict]) -> Optional[str]:
    """Get category UUID by slug"""
    category = next((c for c in categories if c.get('slug') == category_slug), None)
    return category.get('id') if category else None

def get_tag_uuids(tag_slugs: List[str], tags: List[Dict]) -> List[str]:
    """Get tag UUIDs by slugs"""
    tag_uuids = []
    for slug in tag_slugs:
        tag = next((t for t in tags if t.get('slug') == slug), None)
        if tag and tag.get('id'):
            tag_uuids.append(tag['id'])
    return tag_uuids

def supabase_insert(table: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Insert single record into Supabase table with better error handling"""
    logger.info(f"Inserting record into {table}")
    logger.debug(f"Data to insert: {json.dumps(data, indent=2, default=str)}")
    
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    headers = {
        **SUPABASE_HEADERS,
        'Prefer': 'return=representation'  # Ensure we get the inserted data back
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        
        # Log response details for debugging
        logger.debug(f"Response status: {response.status_code}")
        logger.debug(f"Response headers: {dict(response.headers)}")
        logger.debug(f"Response text: {response.text}")
        
        response.raise_for_status()
        
        # Handle empty response
        if not response.text.strip():
            raise BlogGenerationError(f"Empty response from Supabase when inserting into {table}")
        
        result = response.json()
        
        # Supabase returns array when using return=representation
        if isinstance(result, list) and len(result) > 0:
            result = result[0]
        
        logger.info(f"Successfully inserted record into {table}")
        return result
        
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP Error inserting into {table}: {e}")
        logger.error(f"Response content: {response.text if 'response' in locals() else 'No response'}")
        raise BlogGenerationError(f"HTTP Error inserting into {table}: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error inserting into {table}: {e}")
        logger.error(f"Response text: {response.text if 'response' in locals() else 'No response'}")
        raise BlogGenerationError(f"Invalid JSON response from {table}: {e}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error inserting into {table}: {e}")
        raise BlogGenerationError(f"Failed to insert into {table}: {e}")


def supabase_insert_multiple(table: str, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    logger.info(f"Inserting {len(data_list)} records into {table}")
    logger.debug(f"Data to insert: {json.dumps(data_list, indent=2, default=str)}")

    url = f"{SUPABASE_URL}/rest/v1/{table}"
    headers = {
        'apikey': SUPABASE_KEY,
        'Authorization': f'Bearer {SUPABASE_KEY}',
        'Content-Type': 'application/json',
        'Prefer': 'return=representation'
    }

    # Defensive: Remove None or empty 'tag_id' or 'post_id' to avoid invalid inserts
    filtered_data = [item for item in data_list if item.get('post_id') and item.get('tag_id')]
    if not filtered_data:
        raise BlogGenerationError("No valid post_tags data to insert.")

    try:
        response = requests.post(url, headers=headers, json=filtered_data)
        response.raise_for_status()
        result = response.json()
        logger.info(f"Successfully inserted {len(filtered_data)} records into {table}")
        return result
    except requests.exceptions.RequestException as e:
        logger.error(f"Error inserting multiple records into {table}: {e}")
        logger.error(f"Response content: {response.text if 'response' in locals() else 'No response'}")
        raise BlogGenerationError(f"Failed to insert multiple records into {table}: {e}")


def send_telegram_message(text: str):
    """Send notification to Telegram"""
    logger.info("Sending Telegram notification")
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': text,
        'parse_mode': 'Markdown'
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        logger.info("Telegram notification sent successfully")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send Telegram notification: {str(e)}")

def generate_blog_post():
    """Main function to generate and publish a blog post"""
    try:
        logger.info("=== Starting Blog Generation Workflow ===")
        
        # Step 1: Fetch data from Supabase
        logger.info("Step 1: Fetching data from Supabase")
        posts = supabase_get_all('posts')
        tags = supabase_get_all('tags')
        categories = supabase_get_all('categories')
        
        # Step 2: Generate content idea with Perplexity
        logger.info("Step 2: Generating content idea with Perplexity")
        perplexity_prompt = build_perplexity_prompt(posts, tags, categories)
        perplexity_response = call_perplexity_api(perplexity_prompt)
        perplexity_data = parse_perplexity_output(perplexity_response)
        
        # Step 3: Generate blog content with OpenRouter
        logger.info("Step 3: Generating blog content with OpenRouter")
        openrouter_messages = build_openrouter_messages(perplexity_data)
        openrouter_response = call_openrouter_chat(openrouter_messages)
        parsed_blog = parse_openrouter_output(openrouter_response)
        
        # Step 4: Generate unique slug
        logger.info("Step 4: Generating unique slug")

        # Safely extract title for slug with multiple fallbacks
        slug_source = None

        # Try meta_title first
        if parsed_blog.get('meta_title') and parsed_blog['meta_title'].strip():
            slug_source = parsed_blog['meta_title']
        # Then try first 50 chars of markdown (safely)
        elif parsed_blog.get('markdown') and parsed_blog['markdown']:
            slug_source = parsed_blog['markdown'][:50]
        # Then try the topic from perplexity
        elif perplexity_data.get('topic'):
            slug_source = perplexity_data['topic']
        # Finally, use a default
        else:
            slug_source = 'ai-business-automation-post'

        logger.info(f"Using slug source: {slug_source[:50]}...")
        slug = slugify(slug_source)
        logger.info(f"Generated slug: {slug}")
        
        # Step 5: Get images from Pexels
        logger.info("Step 5: Fetching images from Pexels")

        # Safely extract image query with better fallbacks
        image_query = None

        # Try suggested_image_query first
        if parsed_blog.get('suggested_image_query'):
            image_query = parsed_blog['suggested_image_query']
        # Then try meta_title
        elif parsed_blog.get('meta_title'):
            image_query = parsed_blog['meta_title']
        # Then try first focus_keyword (safely)
        elif parsed_blog.get('focus_keywords') and len(parsed_blog['focus_keywords']) > 0:
            image_query = parsed_blog['focus_keywords'][0]
        # Then try first keyword_cluster (safely)
        elif parsed_blog.get('keyword_clusters') and len(parsed_blog['keyword_clusters']) > 0:
            image_query = parsed_blog['keyword_clusters'][0]
        # Finally, use topic from perplexity as fallback
        else:
            image_query = perplexity_data.get('topic', 'AI business automation')

        logger.info(f"Using image query: {image_query}")
        image_data = get_pexels_image(image_query)

        
        # Step 6: Assign featured or trending status
        logger.info("Step 6: Assigning featured/trending status")
        featured_trending = assign_featured_or_trending()
        
        # Step 7: Get UUIDs for category and tags
        logger.info("Step 7: Resolving category and tag UUIDs")
        category_id = get_category_uuid(parsed_blog.get('category', ''), categories)
        tag_ids = get_tag_uuids(parsed_blog.get('tags', []), tags)
        
        # Step 8: Prepare and insert post data
        logger.info("Step 8: Creating blog post")

        # Generate canonical URL from slug
        canonical_url = f"https://yourdomain.com/{slug}"  # Replace with your actual domain

        # Prepare comprehensive post data with all required fields
        post_data = {
            # Basic content fields
            'title': parsed_blog.get('meta_title', '')[:255],  # Ensure within length limits
            'slug': slug,
            'excerpt': parsed_blog.get('excerpt', '')[:500],  # Limit excerpt length
            'content': parsed_blog['markdown'],
            
            # Author and timestamps
            'author_id': '550e8400-e29b-41d4-a716-446655440003',  # Hardcoded as requested
            'published_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            
            # Category and metrics
            'category_id': category_id,
            'image_url': image_data.get('image_url'),
            'reading_time': parsed_blog.get('reading_time', estimate_reading_time(parsed_blog['markdown'])),
            'views': random.randint(500, 1000),  # Random views as requested
            'is_featured': featured_trending['is_featured'],
            'is_trending': featured_trending['is_trending'],
            'status': 'published',  # Set to published as requested
            
            # SEO Meta fields
            'meta_title': parsed_blog.get('meta_title', '')[:255],
            'meta_description': parsed_blog.get('meta_description', '')[:500],
            'meta_image': image_data.get('meta_image'),
            'canonical_url': canonical_url,
            
            # Open Graph fields
            'og_title': parsed_blog.get('meta_title', '')[:255],
            'og_description': parsed_blog.get('meta_description', '')[:500],
            'og_image': image_data.get('og_image'),
            
            # Twitter fields
            'twitter_title': parsed_blog.get('meta_title', '')[:255],
            'twitter_description': parsed_blog.get('meta_description', '')[:500],
            'twitter_image': image_data.get('twitter_image'),
            
            # Keyword arrays (ensure they're proper arrays)
            'keyword_clusters': parsed_blog.get('keyword_clusters', []) or [],
            'long_tail_keywords': parsed_blog.get('long_tail_keywords', []) or [],
            'focus_keywords': parsed_blog.get('focus_keywords', []) or [],
            
            # SEO JSON for additional AI-generated data
            'seo_json': {
                'ai_generated': True,
                'generation_timestamp': datetime.now().isoformat(),
                'perplexity_topic': perplexity_data.get('topic', ''),
                'selected_tags': parsed_blog.get('tags', []),
                'selected_category': parsed_blog.get('category', ''),
                'image_query_used': image_query or '',
                'additional_keywords': parsed_blog.get('keyword_clusters', [])
            }
        }

        # Remove None values to avoid Supabase issues
        post_data = {k: v for k, v in post_data.items() if v is not None}

        # Ensure category_id exists
        if not category_id:
            logger.warning("No category ID found, will insert without category")
            post_data.pop('category_id', None)

        # Log the data being inserted (for debugging)
        logger.debug(f"Post data to insert: {json.dumps(post_data, indent=2, default=str)}")

        # Insert post
        try:
            inserted_post = supabase_insert('posts', post_data)
            post_id = inserted_post.get('id')
            
            if not post_id:
                raise BlogGenerationError("Failed to get post ID after insertion")
                
            logger.info(f"Successfully created post with ID: {post_id}")
            
        except Exception as e:
            logger.error(f"Failed to insert post: {str(e)}")
            raise BlogGenerationError(f"Post insertion failed: {str(e)}")

        
        # Step 9: Create post-tag relationships
        logger.info("Step 9: Creating post-tag relationships")
        if tag_ids:
            post_tags_data = [{'post_id': post_id, 'tag_id': tag_id} for tag_id in tag_ids]
            supabase_insert_multiple('post_tags', post_tags_data)
        
        # Step 10: Send success notification
        logger.info("Step 10: Sending success notification")
        notification_text = f"""✅ *New Blog Post Created Successfully!*

📝 **Title:** {parsed_blog['meta_title']}
🔗 **Slug:** `{slug}`
📂 **Category:** {parsed_blog.get('category', 'None')}
🏷️ **Tags:** {', '.join(parsed_blog.get('tags', []))}
📊 **Reading Time:** {post_data['reading_time']} min
⭐ **Featured:** {'Yes' if featured_trending['is_featured'] else 'No'}
🔥 **Trending:** {'Yes' if featured_trending['is_trending'] else 'No'}

*Blog generation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"""
        
        send_telegram_message(notification_text)
        
        logger.info("=== Blog Generation Workflow Completed Successfully ===")
        
    except Exception as e:
        logger.error(f"Blog generation failed: {str(e)}")
        
        # Send error notification
        error_text = f"""❌ *Blog Generation Failed*

**Error:** {str(e)}
**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Please check the logs for more details."""
        
        try:
            send_telegram_message(error_text)
        except:
            logger.error("Failed to send error notification to Telegram")

def schedule_blog_generation():
    """Set up scheduled blog generation (twice per week)"""
    logger.info("Setting up blog generation schedule")
    
    # Schedule for Tuesday and Friday at 9:00 AM
    schedule.every().tuesday.at("09:00").do(generate_blog_post)
    schedule.every().friday.at("09:00").do(generate_blog_post)
    
    logger.info("Blog generation scheduled for Tuesday and Friday at 9:00 AM")
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    # Validate environment variables
    required_vars = [
        'SUPABASE_URL', 'SUPABASE_KEY', 'PERPLEXITY_API_KEY', 
        'OPENROUTER_API_KEY', 'PEXELS_API_KEY', 
        'TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        exit(1)
    
    logger.info("Starting Blog Generation Service")
    
    # For testing, you can run immediately:
    # generate_blog_post()
    
    # For production, run the scheduler:
    schedule_blog_generation()
