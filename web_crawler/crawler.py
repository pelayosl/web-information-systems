import os
import time as t
from warcio.capture_http import capture_http
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import urllib.robotparser
import shutil

headers = {'user-agent': 'SIWbot'}
DEPTH_LIMIT = 5
ROBOTS_BLOCKED_COUNT = 0

'''
Función principal que procesa las semillas y controla el flujo del crawler.
'''
def process_seeds(file_name, max_num, seconds, algorithm, robot_compliance):
    pages_crawled = 0
    url_list = []
    if os.path.exists('pages'):
        shutil.rmtree('pages')

    prepare_links_file(max_num, seconds, algorithm, robot_compliance)

    with open(file_name) as file:
        url_list = file.readlines()
    
    if algorithm.lower() == 'bfs':
        print('Starting BFS crawl...')
        pages_crawled = crawlbfs(url_list, max_num, seconds, pages_crawled, robot_compliance)
    elif algorithm.lower() == 'dfs':
        for seed in url_list:
            if(pages_crawled >= max_num):
                break
            print(f'Starting DFS crawl from seed: {seed.strip()}')
            pages_crawled = crawldfs(seed.strip(), max_num, seconds, pages_crawled, set(), robot_compliance, 0)
    else:
        print('Unknown algorithm specified. Please use "bfs" or "dfs".')
        return

    print('Crawling finished.')

    with open("links.txt", "a") as links:
        links.write(f"\nTotal pages crawled: {pages_crawled}\n")
        links.write(f"Total URLs blocked by robots.txt: {ROBOTS_BLOCKED_COUNT}\n")

'''
Prepara el archivo links.txt con la configuración inicial del crawler.
'''
def prepare_links_file(max_num, seconds, algorithm, robot_compliance):
    os.makedirs('pages', exist_ok=True)
    with open("links.txt", "w") as links:
        links.write(f"Max number of pages to crawl: {max_num}\n")
        links.write(f"Crawl delay between requests: {seconds} seconds\n")
        links.write(f"Crawling algorithm: {algorithm.upper()}\n")
        links.write(f"Robots.txt compliance: {'enabled' if robot_compliance else 'disabled'}\n\n")
        links.write("Crawled Links:\n")

'''
Verifica si se puede acceder a una URL según el archivo robots.txt.
'''
def can_fetch(url):
    global ROBOTS_BLOCKED_COUNT
    rp = urllib.robotparser.RobotFileParser()
    robots_url = f"{url}/robots.txt"
    rp.set_url(robots_url)
    try:
        rp.read()
        fetchable = rp.can_fetch(headers['user-agent'], url)
        if(not fetchable):
            print(f"Blocked by robots.txt: {url}")
            ROBOTS_BLOCKED_COUNT += 1
        return fetchable
    except:
        return False
    

'''
Utiliza DFS para rastrear páginas web.
'''
def crawldfs(url, max_num, seconds, pages_crawled, explored, robot_compliance, depth):
    domain = urlparse(url).netloc

    if pages_crawled >= max_num or url in explored or depth >= DEPTH_LIMIT:
        return pages_crawled
    
    pages_crawled, soup = fetch_url(url, pages_crawled, domain, seconds, depth)
    if soup is None:
        return pages_crawled
    else:
        explored.add(url)

    links = [a['href'] for a in soup.find_all('a', href=True)]

    for link in links:
        if pages_crawled >= max_num:
            break
        l = normalize_link(url, link)
        if not robot_compliance:
            pages_crawled = crawldfs(l, max_num, seconds, pages_crawled, explored, robot_compliance, depth + 1)
            continue
        if can_fetch(l):
            pages_crawled = crawldfs(l, max_num, seconds, pages_crawled, explored, robot_compliance, depth + 1)
    return pages_crawled

'''
Utiliza BFS para rastrear páginas web.
'''
def crawlbfs(urls, max_num, seconds, pages_crawled, robot_compliance):
    frontier = []
    for url in urls:
        frontier.append((url.strip(), 0))
    explored = set()
    
    while frontier and pages_crawled < max_num:
        current_url, depth = frontier.pop(0)
        domain = urlparse(current_url).netloc
        
        if current_url not in explored:
            explored.add(current_url)

            pages_crawled, soup = fetch_url(current_url, pages_crawled, domain, seconds, depth)
            if soup is None:
                continue

            links = [a['href'] for a in soup.find_all('a', href=True)]
            for link in links:
                l = normalize_link(current_url, link)
                if l not in explored:
                    if not robot_compliance:
                        frontier.append((l, depth + 1))
                        continue
                    if can_fetch(l):
                        frontier.append((l, depth + 1))

    return pages_crawled

'''
Realiza la solicitud HTTP para obtener el contenido de una URL y guarda la respuesta en un archivo WARC
siempre y cuando el contenido sea HTML.
'''
def fetch_url(url, pages_crawled, domain, seconds, depth):
    soup = None
    try:
        if is_html(url):
            warc_file = f'pages/{domain}.warc.gz'
            pages_crawled += 1

            with capture_http(warc_file, gzip=True):
                html = requests.get(url, headers=headers)
                print(f"Crawled ({pages_crawled}): {url} at depth {depth}")
                with open("links.txt", "a") as links:
                    links.write(f"\t{url}\n")
            
            soup = BeautifulSoup(html.text, 'html.parser')
            t.sleep(seconds)
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
    return pages_crawled, soup

'''
Verifica si el contenido de la URL es HTML.
'''
def is_html(url):
    head = requests.head(url, headers=headers, allow_redirects=True, timeout=5)
    content_type = head.headers.get("Content-Type", "").lower()
    val = "text/html" in content_type
    if(not val):
        print(f"Non-compatible content: {content_type}")
    return val

def normalize_link(url, link):
    if link.startswith("/") or link.startswith("#"):
        return urljoin(url, link)
    return link

file_name = 'crawler/seeds3.txt'
algorithm = 'bfs'  # dfs - bfs
robot_compliance = True
process_seeds(file_name, 6, 2, algorithm, robot_compliance)