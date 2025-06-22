import argparse
import requests
import csv
import math
import time
import random
import os
import glob
from datetime import datetime
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from fake_useragent import UserAgent
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import threading

# === Configuration ===
BASE_URL = "https://www.arquivoestado.sp.gov.br/web/acervo/memoria_imigrante/pesquisaLivroDelegaciaEstrangeiro"
MAX_RETRIES = 3
MAX_WORKERS = 5
POLITE_DELAY_RANGE = (1.0, 2.5)

PROXIES = []  # Optional: e.g., ["http://user:pass@proxy.example.com:port"]

ua = UserAgent()
thread_local = threading.local()

def parse_args():
    parser = argparse.ArgumentParser(description="Scrape Delegacia do Estrangeiro records by nationality.")
    parser.add_argument("--list-only", "-l", action="store_true")
    parser.add_argument("--scrape-only", "-s", action="store_true")
    parser.add_argument("--resume-from", "-r", type=str)
    parser.add_argument("--resume-all", action="store_true", help="Resume from all .csv files in current directory except falhas_scraping*")
    return parser.parse_args()

def build_form(nacionalidade):
    return urlencode({
        "nome": "",
        "nome_mae": "",
        "nome_pai": "",
        "nacionalidade": nacionalidade,
        "ano_entrada": "",
        "carteira_identidade": "",
        "numero_registro": ""
    })

def select_pages(total_pages, num_pages):
    if total_pages == num_pages:
        return list(range(1, total_pages + 1))
    step = total_pages / (num_pages + 1)
    pages = [math.floor(step * (i + 1)) for i in range(num_pages + 2)]
    clean = [p for p in sorted(set(pages)) if 1 < p < total_pages]
    return clean[:num_pages]

def get_headers():
    return {
        "User-Agent": ua.random,
        "X-Requested-With": "XMLHttpRequest",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "Origin": "https://www.arquivoestado.sp.gov.br",
        "Referer": "https://www.arquivoestado.sp.gov.br/web/acervo/solicitacao_certidoes/delegacia"
    }

def get_proxy():
    return {"http": random.choice(PROXIES), "https": random.choice(PROXIES)} if PROXIES else None

def get_session():
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
    return thread_local.session

def fetch_total_pages(nacionalidade):
    session = get_session()
    headers = get_headers()
    data = {"frm": build_form(nacionalidade), "page": 1, "limit": 10}
    response = session.post(BASE_URL, headers=headers, data=data, timeout=20)
    response.raise_for_status()
    return response.json()["total_paginas"]

@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=0.5, min=1, max=10),
    retry=retry_if_exception_type((requests.exceptions.RequestException,))
)
def fetch_page(nacionalidade, page):
    session = get_session()
    headers = get_headers()
    proxy = get_proxy()
    data = {
        "frm": build_form(nacionalidade),
        "page": page,
        "limit": 10
    }
    response = session.post(BASE_URL, headers=headers, data=data, proxies=proxy, timeout=20)
    response.raise_for_status()
    return response.json()

def fetch_page_safe(nacionalidade, page):
    time.sleep(random.uniform(*POLITE_DELAY_RANGE))  # single delay before initial request
    try:
        data = fetch_page(nacionalidade, page)
        return nacionalidade, page, data
    except Exception as e:
        return nacionalidade, page, {"error": str(e)}

def write_to_csv(filename, rows, fieldnames):
    with open(filename, "a", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for row in rows:
            writer.writerow(row)

def write_failures(filename, nat, page, error):
    with open(filename, "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([nat, page, error])

def load_scraped_pages(path):
    scraped = set()
    print(f"Loading already scraped pages from {path}...")
    try:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                nat = row.get("nacionalidade_query")
                page = row.get("pagina")
                if nat and page:
                    try:
                        scraped.add((nat.strip(), int(page)))
                    except ValueError:
                        continue
    except FileNotFoundError:
        print(f"File {path} not found. Starting fresh.")
    return scraped

def load_scraped_from_all():
    all_scraped = set()
    for file in glob.glob("*.csv"):
        if file.startswith("falhas_scraping"):
            continue
        all_scraped.update(load_scraped_pages(file))
    return all_scraped

def main(targets, list_only=False, scrape_only=False, resume_from=None, resume_all=False):
    nacionalidade_info = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_csv = f"delegacia_estrangeiro_scraped_{timestamp}.csv"
    fail_csv = f"falhas_scraping_{timestamp}.csv"
    fieldnames = None

    if resume_from:
        already_scraped = load_scraped_pages(resume_from)
    elif resume_all:
        print("Resuming from all available output .csv files...")
        already_scraped = load_scraped_from_all()
    else:
        already_scraped = set()

    if not scrape_only:
        print("Fetching total pages for each nationality...")
    for nat, (num_pages, total_hint) in targets.items():
        if total_hint is not None:
            nacionalidade_info[nat] = total_hint
            print(f"{nat}: total informado = {total_hint} páginas")
        elif not scrape_only:
            try:
                total = fetch_total_pages(nat)
                nacionalidade_info[nat] = total
                print(f"{nat}: total encontrado = {total} páginas")
            except Exception as e:
                print(f"Erro ao buscar '{nat}': {e}")
                nacionalidade_info[nat] = 0

    if list_only:
        print("Listagem concluída.")
        return

    if not scrape_only:
        proceed = input("Deseja continuar com a raspagem? [y/N]: ").strip().lower()
        if proceed != 'y':
            print("Abortado.")
            return

    with open(fail_csv, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["nacionalidade", "pagina", "erro"])

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for nat, (num_pages, _) in targets.items():
            total_pages = nacionalidade_info.get(nat)
            if not total_pages:
                continue
            pages = select_pages(total_pages, num_pages)
            for p in pages:
                if (nat, p) in already_scraped:
                    continue
                futures.append(executor.submit(fetch_page_safe, nat, p))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Scraping"):
            nat, page, result = future.result()

            if "error" in result:
                print(f"[{nat} - pág. {page}] ERRO: {result['error']}")
                write_failures(fail_csv, nat, page, result["error"])
                continue

            if not result.get("status") or not result.get("dados"):
                continue

            rows = result["dados"]
            for r in rows:
                r["nacionalidade_query"] = nat
                r["pagina"] = page

            if fieldnames is None:
                fieldnames = list(rows[0].keys())
                with open(output_csv, "w", newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()

            write_to_csv(output_csv, rows, fieldnames)

    print(f"\nRaspagem concluída. Dados salvos em: {output_csv}")
    print(f"Falhas registradas em: {fail_csv}")

if __name__ == "__main__":
    args = parse_args()

    targets = {
        "francesa": (1842, 1842),
        "italiana": (16740, 16740),
        "portuguesa": (30866, 30866),
        "espanhola": (15789, 15789),
        "alemã": (7157, 7157),
        "polonesa": (1329, 1329),
        "ucraniana": (150, 150),
        "russa": (470, 470),
        "japonesa": (20642, 20642),
        "libanesa": (2424, 2424),
        "siria": (908, 908),
        "israel": (932, 932)
    }

    main(
        targets,
        list_only=args.list_only,
        scrape_only=args.scrape_only,
        resume_from=args.resume_from,
        resume_all=args.resume_all
    )
