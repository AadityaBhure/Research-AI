import asyncio
import ipaddress
import re
import socket
from urllib.parse import urljoin, urlsplit

import httpx
import pymupdf

from app.models import ExtractedPage
from app.services.errors import ServiceError


def validate_pdf(data: bytes, maximum: int) -> None:
    if not data or len(data) > maximum:
        raise ServiceError('PDF is empty or exceeds the 30 MB limit.', 400)
    if not data[:1024].lstrip().startswith(b'%PDF-'):
        raise ServiceError('The file is not a valid PDF. Upload the actual paper PDF.', 400)


def extract_pages(data: bytes) -> list[ExtractedPage]:
    try:
        with pymupdf.open(stream=data, filetype='pdf') as doc:
            if doc.needs_pass:
                raise ServiceError('Password-protected PDFs are not supported.', 400)
            if len(doc) > 1000:
                raise ServiceError('Please upload a PDF with at most 1,000 pages.', 400)
            pages = []
            for index, page in enumerate(doc):
                text = page.get_text('text', sort=True)
                text = re.sub(r'[^\S\n]+', ' ', text)
                text = re.sub(r'\n{3,}', '\n\n', text).strip()
                if text:
                    pages.append(ExtractedPage(page_number=index + 1, text=text))
            if not pages:
                raise ServiceError('This PDF has no extractable text. Scanned PDFs need OCR, which is outside V1.', 400)
            return pages
    except ServiceError:
        raise
    except Exception:
        raise ServiceError('PDF text extraction failed. Try a different PDF.', 400) from None


async def public_target(url: str) -> tuple[str, str]:
    """Resolve and pin a public IP, preventing redirects/DNS rebinding into private services."""
    try:
        parsed = urlsplit(url)
        if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.port not in (None, 443):
            raise ValueError()
        host = parsed.hostname.encode('idna').decode('ascii')
        addresses = await asyncio.get_running_loop().getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        ips = [ipaddress.ip_address(item[4][0]) for item in addresses]
        if not ips or any(not ip.is_global for ip in ips):
            raise ValueError()
        ip = next((ip for ip in ips if ip.version == 4), ips[0])
        return str(httpx.URL(url).copy_with(host=str(ip))), host
    except (ValueError, OSError, UnicodeError):
        raise ServiceError('PDF URL must point to a publicly accessible HTTPS server.', 400) from None


async def download_pdf(url: str, maximum: int) -> bytes:
    # No ambient proxies or service credentials are sent to arbitrary paper hosts.
    async with httpx.AsyncClient(timeout=30, follow_redirects=False, trust_env=False) as client:
        try:
            for _ in range(6):
                pinned, host = await public_target(url)
                async with client.stream('GET', pinned, headers={'Host': host, 'User-Agent': 'ResearchAI/1.0'},
                                         extensions={'sni_hostname': host}) as response:
                    if response.status_code in (301, 302, 303, 307, 308):
                        url = urljoin(url, response.headers.get('location', ''))
                        continue
                    response.raise_for_status()
                    content_type = response.headers.get('content-type', '').split(';')[0].lower()
                    if content_type and content_type not in ('application/pdf', 'application/octet-stream', 'binary/octet-stream'):
                        raise ServiceError('The download returned a webpage rather than a PDF. Please upload the PDF manually.', 400)
                    length = response.headers.get('content-length')
                    if length and int(length) > maximum:
                        raise ServiceError('PDF exceeds the 30 MB limit.', 400)
                    data = bytearray()
                    async for block in response.aiter_bytes():
                        data.extend(block)
                        if len(data) > maximum:
                            raise ServiceError('PDF exceeds the 30 MB limit.', 400)
                    validate_pdf(bytes(data), maximum)
                    return bytes(data)
            raise ServiceError('PDF download redirected too many times.', 400)
        except (httpx.HTTPError, ValueError):
            raise ServiceError('PDF could not be downloaded. Please upload it manually.', 400) from None
