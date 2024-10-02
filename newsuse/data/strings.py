import re
from html.parser import HTMLParser
from io import StringIO
from urllib.parse import urlparse

from find_domains import find_domains

from newsuse.types import Namespace

__all__ = ("remove_html", "remove_domains", "remove_urls", "remove_emails", "sanitize")


_rx = Namespace(multispace=re.compile(r" +"))


class HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = StringIO()

    def handle_data(self, data: str) -> None:
        self.text.write(data)

    def get_data(self) -> str:
        return self.text.getvalue()


def remove_html(text: str) -> str:
    """Remove HTML tags.

    Examples
    --------
    >>> text = 'Hey this is <a href="dupa.html"> and <div>with some data</div>'
    >>> remove_html(text)
    'Hey this is  and with some data'
    """
    s = HTMLStripper()
    s.feed(text)
    return s.get_data()


def remove_domains(text: str) -> str:
    """Remove domain names from ``text``.

    Examples
    --------
    >>> text = "This is news on cnn.com and wp.pl."
    >>> remove_domains(text)
    'This is news on  and .'
    """
    domains = "|".join(re.escape(d.strip()) for d in find_domains(text))
    return re.sub(domains, r"", text)


def remove_urls(text: str) -> str:
    """Remove URLs from ``text``.

    Examples
    --------
    >>> text = "This is a text with a URL https://www.java2blog.com/ to remove."
    >>> remove_urls(text)
    'This is a text with a URL  to remove.'
    """
    urls = []
    for token in text.split():
        token = token.strip()
        if urlparse(token).scheme:
            urls.append(re.escape(token))
    return re.sub("|".join(urls), r"", text)


def remove_emails(text: str) -> str:
    """Remove emails from ``text``.

    Examples
    --------
    >>> text = "This is our address: hey@there.uk.com."
    >>> remove_emails(text)
    'This is our address: .'
    """
    return re.sub(r"[.\w]+@[\w.]*[\w]+", r"", text)


def remove_multispaces(text: str) -> str:
    """Remove multispaces from ``text``.

    Examples
    --------
    >>> text = "This is a long   space."
    >>> remove_multispaces(text)
    'This is a long space.'
    """
    return _rx.multispace.sub(r" ", text)


def sanitize(
    text: str,
    *,
    html: bool = True,
    urls: bool = True,
    emails: bool = True,
    domains: bool = True,
    multispaces: bool = True,
) -> str:
    """Sanitize strings by removing non-natural text elements.

    Parameters
    ----------
    string
        String to sanitize.
    html, domains, urls, emails
        Should they be removed.
    """
    text = text.strip()
    if html:
        text = remove_html(text)
    if urls:
        text = remove_urls(text)
    if emails:
        text = remove_emails(text)
    if domains:
        text = remove_domains(text)
    if multispaces:
        text = remove_multispaces(text)
    return text.strip()
