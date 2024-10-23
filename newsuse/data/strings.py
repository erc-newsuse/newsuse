import re
from html.parser import HTMLParser
from io import StringIO
from types import SimpleNamespace
from urllib.parse import urlparse

from find_domains import find_domains

__all__ = (
    "remove_html",
    "remove_domains",
    "remove_urls",
    "remove_emails",
    "remove_trailing_hashtags",
    "sanitize",
)

_url_schemes = (
    "file",
    "ftp",
    "http",
    "https",
    "imap",
    "irc",
    "nntp",
    "acap",
    "icap",
    "mtqp",
    "wss",
)

_rx = SimpleNamespace(
    multispace=re.compile(r" +"),
    hashtags=re.compile(r"\s|(?=#)", re.MULTILINE),
    noncontent=re.compile(r"^[\s\W\d_]*$", re.IGNORECASE | re.MULTILINE),
    urlschemes=re.compile(rf"(?={"|".join(_url_schemes)})", re.IGNORECASE | re.MULTILINE),
)


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
    return s.get_data().strip()


def remove_domains(text: str) -> str:
    """Remove domain names from ``text``.

    Examples
    --------
    >>> text = "This is news on cnn.com and wp.pl."
    >>> remove_domains(text)
    'This is news on  and .'
    """
    domains = "|".join(re.escape(d.strip()) for d in find_domains(text))
    return re.sub(domains, r"", text).strip()


def remove_urls(text: str) -> str:
    """Remove URLs from ``text``.

    Examples
    --------
    >>> text = "This is a text with a URL https://www.java2blog.com/ to remove."
    >>> remove_urls(text)
    'This is a text with a URL  to remove.'
    >>> text = "Daniel Passent: Co prawda do Indonezji daleko, ale na Węgry blisko."
    >>> remove_urls(text)
    'Daniel Passent: Co prawda do Indonezji daleko, ale na Węgry blisko.'
    >>> text = "Pętla indukcyjna...https://www.gosc.pl/doc/7451826.Nie-krzycz-tak"
    >>> remove_urls(text)
    'Pętla indukcyjna...'
    """
    urls = []
    text = _rx.urlschemes.sub(r" ", text)
    for token in text.split():
        token = token.strip()
        try:
            url = urlparse(token)
        except ValueError:
            urls.append(re.escape(token))
            continue
        if url.scheme in _url_schemes and url.netloc:
            urls.append(re.escape(token))
    return re.sub("|".join(urls), r"", text).strip()


def remove_emails(text: str) -> str:
    """Remove emails from ``text``.

    Examples
    --------
    >>> text = "This is our address: hey@there.uk.com."
    >>> remove_emails(text)
    'This is our address: .'
    """
    return re.sub(r"[.\w]+@[\w.]*[\w]+", r"", text).strip()


def remove_trailing_hashtags(text: str) -> str:
    """Remove trailing hashtags.

    Examples
    --------
    >>> text = "Hej hej #tag1 hej#tag1 #tag2"
    >>> remove_trailing_hashtags(text)
    'Hej hej #tag1 hej'
    """
    trailing_hashtags = []
    for token in reversed(_rx.hashtags.split(text)):
        if not token:
            continue
        if token.startswith("#"):
            trailing_hashtags.append(token)
        else:
            break
    text = text[::-1]
    for tag in trailing_hashtags:
        text = text.replace(tag[::-1], "", 1)
    return text[::-1].strip()


def nullify_noncontent(text: str) -> str:
    """Nullify non-content texts.

    Examples
    --------
    >>> nullify_noncontent("0")
    ''
    >>> nullify_noncontent("138_32")
    ''
    >>> nullify_noncontent("Hey 1_329")
    'Hey 1_329'
    """
    return _rx.noncontent.sub(r"", text).strip()


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
    trailing_hashtags: bool = True,
    noncontent: bool = True,
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
    if trailing_hashtags:
        text = remove_trailing_hashtags(text)
    if noncontent:
        text = nullify_noncontent(text)
    if multispaces:
        text = remove_multispaces(text)
    return text.strip()
