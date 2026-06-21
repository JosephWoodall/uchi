import json
import os
import pickle
import random
import urllib.request

_CACHE_DIR = os.path.dirname(__file__)


def _cache_path(name: str) -> str:
    return os.path.join(_CACHE_DIR, f'_{name}_cache.pkl')


def _load_cache(name: str):
    p = _cache_path(name)
    if os.path.exists(p):
        with open(p, 'rb') as f:
            return pickle.load(f)
    return None


def _save_cache(name: str, data) -> None:
    with open(_cache_path(name), 'wb') as f:
        pickle.dump(data, f)


def load_airline_passengers() -> list[float]:
    """Monthly airline passengers 1949–1960 (144 obs, strong seasonality)."""
    try:
        import statsmodels.api as sm  # type: ignore
        data = sm.datasets.get_rdataset("AirPassengers", "datasets").data
        return data["AirPassengers"].tolist()
    except ImportError:
        return [
            112,118,132,129,121,135,148,148,136,119,104,118,
            115,126,141,135,125,149,170,170,158,133,114,140,
            145,150,178,163,172,178,199,199,184,162,146,166,
            171,180,193,181,183,218,230,242,209,191,172,194,
            196,196,236,235,229,243,264,272,237,211,180,201,
            204,188,235,227,234,264,302,293,259,229,203,229,
            242,233,267,269,270,315,364,347,312,274,237,278,
            284,277,317,313,318,374,413,405,355,306,271,306,
            315,301,356,348,355,422,465,467,404,347,305,336,
            340,318,362,348,363,435,491,505,404,359,310,337,
            360,342,406,396,420,472,548,559,463,407,362,405,
            417,391,419,461,472,535,622,606,508,461,390,432,
        ]


def load_gutenberg_text(n_chars: int | None = 1500) -> list[str]:
    """
    Character-level sequence from Alice in Wonderland (Project Gutenberg).
    n_chars=None returns all available alpha/space characters (~130K).
    Results are cached locally after the first download.
    """
    cached = _load_cache('alice')
    if cached is None:
        url = "https://www.gutenberg.org/files/11/11-0.txt"
        req = urllib.request.Request(url, headers={"User-Agent": "markov-exploration/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        start = raw.find("CHAPTER I")
        text  = raw[start:] if start != -1 else raw
        cached = [c.lower() for c in text if c.isalpha() or c == " "]
        _save_cache('alice', cached)
    return cached if n_chars is None else cached[:n_chars]


def load_moby_dick(n_chars: int | None = 50_000) -> list[str]:
    """
    Character-level sequence from Moby Dick (Project Gutenberg #2701).
    ~550K alpha/space chars available; default uses 50K.
    Results are cached locally after the first download.
    """
    cached = _load_cache('mobydick')
    if cached is None:
        url = "https://www.gutenberg.org/cache/epub/2701/pg2701.txt"
        req = urllib.request.Request(url, headers={"User-Agent": "markov-exploration/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        start = raw.find("CHAPTER 1")
        if start == -1:
            start = raw.find("CHAPTER I")
        text  = raw[start:] if start != -1 else raw
        cached = [c.lower() for c in text if c.isalpha() or c == " "]
        _save_cache('mobydick', cached)
    return cached if n_chars is None else cached[:n_chars]


def load_dna_sequence(n_bases: int | None = 1500) -> list[str]:
    """
    Bacteriophage lambda genome nucleotides (NCBI accession J02459, 48,502 bp).
    n_bases=None returns the full genome.
    Results are cached locally after the first download.
    """
    cached = _load_cache('dna')
    if cached is None:
        url = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            "?db=nucleotide&id=J02459&rettype=fasta&retmode=text"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "markov-exploration/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("ascii", errors="ignore")
        seq = "".join(line.strip() for line in raw.splitlines() if not line.startswith(">"))
        cached = list(seq.lower())
        _save_cache('dna', cached)
    return cached if n_bases is None else cached[:n_bases]


def load_weather_events(n_days: int | None = 500) -> list[str]:
    """Daily WMO weather codes for NYC 2020–2021, coarsened to 5 categories."""
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        "?latitude=40.7128&longitude=-74.0060"
        "&start_date=2020-01-01&end_date=2021-06-30"
        "&daily=weather_code&timezone=America%2FNew_York"
    )
    with urllib.request.urlopen(url, timeout=15) as resp:
        data = json.loads(resp.read().decode())
    codes = data["daily"]["weather_code"]
    if n_days is not None:
        codes = codes[:n_days]
    return [_coarsen(c) for c in codes]


def _coarsen(code: int) -> str:
    if code <= 1:
        return "clear"
    elif code <= 3:
        return "cloudy"
    elif code <= 48:
        return "overcast"
    elif code <= 82:
        return "rain"
    else:
        return "storm"


def random_integers(n: int = 500, low: int = 0, high: int = 9, seed: int | None = None) -> list[int]:
    """Uniformly random integers — the unpredictable baseline."""
    rng = random.Random(seed) if seed is not None else random
    return [rng.randint(low, high) for _ in range(n)]


def load_electricity(n: int | None = None) -> list[int]:
    """
    Electricity market dataset (Harries 1999) — 45,312 binary steps.
    Target: 0=DOWN, 1=UP (NSW electricity price movement relative to a
    24-hour moving average).  Standard concept-drift benchmark: consumption
    patterns and market structure shift across a 2-year window (May 1996 –
    Dec 1998).  Source: OpenML dataset #151 via sklearn.
    """
    cache = os.path.join(_CACHE_DIR, '_elec_cache.pkl')
    if os.path.exists(cache):
        with open(cache, 'rb') as f:
            labels = pickle.load(f)
    else:
        from sklearn.datasets import fetch_openml
        data   = fetch_openml('electricity', version=1, as_frame=True, parser='auto')
        labels = [1 if v == 'UP' else 0 for v in data.target.tolist()]
        with open(cache, 'wb') as f:
            pickle.dump(labels, f)
    return labels[:n] if n is not None else labels
