"""Short usage tips shown in turn while `decider-mlx fetch` downloads the weights."""
from __future__ import annotations

TIPS = (
    ('Array batching',
     'POST a JSON array of request bodies to /v1/decisions to get a same-order array back.',
     'An invalid item gets {"error": ...} in its place; the others are answered.',
     'Arrays of 64 or more requests are packed into fixed-size batches internally (fastest for bulk work).'),
    ('Isolated mode',
     'Add ?mode=isolated to read the state once per request and answer every question from it.',
     'Cheaper when many questions share a long state; answers can shift slightly with the set of questions.',
     'The default, separate, answers each question in its own sequence.'),
    ('Temperature',
     'Without ?temperature= the model\'s saved temperature is used, as in the official API.',
     'Pass ?temperature=1 (HTTP) or temperature=1.0 (Python) to override it, and ?raw=1 for raw logits.'),
    ('Locate',
     'A question {"type": "locate", "instructions": "error codes"} returns verbatim spans of the state',
     'with character offsets and scores. The Locate weights load on the first Locate request.'),
    ('Images',
     'Put OpenAI-style parts {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}',
     'anywhere in "state" (PNG, JPEG or WebP, up to 8 images). They are placed before the text.'),
    ('Files in, files out',
     'decider-mlx run --model M --input requests.jsonl --output responses.jsonl',
     'One request per line, or {"id", "request", "options"} with per-line execution_mode / temperature / raw.',
     'Output lines keep the input order: {"id", "index", "response" | "error"}.'),
    ('Python API',
     'from decider_mlx import load; app = load("pplx-decider-flash-v1.3")',
     'app.decide(request) or app.decide([requests]); thread-safe, calls are served in arrival order.',
     'app.decide_each([...]) returns an exception in place of each invalid request instead of raising.'),
    ('Two models at once',
     'Run one server per model on different ports, for example:',
     'decider-mlx serve --model pplx-decider-flash-v1.3 --port 8000',
     'decider-mlx serve --model pplx-decider-v1.3 --port 8001   (independent processes; both can run at once)'),
    ('Startup',
     'serve accepts requests about half a second after start (503 with Retry-After until then).',
     'Graphs for common shapes are prepared in the background between requests; GET /metrics shows the state.'),
)


def tip(i):
    """(title, lines) of tip i (cycling)."""
    t = TIPS[i % len(TIPS)]
    return t[0], t[1:]
