import logging
from jinja2 import Environment

logger = logging.getLogger(__name__)


def _wrap_in_html_template(paper_html_content: str) -> str:
    base_template = """
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\">
  <title>Research Paper</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      margin: 2rem auto;
      max-width: 800px;
      padding: 0 1rem;
      line-height: 1.6;
      color: #333;
      background-color: #fff;
    }
    h2 {
      border-bottom: 2px solid #ddd;
      padding-bottom: 0.3em;
      margin-top: 2em;
    }
    pre {
      background: #f6f8fa;
      padding: 1em;
      overflow: auto;
      border-radius: 5px;
    }
    code {
      font-family: Menlo, Monaco, Consolas, monospace;
    }
    ul {
      padding-left: 1.5em;
    }
    figure {
      text-align: center;
      margin: 1.5em 0;
    }
    figcaption {
      font-size: 0.9em;
      color: #666;
    }
  </style>
</head>
<body>
{{ content }}
</body>
</html>"""
    env = Environment()
    template = env.from_string(base_template)
    return template.render(content=paper_html_content)

def render_html(paper_html_content: str) -> str:
    return _wrap_in_html_template(paper_html_content)
