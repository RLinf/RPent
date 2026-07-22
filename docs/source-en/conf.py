import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

project = "RPent"
author = "RPent Contributors"
copyright = "2026, RPent Contributors"
version = "latest"
release = version

extensions = [
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_sitemap",
]

source_suffix = {".rst": "restructuredtext"}
root_doc = "index"
templates_path = ["_templates"]
exclude_patterns = []
default_role = "code"

language = "en"
html_search_language = "en"
html_theme = "pydata_sphinx_theme"
html_title = "RPent Documentation"
html_show_sourcelink = False
html_baseurl = os.environ.get(
    "READTHEDOCS_CANONICAL_URL",
    "https://rpent.readthedocs.io/en/latest/",
)
sitemap_url_scheme = "{link}"
html_static_path = ["_static"]
html_extra_path = ["../architecture.svg"]
html_css_files = ["css/custom.css"]
html_js_files = [
    "js/version-switcher.js",
    "js/lang-switcher.js",
    "js/sidebar-nav.js",
    "js/theme-toggle.js",
]
html_sidebars = {
    "**": [
        "sidebar-brand",
        "search-field",
        "sidebar-tools",
        "global-sidebar-nav",
    ]
}

html_theme_options = {
    "search_bar_text": "Search docs…",
    "navbar_start": [],
    "navbar_center": [],
    "navbar_end": [],
    "navbar_align": "left",
    "secondary_sidebar_items": {"**": ["page-toc"], "index": []},
    "collapse_navigation": False,
    "show_nav_level": 1,
    "navigation_depth": 5,
    "header_links_before_dropdown": 10,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/RLinf/RPent",
            "icon": "fab fa-github",
            "type": "fontawesome",
        }
    ],
    "switcher": {
        "json_url": "_static/versions.json",
        "version_match": version,
    },
}
