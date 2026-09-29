# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    "sphinx_reredirects",
    "sphinx_sitemap",
]

# Relative targets keep old links within the same language and build version.
redirects = {
    "rst_source/benchmarks": "leaderboard/index.html",
    "rst_source/installation": "get_started/quickstart.html",
    "rst_source/overview": "get_started/overview.html",
    "rst_source/quickstart": "get_started/quickstart.html",
    "rst_source/leaderboard": "leaderboard/index.html",
    "rst_source/awesome_works/harnessvla": "../resources/harnessvla.html",
    "rst_source/usage/simulation": "../simulators/index.html",
    "rst_source/usage/real_robots": "../real_world_robots/index.html",
    "rst_source/usage/memory": "../guides/memory.html",
    "rst_source/usage/configure_primitives": "../guides/configure_primitives.html",
    "rst_source/usage/configure_planner": "../guides/configure_planner.html",
    "rst_source/usage/cli": "../guides/cli.html",
    "rst_source/usage/dashboard": "../guides/dashboard.html",
    "rst_source/usage/flash": "../guides/flash.html",
    "rst_source/usage/flywheel": "../guides/flywheel.html",
    "rst_source/usage/advanced_deployment": "../guides/advanced_deployment.html",
    "rst_source/usage/libero": "../simulators/libero.html",
    "rst_source/usage/robocasa": "../simulators/robocasa.html",
    "rst_source/usage/robotwin": "../simulators/robotwin.html",
    "rst_source/usage/franka": "../real_world_robots/franka.html",
    "rst_source/usage/dual_franka": "../real_world_robots/dual_franka.html",
    "rst_source/usage/yam": "../real_world_robots/yam.html",
    "rst_source/usage/so101": "../real_world_robots/so101.html",
    "rst_source/usage/real_world_demos_franka": "../real_world_demos/franka.html",
    "rst_source/usage/real_world_demos_yam": "../real_world_demos/yam.html",
}

source_suffix = {".rst": "restructuredtext"}
root_doc = "index"
templates_path = ["_templates"]
exclude_patterns = ["**/_*.rst"]
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
html_static_path = ["../_static", "_static"]
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


def show_simulator_subsections(app, pagename, templatename, context, doctree):
    if pagename.startswith("rst_source/simulators/"):
        context["theme_show_toc_level"] = 2


def setup(app):
    app.connect("html-page-context", show_simulator_subsections, priority=400)
