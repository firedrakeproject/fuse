# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))
sys.path.append(os.path.abspath('sphinxext'))
import sphinx_bootstrap_theme

# -- Project information -----------------------------------------------------

project = 'FUSE'
copyright = '2025, India Marsden, David A. Ham, Patrick E. Farrell'
author = 'India Marsden, David A. Ham, Patrick E. Farrell'


# The full version, including alpha/beta/rc tags
release = '2025.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              "sphinx.ext.autosectionlabel",
              'sphinx.ext.mathjax',
              'sphinx.ext.intersphinx',
              "matplotlib.sphinxext.plot_directive",
              "sphinx_design",
              'sphinxcontrib.fulltoc',
              'sphinx_remove_toctrees']

autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []
# nitpick_ignore = [('py:class', 'type')]
numpydoc_class_members_toctree = False

# -- Autodoc configuration ---------------------------------------------------

# autodoc_mock_imports = ["firedrake", "FIAT", "finat"]

# Make sure the target is unique
autosectionlabel_prefix_document = True

intersphinx_mapping = {
    'ufl': ('https://docs.fenicsproject.org/ufl/main/', None),
    'FIAT': ('https://firedrakeproject.org/fiat', None),
    'matplotlib': ('https://matplotlib.org/', None),
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'networkx': ("https://networkx.org/documentation/stable/", None)}


remove_from_toctrees = ["_autosummary/fuse.cells.Arrow3D.rst"]
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'bootstrap'
html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
# html_css_files = ["additional.css"]
def setup(app):
    app.add_css_file('additional.css')  

html_theme_options = {
    'navbar_links': [
        ("About", "about"),
        ("Installation", "install"),
        ("Documentation", "api"),
    ],
    'bootswatch_theme': 'cosmo',
    'source_link_position': None,
    #  Render the next and previous page links in navbar. (Default: true)
    'navbar_sidebarrel': False,

    # Render the current pages TOC in the navbar. (Default: true)
    'navbar_pagenav': False,

    # Tab name for the current pages TOC. (Default: "Page")
    'navbar_pagenav_name': "Page",
    'globaltoc_depth': 2,
}

html_sidebars = {'api/*': ['localtoc.html'], 
                 'api': ['localtoc.html'],
                 'about': [],
                 'install': [],
                 'index': ['localtoc.html'],
                 'manual': ['localtoc.html'],
                 'manual/*': ['localtoc.html'],
                 '_autosummary/*': ['custom-sidebar.html'], }