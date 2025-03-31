Building
========

To build the website locally, you need Sphinx and a couple of plugins:
```console
  python -m pip install sphinx sphinx_bootstrap_theme sphinx_design
```
Then:
```console
  make html
```
The resulting files will appear in `build/html`.

If you further install sphinx-autobuild:
```console
  python -m pip install sphinx-autobuild
```
then you can start a local web server that automatically rebuilds when you save any source file with:
```console
  make auto-html
```
Point your web browser at [http://127.0.0.1:8000/](http://127.0.0.1:8000/) to see the site.