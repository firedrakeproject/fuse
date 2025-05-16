# Adds file annotations to Github Actions (only useful on CI)
GITHUB_ACTIONS_FORMATTING=0
ifeq ($(GITHUB_ACTIONS_FORMATTING), 1)
	FLAKE8_FORMAT=--format='::error file=%(path)s,line=%(row)d,col=%(col)d,title=%(code)s::%(path)s:%(row)d:%(col)d: %(code)s %(text)s'
else
	FLAKE8_FORMAT=
endif
.PHONY: test docs

docs:
	# put -n back in when things are better documented
	@(cd docs/ && (make html SPHINXOPTS="-W --keep-going"))

lint:
	@echo "    Linting FUSE codebase"
	@python3 -m flake8 $(FLAKE8_FORMAT) fuse
	@echo "    Linting FUSE test suite"
	@python3 -m flake8 $(FLAKE8_FORMAT) test

test_examples:
	@echo "    Running examples"
	@python3 -m pytest test/test_2d_examples_docs.py
	@python3 -m pytest test/test_3d_examples_docs.py

tests:
	@echo "    Running all tests"
	@python3 -m coverage run -p -m pytest -rx test

coverage:
	@python3 -m coverage combine
	@python3 -m coverage report -m 
	@python3 -m coverage json

test_cells:
	@echo "    Running all cell comparison tests"
	@firedrake-clean
	@python3 -m pytest -rPx --run-cleared test/test_cells.py::test_ref_els[expect0]
	@firedrake-clean
	@python3 -m pytest -rPx --run-cleared test/test_cells.py::test_ref_els[expect1]

clean:
	@(cd docs/ && make clean)

prepush: lint tests
	@rm .coverage.*
	make clean docs