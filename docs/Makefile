# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = _build

# run python script before making html to update the python table
html:
	python source/make_clock_glossary.py
	ln -sf ../../tutorials $(SOURCEDIR)/tutorials # Create a symbolic link for tutorials directory
	ln -sf ../../clocks/notebooks $(SOURCEDIR)/clock_notebooks # Create a symbolic link for clocks directory
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Clean the build directory and the file transfers to the source directory
clean:
	rm -rf $(BUILDDIR)/*
	rm -rf $(SOURCEDIR)/tutorials
	rm -rf $(SOURCEDIR)/clock_notebooks

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
