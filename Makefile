.PHONY: lint format update build install update-glossary update-clocks-notebooks update-all-clocks process-tutorials test test-tutorials docs version commit tag release

VERSION ?= v0.1.14
COMMIT_MSG ?= "Bump to $(VERSION)"
RELEASE_MSG ?= "Release $(VERSION)"

lint:
	@echo "Running ruff for linting..."
	ruff check pyaging --fix

format:
	@echo "Running ruff for code formatting..."
	ruff format pyaging

update:
	@echo "Running poetry update..."
	poetry update

build: lint format
	@echo "Building the package..."
	poetry build

install: build
	@echo "Installing the package..."
	poetry install

update-glossary:
	@echo "Updating clock glossary..."
	@cd docs && python3 source/make_clock_glossary.py || { echo "Clock glossary update failed"; exit 1; } && cd ..

update-clocks-notebooks:
	@echo "Updating clocks and notebooks..."
	@cd clocks/notebooks && \
	total=$$(ls *.ipynb | wc -l) && \
	counter=1 && \
	for notebook in *.ipynb; do \
		if [ "$$notebook" = "template.ipynb" ]; then \
			echo "Skipping template.ipynb"; \
			continue; \
		fi; \
		echo "Processing clock notebook ($$counter/$$total): $$notebook"; \
		jupyter nbconvert --execute --inplace "$$notebook" || { echo "Error processing $$notebook"; exit 1; }; \
		counter=$$((counter+1)); \
	done && cd ../..

update-all-clocks:
	@echo "Running script to update all clocks..."
	@cd clocks && python3 update_all_clocks.py $(VERSION) || { echo "Updating clocks failed"; exit 1; } && cd ..
	@echo "Reminder: Upload all clocks and metadata to S3!"

process-tutorials:
	@echo "Processing tutorials..."
	@cd tutorials && \
	for notebook in *.ipynb; do \
		echo "Processing tutorial notebook: $$notebook"; \
		jupyter nbconvert --ExecutePreprocessor.timeout=600 --to notebook --execute --inplace "$$notebook" || { echo "Error processing $$notebook"; exit 1; }; \
	done && cd ..

test:
	@echo "Running gold standard tests..."
	poetry run pytest || { echo "Gold standard tests failed"; exit 1; }

test-tutorials:
	@echo "Running tutorial tests..."
	poetry run pytest --nbmake tutorials/ || { echo "Tutorial tests failed"; exit 1; }

docs:
	@echo "Building documentation..."
	cp tutorials/*.ipynb docs/source/tutorials
	cp clocks/notebooks/*.ipynb docs/source/clock_notebooks
	@cd docs && make html || { echo "Documentation build failed"; exit 1; }

version:
	@echo "Updating version in pyproject.toml to $(VERSION)..."
	sed -i '' "s/^version = \".*\"/version = \"$(patsubst v%,%,$(VERSION))\"/" pyproject.toml || { echo "Error updating version in pyproject.toml"; exit 1; }
	@echo "Updating version in pyaging/__init__.py to $(VERSION)..."
	sed -i '' "s/^__version__ = \".*\"/__version__ = \"$(patsubst v%,%,$(VERSION))\"/" pyaging/__init__.py || { echo "Error updating version in pyaging/__init__.py"; exit 1; }

commit:
	@echo "Committing and pushing changes..."
	git add .
	git commit -m $(COMMIT_MSG) || { echo "Git commit failed"; exit 1; }
	git push || { echo "Git push failed"; exit 1; }

tag:
	@echo "Creating and pushing tag $(VERSION)..."
	git tag -a "$(VERSION)" -m $(RELEASE_MSG)
	git push origin "$(VERSION)" || { echo "Git tag creation or push failed"; exit 1; }

release: version lint format update build install update-glossary update-clocks-notebooks update-all-clocks process-tutorials test test-tutorials docs commit tag
	@echo "Release $(VERSION) completed successfully"