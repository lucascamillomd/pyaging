.PHONY: lint format update build install update-clocks-notebooks update-all-clocks upload-to-s3 process-tutorials test test-tutorials docs version commit tag release release-slim

VERSION ?= v0.1.27
COMMIT_MSG ?= "Bump to $(VERSION)"
RELEASE_MSG ?= "Release $(VERSION)"

lint:
	@echo "Running ruff for linting..."
	ruff check pyaging --fix

format:
	@echo "Running ruff for code formatting..."
	ruff format pyaging

update:
	@echo "Running uv sync..."
	uv sync

build: lint format
	@echo "Building the package..."
	uv build

install: build
	@echo "Installing the package..."
	uv sync

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
		jupyter nbconvert --execute --inplace "$$notebook" || { \
			echo ""; \
			echo "ERROR: ================================================================"; \
			echo "ERROR: Failed to process notebook: $$notebook"; \
			echo "ERROR: ================================================================"; \
			echo ""; \
			counter=$$((counter+1)); \
			continue; \
		}; \
		counter=$$((counter+1)); \
	done && cd ../..

update-all-clocks:
	@echo "Running script to update all clocks..."
	@cd clocks && python3 update_all_clocks.py $(VERSION) || { echo "Updating clocks failed"; exit 1; } && cd ..

upload-to-s3:
	@echo "Uploading clock metadata to S3..."
	aws s3 cp clocks/metadata/all_clock_metadata.pt s3://pyaging/clocks/metadata0.1.0/all_clock_metadata.pt || { echo "Uploading metadata failed"; exit 1; }
	@echo "Syncing clock weights to S3..."
	aws s3 sync clocks/weights/ s3://pyaging/clocks/weights0.1.0/ || { echo "Syncing weights failed"; exit 1; }
	@echo "Clock data uploaded to S3 successfully!"

process-tutorials:
	@echo "Processing tutorials..."
	@cd tutorials && \
	for notebook in *.ipynb; do \
		echo "Processing tutorial notebook: $$notebook"; \
		jupyter nbconvert --ExecutePreprocessor.timeout=600 --to notebook --execute --inplace "$$notebook" || { echo "Error processing $$notebook"; exit 1; }; \
	done && cd ..

test:
	@echo "Running gold standard tests..."
	uv sync --quiet || { echo "Failed to sync dependencies"; exit 1; }
	tox || { echo "Gold standard tests failed"; exit 1; }

test-tutorials:
	@echo "Running tutorial tests..."
	uv run pytest --nbmake tutorials/ || { echo "Tutorial tests failed"; exit 1; }

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

release: version lint format update build install update-clocks-notebooks update-all-clocks upload-to-s3 process-tutorials test test-tutorials docs commit tag
	@echo "Release $(VERSION) completed successfully"

release-slim: version lint format update build install update-all-clocks upload-to-s3 test docs commit tag
	@echo "Release $(VERSION) (slim) completed successfully"