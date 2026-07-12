.PHONY: lint format update build install update-clocks-notebooks update-all-clocks verify-hf-auth verify-hf-data-repo-public create-hf-data-repo upload-clocks-to-hf upload-static-data-to-hf process-tutorials test test-tutorials docs version commit tag release release-slim

VERSION ?= v0.3.0
HF_REPO_ID ?= lucascamillomd/pyaging-data
HF_REPO_OWNER ?= lucascamillomd
HF_STATIC_DIR ?= hf_static_data
COMMIT_MSG ?= "Bump to $(VERSION)"
RELEASE_MSG ?= "Release $(VERSION)"

lint:
	@echo "Running ruff for linting..."
	uv run ruff check pyaging --fix

format:
	@echo "Running ruff for code formatting..."
	uv run ruff format pyaging

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
	@cd clocks && uv run python update_all_clocks.py $(VERSION) || { echo "Updating clocks failed"; exit 1; } && cd ..

verify-hf-auth:
	@account=$$(uv run hf auth whoami --format json | uv run python -c 'import json, sys; print(json.load(sys.stdin)["user"])'); \
	if [ "$$account" != "$(HF_REPO_OWNER)" ]; then \
		echo "Expected HF account $(HF_REPO_OWNER), got $$account"; \
		exit 1; \
	fi

verify-hf-data-repo-public: verify-hf-auth
	@uv run hf models info "$(HF_REPO_ID)" --format json | uv run python -c 'import json, sys; private=json.load(sys.stdin)["private"]; sys.exit("Expected public HF repository $(HF_REPO_ID)") if private is not False else print("Verified public HF repository: $(HF_REPO_ID)")'

create-hf-data-repo: verify-hf-auth
	uv run hf repos create "$(HF_REPO_ID)" --type model --public --exist-ok
	$(MAKE) verify-hf-data-repo-public
	uv run hf upload "$(HF_REPO_ID)" clocks/huggingface/README.md README.md --type model --commit-message "Document pyaging data repository"

upload-clocks-to-hf: verify-hf-data-repo-public
	@echo "Uploading changed clock weights to Hugging Face..."
	uv run hf upload "$(HF_REPO_ID)" clocks/weights . --type model --commit-message "Update pyaging clock weights"
	@echo "Publishing aggregate metadata after weights..."
	uv run hf upload "$(HF_REPO_ID)" clocks/metadata/all_clock_metadata.pt all_clock_metadata.pt --type model --commit-message "Update aggregate clock metadata"
	@uv run hf models info "$(HF_REPO_ID)" --format json | uv run python -c 'import json, sys; print("HF revision:", json.load(sys.stdin)["sha"])'

upload-static-data-to-hf: verify-hf-data-repo-public
	@test -d "$(HF_STATIC_DIR)/repo" || { echo "Missing $(HF_STATIC_DIR)/repo staging directory"; exit 1; }
	uv run hf upload "$(HF_REPO_ID)" "$(HF_STATIC_DIR)/repo" . --type model --commit-message "Add current pyaging static data dependencies"

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
	uv run tox || { echo "Gold standard tests failed"; exit 1; }

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

release:
	$(MAKE) version
	$(MAKE) lint
	$(MAKE) format
	$(MAKE) update
	$(MAKE) build
	$(MAKE) install
	$(MAKE) update-clocks-notebooks
	$(MAKE) update-all-clocks
	$(MAKE) process-tutorials
	$(MAKE) test
	$(MAKE) test-tutorials
	$(MAKE) docs
	$(MAKE) upload-clocks-to-hf
	$(MAKE) commit
	$(MAKE) tag
	@echo "Release $(VERSION) completed successfully"

release-slim:
	$(MAKE) version
	$(MAKE) lint
	$(MAKE) format
	$(MAKE) update
	$(MAKE) build
	$(MAKE) install
	$(MAKE) update-all-clocks
	$(MAKE) test
	$(MAKE) docs
	$(MAKE) upload-clocks-to-hf
	$(MAKE) commit
	$(MAKE) tag
	@echo "Release $(VERSION) (slim) completed successfully"
