.PHONY: lint format update build install update-clocks-notebooks update-all-clocks verify-hf-auth verify-hf-data-repo-public create-hf-data-repo upload-clocks-to-hf upload-static-data-to-hf tag-hf-data-repo process-tutorials test test-all test-tutorials docs version release-preflight commit tag release release-slim clean

VERSION ?= v$(shell sed -n 's/^__version__ = "\(.*\)"/\1/p' src/pyaging/__init__.py)
HF_REPO_ID ?= lucascamillomd/pyaging-data
HF_REPO_OWNER ?= lucascamillomd
HF_STATIC_DIR ?= hf_static_data
COMMIT_MSG ?= "Bump to $(VERSION)"
RELEASE_MSG ?= "Release $(VERSION)"
# Git and Hugging Face tags keep the leading "v"; the package and the clock
# metadata use the bare number, as the registry and earlier releases do
RELEASE_NUMBER = $(patsubst v%,%,$(VERSION))
# Same list as the CI tutorials job: these notebooks pip install unpinned packages
TUTORIALS_IGNORE ?= tutorials/tutorial_cpgptgrimage3.ipynb

# Re-runs must move the tag to the freshly uploaded revision, so delete any
# existing tag before recreating it (exist_ok would silently keep the old one)
define TAG_HF_DATA_REPO_SCRIPT
import contextlib
import os

from huggingface_hub import create_tag, delete_tag
from huggingface_hub.errors import HfHubHTTPError, RevisionNotFoundError

repo_id = os.environ["TAG_HF_REPO_ID"]
tag = os.environ["TAG_HF_TAG"]
with contextlib.suppress(RevisionNotFoundError, HfHubHTTPError):
    delete_tag(repo_id, tag=tag)
create_tag(repo_id, tag=tag)
print(f"Tagged {repo_id} @ {tag}")
endef
export TAG_HF_DATA_REPO_SCRIPT

lint:
	@echo "Running ruff for linting..."
	uv run ruff check src/pyaging --fix

format:
	@echo "Running ruff for code formatting..."
	uv run ruff format src/pyaging

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
		uv run --no-sync jupyter nbconvert --execute --inplace "$$notebook" || { \
			echo ""; \
			echo "ERROR: ================================================================"; \
			echo "ERROR: Failed to process notebook: $$notebook"; \
			echo "ERROR: ================================================================"; \
			echo ""; \
			exit 1; \
		}; \
		counter=$$((counter+1)); \
	done && cd ../..

update-all-clocks:
	@echo "Running script to update all clocks..."
	@cd clocks && uv run python update_all_clocks.py $(RELEASE_NUMBER) || { echo "Updating clocks failed"; exit 1; } && cd ..

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
	@echo "Syncing per-clock repos under the pyaging org..."
	uv run python clocks/hf_repo_sync.py || { echo "Per-clock repo sync failed"; exit 1; }
	@echo "Publishing aggregate metadata..."
	uv run hf upload "$(HF_REPO_ID)" clocks/metadata/all_clock_metadata.pt all_clock_metadata.pt --type model --commit-message "Update aggregate clock metadata"
	@uv run hf models info "$(HF_REPO_ID)" --format json | uv run python -c 'import json, sys; print("HF revision:", json.load(sys.stdin)["sha"])'

tag-hf-data-repo: verify-hf-auth
	@echo "Tagging HF data repo $(HF_REPO_ID) with $(VERSION)..."
	TAG_HF_REPO_ID='$(HF_REPO_ID)' TAG_HF_TAG='$(VERSION)' uv run python -c "$$TAG_HF_DATA_REPO_SCRIPT" || { echo "Tagging HF data repo failed"; exit 1; }
	@echo "Tagging per-clock repos with $(VERSION)..."
	uv run python clocks/hf_repo_sync.py --tag "$(VERSION)" --tag-only || { echo "Tagging per-clock repos failed"; exit 1; }

upload-static-data-to-hf: verify-hf-data-repo-public
	@test -d "$(HF_STATIC_DIR)/repo" || { echo "Missing $(HF_STATIC_DIR)/repo staging directory"; exit 1; }
	uv run hf upload "$(HF_REPO_ID)" "$(HF_STATIC_DIR)/repo" . --type model --commit-message "Add current pyaging static data dependencies"

process-tutorials:
	@echo "Processing tutorials..."
	@cd tutorials && \
	for notebook in *.ipynb; do \
		case " $(TUTORIALS_IGNORE) " in *" tutorials/$$notebook "*) echo "Skipping $$notebook"; continue;; esac; \
		echo "Processing tutorial notebook: $$notebook"; \
		uv run --no-sync jupyter nbconvert --ExecutePreprocessor.timeout=600 --to notebook --execute --inplace "$$notebook" || { echo "Error processing $$notebook"; exit 1; }; \
	done && cd ..

test:
	@echo "Running offline tests..."
	uv run pytest || { echo "Offline tests failed"; exit 1; }

test-all:
	@echo "Running offline tests across supported Python versions..."
	@for py in 3.11 3.12 3.13 3.14; do \
		echo "Testing with Python $$py..."; \
		uv run --python $$py pytest || { echo "Tests failed on Python $$py"; exit 1; }; \
	done

test-tutorials:
	@echo "Running tutorial tests..."
	uv run pytest --nbmake tutorials/ $(addprefix --ignore=,$(TUTORIALS_IGNORE)) || { echo "Tutorial tests failed"; exit 1; }

docs:
	@echo "Building documentation..."
	uv run make -C docs html

version:
	@echo "Updating version in src/pyaging/__init__.py to $(VERSION)..."
	uv run --no-sync python -c 'import json, pathlib, re, sys; path = pathlib.Path("src/pyaging/__init__.py"); path.write_text(re.sub(r"(?m)^__version__ = .*$$", "__version__ = " + json.dumps(sys.argv[1]), path.read_text()))' "$(RELEASE_NUMBER)"

# Runs before anything is changed or published: an existing tag would otherwise
# only fail the final `tag` step, after the data had already gone out
release-preflight:
	@echo "Checking release preconditions for $(VERSION)..."
	@echo "$(VERSION)" | grep -Eq '^v[0-9]+\.[0-9]+\.[0-9]+$$' || { echo "VERSION must look like v1.2.3, got '$(VERSION)'"; exit 1; }
	@branch=$$(git rev-parse --abbrev-ref HEAD) && [ "$$branch" = "main" ] || { echo "Releases must run from main, not '$$branch'"; exit 1; }
	@if git rev-parse -q --verify "refs/tags/$(VERSION)" >/dev/null; then echo "Tag $(VERSION) already exists locally"; exit 1; fi
	@remote=$$(git ls-remote --tags origin "refs/tags/$(VERSION)") || { echo "Could not list tags on origin"; exit 1; }; \
	if [ -n "$$remote" ]; then echo "Tag $(VERSION) already exists on origin"; exit 1; fi

# A re-run after a later step failed has nothing new to commit, which is fine;
# a failed push is not, because every publish step after it assumes main is pushed
commit:
	@echo "Committing and pushing changes..."
	git add src/pyaging uv.lock pyproject.toml Makefile .github tests clocks/notebooks clocks/metadata tutorials docs/source docs/_static docs/development README.md CHANGELOG.md
	git diff --cached --quiet || git commit -m $(COMMIT_MSG) || { echo "Git commit failed"; exit 1; }
	git push || { echo "Git push failed"; exit 1; }

tag:
	@echo "Creating and pushing tag $(VERSION)..."
	git tag -a "$(VERSION)" -m $(RELEASE_MSG)
	git push origin "$(VERSION)" || { echo "Git tag creation or push failed"; exit 1; }

# Installed packages read Hugging Face `main`, so data is published only once
# the matching commit is on origin, and the tag that triggers PyPI goes last
release:
	$(MAKE) release-preflight
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
	$(MAKE) commit
	$(MAKE) upload-clocks-to-hf
	$(MAKE) tag-hf-data-repo
	$(MAKE) tag
	@echo "Release $(VERSION) completed successfully"

release-slim:
	$(MAKE) release-preflight
	$(MAKE) version
	$(MAKE) lint
	$(MAKE) format
	$(MAKE) update
	$(MAKE) build
	$(MAKE) install
	$(MAKE) update-all-clocks
	$(MAKE) test
	$(MAKE) docs
	$(MAKE) commit
	$(MAKE) upload-clocks-to-hf
	$(MAKE) tag-hf-data-repo
	$(MAKE) tag
	@echo "Release $(VERSION) (slim) completed successfully"

clean:
	@echo "Removing build and cache directories..."
	rm -rf .pytest_cache .ruff_cache build dist docs/_build
