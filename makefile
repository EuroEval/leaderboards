# Set the PATH env var used by cargo and uv
export PATH := ${HOME}/.local/bin:${HOME}/.cargo/bin:$(PATH)

update: pull check download generate_leaderboards publish

force-update: pull check download force_generate_leaderboards publish

no-download-update: pull check generate_leaderboards publish

pull:
	@git pull

download:
	@touch new_results.jsonl
	@scp -o ConnectTimeout=5 ucloud:/work/euroeval_benchmark_results.jsonl ucloud_results.jsonl || true
	@if [ -f ucloud_results.jsonl ]; then \
		cat ucloud_results.jsonl >> new_results.jsonl; \
		rm ucloud_results.jsonl; \
	fi
	@scp -o ConnectTimeout=5 ucloud:/work/api/euroeval_benchmark_results.jsonl ucloud_api_results.jsonl || true
	@if [ -f ucloud_api_results.jsonl ]; then \
		cat ucloud_api_results.jsonl >> new_results.jsonl; \
		rm ucloud_api_results.jsonl; \
	fi
	@scp -o ConnectTimeout=5 large:/work/euroeval_benchmark_results.jsonl large_results.jsonl || true
	@if [ -f large_results.jsonl ]; then \
		cat large_results.jsonl >> new_results.jsonl; \
		rm large_results.jsonl; \
	fi
	@scp -o ConnectTimeout=5 bosnian:/work/euroeval_benchmark_results.jsonl bosnian_results.jsonl || true
	@if [ -f bosnian_results.jsonl ]; then \
		cat bosnian_results.jsonl >> new_results.jsonl; \
		rm bosnian_results.jsonl; \
	fi
	@scp -o ConnectTimeout=5 slovene:/work/euroeval_benchmark_results.jsonl slovene_results.jsonl || true
	@if [ -f slovene_results.jsonl ]; then \
		cat slovene_results.jsonl >> new_results.jsonl; \
		rm slovene_results.jsonl; \
	fi

generate_leaderboards:
	@uv run src/scripts/generate_leaderboards.py

force_generate_leaderboards:
	@uv run src/scripts/generate_leaderboards.py --force

publish:
	@for leaderboard in leaderboards/*.csv; do \
		git add $${leaderboard}; \
	done
	@for leaderboard in leaderboards/*.json; do \
		git add $${leaderboard}; \
	done
	@git add results.tar.gz
	@git commit -m "feat: Update leaderboards" || true
	@git push
	@echo "Published leaderboards!"

install:
	@echo "Installing the 'leaderboards' project..."
	@$(MAKE) --quiet install-rust
	@$(MAKE) --quiet install-uv
	@$(MAKE) --quiet install-dependencies
	@$(MAKE) --quiet install-pre-commit
	@echo "Installed the 'leaderboards' project.."

install-rust:
	@if [ "$(shell which rustup)" = "" ]; then \
		curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; \
		echo "Installed Rust."; \
	fi

install-uv:
	@if [ "$(shell which uv)" = "" ]; then \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
        echo "Installed uv."; \
    else \
		echo "Updating uv..."; \
		uv self update; \
	fi

install-dependencies:
	@uv python install 3.11
	@uv sync --all-extras --python 3.11

install-pre-commit:
	@uv run pre-commit install

lint:
	uv run ruff check . --fix

format:
	uv run ruff format .

type-check:
	@uv run mypy . \
		--install-types \
		--non-interactive \
		--ignore-missing-imports \
		--show-error-codes \
		--check-untyped-defs

check: lint format type-check
