# Set the PATH env var used by cargo and uv
export PATH := ${HOME}/.local/bin:${HOME}/.cargo/bin:$(PATH)

update: pull check download generate_leaderboards publish

force-update: pull check download force_generate_leaderboards publish

no-download-update: pull check generate_leaderboards publish

pull:
	@git pull

download:
	@scp -o ConnectTimeout=5 annika-ucloud:/home/ucloud/dan/euroeval_benchmark_results.jsonl annika_ucloud_results.jsonl || true
	@scp -o ConnectTimeout=5 70b-ucloud:/home/ucloud/euroeval_benchmark_results.jsonl 70b_ucloud_results.jsonl || true
	@scp -o ConnectTimeout=5 70b-ucloud:/home/ucloud/eval2/euroeval_benchmark_results.jsonl 70b_ucloud_2_results.jsonl || true
	@scp -o ConnectTimeout=5 70b-ucloud:/home/ucloud/api/euroeval_benchmark_results.jsonl 70b_ucloud_api_results.jsonl || true
	@scp -o ConnectTimeout=5 pt-ucloud:/home/ucloud/euroeval_benchmark_results.jsonl pt_ucloud_results.jsonl || true
	@scp -o ConnectTimeout=5 b200:/work/euroeval_benchmark_results.jsonl b200_results.jsonl || true
	@touch results/results.jsonl
	@if [ -f annika_ucloud_results.jsonl ]; then \
		cat annika_ucloud_results.jsonl >> results/results.jsonl; \
		rm annika_ucloud_results.jsonl; \
	fi
	@if [ -f 70b_ucloud_results.jsonl ]; then \
		cat 70b_ucloud_results.jsonl >> results/results.jsonl; \
		rm 70b_ucloud_results.jsonl; \
	fi
	@if [ -f 70b_ucloud_2_results.jsonl ]; then \
		cat 70b_ucloud_2_results.jsonl >> results/results.jsonl; \
		rm 70b_ucloud_2_results.jsonl; \
	fi
	@if [ -f 70b_ucloud_api_results.jsonl ]; then \
		cat 70b_ucloud_api_results.jsonl >> results/results.jsonl; \
		rm 70b_ucloud_api_results.jsonl; \
	fi
	@if [ -f pt_ucloud_results.jsonl ]; then \
		cat pt_ucloud_results.jsonl >> results/results.jsonl; \
		rm pt_ucloud_results.jsonl; \
	fi
	@if [ -f b200_results.jsonl ]; then \
		cat b200_results.jsonl >> results/results.jsonl; \
		rm b200_results.jsonl; \
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
	@for results in results/*.jsonl; do \
		git add $${results}; \
	done
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
