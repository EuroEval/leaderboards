# Set the PATH env var used by cargo and uv
export PATH := ${HOME}/.local/bin:${HOME}/.cargo/bin:$(PATH)

update: check download process_results generate_leaderboards publish

force-update: check download process_results force_generate_leaderboards publish

download:
	@scp -o ConnectTimeout=5 lancelot:/home/alex-admin/euroeval/euroeval_benchmark_results.jsonl lancelot_results.jsonl || true
	@scp -o ConnectTimeout=5 blackknight:/home/alex-admin/euroeval/euroeval_benchmark_results.jsonl blackknight_results.jsonl || true
	@scp -o ConnectTimeout=5 runpod:/root/euroeval_benchmark_results.jsonl runpod_results.jsonl || true
	@touch results/results.jsonl
	@if [ -f lancelot_results.jsonl ]; then \
		cat lancelot_results.jsonl >> results/results.jsonl; \
		rm lancelot_results.jsonl; \
	fi
	@if [ -f blackknight_results.jsonl ]; then \
		cat blackknight_results.jsonl >> results/results.jsonl; \
		rm blackknight_results.jsonl; \
	fi
	@if [ -f runpod_results.jsonl ]; then \
		cat runpod_results.jsonl >> results/results.jsonl; \
		rm runpod_results.jsonl; \
	fi

process_results:
	@uv run src/process_results.py results/results.jsonl

generate_leaderboards:
	@for config in leaderboard_configs/*.yaml; do \
		uv run src/generate_leaderboards.py $${config}; \
	done

force_generate_leaderboards:
	@for config in leaderboard_configs/*.yaml; do \
		uv run src/generate_leaderboards.py --force $${config}; \
	done

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
	@echo "Installed the 'leaderboards' project. If you want to use pre-commit hooks, run 'make install-pre-commit'."

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
