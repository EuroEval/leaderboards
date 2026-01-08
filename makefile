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
	@scp -o ConnectTimeout=5 eval-backlog:/work/euroeval_benchmark_results.jsonl eval_backlog_results.jsonl || true
	@if [ -f eval_backlog_results.jsonl ]; then \
		cat eval_backlog_results.jsonl >> new_results.jsonl; \
		rm eval_backlog_results.jsonl; \
	fi
	@scp -o ConnectTimeout=5 reevaluations:/work/euroeval_benchmark_results.jsonl reevaluations_results.jsonl || true
	@if [ -f reevaluations_results.jsonl ]; then \
		cat reevaluations_results.jsonl >> new_results.jsonl; \
		rm reevaluations_results.jsonl; \
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
	@$(MAKE) --quiet install-uv
	@$(MAKE) --quiet install-dependencies
	@$(MAKE) --quiet install-pre-commit
	@echo "Installed the 'leaderboards' project."

install-uv:
	@if [ "$(shell which uv)" = "" ]; then \
		if [ "$(shell which rustup)" = "" ]; then \
			curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; \
			echo "Installed Rust."; \
		fi; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		echo "Installed uv."; \
    else \
		echo "Updating uv..."; \
		uv self update || true; \
	fi

install-dependencies:
	@uv python install 3.11
	@uv sync --all-extras --all-groups --python 3.11

install-pre-commit:
	@uv run pre-commit install
	@uv run pre-commit autoupdate

check:
	@uv run pre-commit run --all-files
