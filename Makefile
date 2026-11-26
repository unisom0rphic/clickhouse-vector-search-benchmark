.PHONY: up down logs preflight inspect prepare load smoke run analyze reset

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f clickhouse

preflight:
	python benchmark.py preflight

inspect:
	python benchmark.py inspect-data

prepare:
	python benchmark.py prepare

load:
	python benchmark.py load

smoke:
	python benchmark.py smoke

run:
	python benchmark.py run --resume

analyze:
	python benchmark.py analyze

reset:
	python benchmark.py reset-tables --yes
