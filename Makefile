PYTHON ?= python

.PHONY: install train play lint lint-fix format

install:
	$(PYTHON) -m pip install -r requirements.txt

train:
	$(PYTHON) train_agent.py

play:
	$(PYTHON) play_trained.py

lint:


lint-fix:
	ruff check . --fix

format:
	ruff format .

