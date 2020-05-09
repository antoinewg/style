install:
	pip3 install -r requirements.txt

freeze:
	pip3 freeze > requirements.txt

lint:
	black .

unitest:
	python3 -m pytest

train:
	@python src/main.py

board:
	@tensorboard --logdir logs/scalars

clear:
	@rm -rf logs/scalars
