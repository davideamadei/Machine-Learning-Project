VERSION?=0.2.0

.PHONY: clean docs format

docs:
	pydoctor \
	--project-name=Machine-Learning-Project	\
	--project-version=$(VERSION) \
	--project-url=https://github.com/davideamadei/Machine-Learning-Project/ddnn/ \
	--make-html \
	--html-output=docs \
	--project-base-dir="ddnn" \
	--docformat=numpy \
	--intersphinx=https://docs.python.org/3/objects.inv \
	./ddnn

format:
	black .

clean:
	rm -rf docs/* __pycache__/*
	python3 -Bc "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.py[co]')]"
	python3 -Bc "import pathlib; [p.rmdir() for p in pathlib.Path('.').rglob('__pycache__')]"