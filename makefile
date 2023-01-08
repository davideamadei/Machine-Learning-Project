VERSION?=0.1.0

.PHONY: clean docs format

docs:
	pydoctor \
    --project-name=Machine-Learning-Project	\
    --project-version=$(VERSION) \
    --project-url=https://github.com/davideamadei/Machine-Learning-Project/ \
    --make-html \
    --html-output=docs/api \
    --project-base-dir="." \
    --docformat=numpy \
    --intersphinx=https://docs.python.org/3/objects.inv \
    .

format:
	black .

clean:
	rm -rf docs/* __pycache__/* 