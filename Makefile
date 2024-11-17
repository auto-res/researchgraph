.PHONY: bump_version build publish

version:
	uv run bumpversion patch
	git add .
	git commit -m "style: bump version"

build:
	uv build

publish:
	uvx twine upload dist/*
	rm -rf dist/*
