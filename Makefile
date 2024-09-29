.PHONY: bump_version build publish

version:
	uv run bumpversion patch

build:
	uv build

publish:
	uvx twine upload dist/* && rm -rf dist/*
