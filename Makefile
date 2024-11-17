.PHONY: bump_version build publish

version:
	git add .
	git commit -m "style: bump version"
	uv run bumpversion patch
	uv build
	uvx twine upload dist/*
	rm -rf dist/*

build:
	uv build

publish:
	uvx twine upload dist/*
	rm -rf dist/*
