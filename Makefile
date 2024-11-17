.PHONY: bump_version build publish

package:
	uv run bumpversion patch
	git add .
	git commit -m "style: bump version"
	uv build
	uvx twine upload dist/*
	rm -rf dist/*
