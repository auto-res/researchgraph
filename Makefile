.PHONY: bump_version build publish

package:
	git add .
	git commit -m "style: bump version"
	uv run bumpversion patch
	uv build
	uvx twine upload dist/*
	rm -rf dist/*
